import datetime
import uuid
from typing import Any, Dict, List, Optional, Tuple

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.super_models.thesis_builder import (
    AttributeType,
    ThesisCall,
    ThesisCallEvidence,
    ThesisComponentPosterior,
    ThesisComponentPrior,
    ThesisComponentType,
    ThesisSegment,
    ThesisSegmentFeatureDefinition,
)
from backend.utils.thesis_builder import (
    assign_segment_by_rules,
    canonicalize_category_id,
    dirichlet_mean,
    drift_level_from_tv,
    infer_segment_posterior,
    total_variation_distance,
    update_dirichlet,
)

router = APIRouter(prefix="/api/thesis-builder")


class AttributeBinModel(BaseModel):
    id: str
    label: Optional[str] = None
    min: Optional[float] = None
    max: Optional[float] = None


class FeatureCreate(BaseModel):
    name: str
    key: str
    type: AttributeType
    bins: Optional[List[AttributeBinModel]] = None
    categories: Optional[List[str]] = None


class ThesisSegmentFeatureOut(BaseModel):
    id: str
    product_id: str
    name: str
    key: str
    type: AttributeType
    bins: Optional[List[AttributeBinModel]] = None
    categories: Optional[List[str]] = None
    created_at: datetime.datetime


class SegmentRuleCondition(BaseModel):
    feature_key: str
    op: str
    values: Optional[List[str]] = None


class SegmentRules(BaseModel):
    conditions: List[SegmentRuleCondition] = Field(default_factory=list)


class SegmentCreate(BaseModel):
    name: str
    description: Optional[str] = None
    rules: SegmentRules
    prior_weight: Optional[float] = 1.0


class ThesisSegmentOut(BaseModel):
    id: str
    product_id: str
    name: str
    description: Optional[str] = None
    rules: SegmentRules
    prior_weight: float
    created_at: datetime.datetime


class DistributionSummary(BaseModel):
    prior: Dict[str, float]
    posterior: Dict[str, float]
    drift: float
    drift_level: str


class AttributeDistributionSummary(DistributionSummary):
    feature_key: str


class PainDistributionSummary(DistributionSummary):
    parent_pain_id: Optional[str] = None


class ZmotDistributionSummary(DistributionSummary):
    parent_pain_id: str


class AspirationDistributionSummary(DistributionSummary):
    parent_pain_id: str


class BeliefDistributionSummary(DistributionSummary):
    parent_pain_id: str
    parent_aspiration_id: str


class SegmentThesisDistributions(BaseModel):
    attributes: List[AttributeDistributionSummary] = Field(default_factory=list)
    pains: List[PainDistributionSummary] = Field(default_factory=list)
    zmots: List[ZmotDistributionSummary] = Field(default_factory=list)
    aspirations: List[AspirationDistributionSummary] = Field(default_factory=list)
    beliefs: List[BeliefDistributionSummary] = Field(default_factory=list)


class SegmentThesisSummaryResponse(BaseModel):
    segment: ThesisSegmentOut
    distributions: SegmentThesisDistributions


class CallPainEvidence(BaseModel):
    pain_id: str
    intensity: float = Field(gt=0.0, le=1.0)


class CallZmotEvidence(BaseModel):
    pain_id: str
    zmot_id: str
    intensity: float = Field(gt=0.0, le=1.0)


class CallAspirationEvidence(BaseModel):
    pain_id: str
    aspiration_id: str
    intensity: float = Field(gt=0.0, le=1.0)


class CallBeliefEvidence(BaseModel):
    pain_id: str
    aspiration_id: str
    belief_level: str
    intensity: float = Field(gt=0.0, le=1.0)


class CallCanonicalSummary(BaseModel):
    segment_hint_id: Optional[str] = None
    pains: List[CallPainEvidence] = Field(default_factory=list)
    zmots: List[CallZmotEvidence] = Field(default_factory=list)
    aspirations: List[CallAspirationEvidence] = Field(default_factory=list)
    beliefs: List[CallBeliefEvidence] = Field(default_factory=list)


class StakeholderRecord(BaseModel):
    role: Optional[str] = None
    title: Optional[str] = None
    department: Optional[str] = None


class CallIngestionPayload(BaseModel):
    account_id: Optional[str] = None
    external_call_id: Optional[str] = None
    attribute_bins: Dict[str, str]
    stakeholders: List[StakeholderRecord] = Field(default_factory=list)
    canonical_summary: CallCanonicalSummary = Field(default_factory=CallCanonicalSummary)
    raw_summary: Optional[Dict[str, Any]] = None
    transcript_text: Optional[str] = None


class DriftAlert(BaseModel):
    component_type: ThesisComponentType
    parent_pain_id: Optional[str] = None
    parent_aspiration_id: Optional[str] = None
    feature_key: Optional[str] = None
    drift: float
    drift_level: str
    message: str


class UpdatedSegmentDrift(BaseModel):
    segment_id: str
    drift_alerts: List[DriftAlert] = Field(default_factory=list)


class CallIngestionResponse(BaseModel):
    call_id: str
    segment_posterior: Dict[str, float]
    updated_segments: List[UpdatedSegmentDrift]


class CallSummaryItem(BaseModel):
    call_id: str
    account_id: Optional[str]
    external_call_id: Optional[str]
    attribute_bins: Dict[str, str]
    segment_id: Optional[str]
    created_at: datetime.datetime


class AcceptDriftComponent(BaseModel):
    component_type: ThesisComponentType
    parent_pain_id: Optional[str] = None
    parent_aspiration_id: Optional[str] = None
    feature_key: Optional[str] = None


class AcceptDriftRequest(BaseModel):
    components: List[AcceptDriftComponent]


class CallTranscriptPayload(BaseModel):
    account_id: Optional[str] = None
    external_call_id: Optional[str] = None
    transcript_text: str
    attribute_bins: Dict[str, str] = Field(default_factory=dict)


def _serialize_feature(feature: ThesisSegmentFeatureDefinition) -> ThesisSegmentFeatureOut:
    return ThesisSegmentFeatureOut(
        id=feature.id,
        product_id=feature.product_id,
        name=feature.name,
        key=feature.key,
        type=feature.type,
        bins=feature.bins,
        categories=feature.categories,
        created_at=feature.created_at,
    )


def _serialize_segment(segment: ThesisSegment) -> ThesisSegmentOut:
    return ThesisSegmentOut(
        id=segment.id,
        product_id=segment.product_id,
        name=segment.name,
        description=segment.description,
        rules=SegmentRules(**segment.rules),
        prior_weight=segment.prior_weight or 1.0,
        created_at=segment.created_at,
    )


def _get_segment_or_404(db: Session, product_id: str, segment_id: str) -> ThesisSegment:
    segment = (
        db.query(ThesisSegment)
        .filter(ThesisSegment.product_id == product_id, ThesisSegment.id == segment_id)
        .first()
    )
    if not segment:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Segment not found")
    return segment


def _get_prior_alphas(
    db: Session,
    product_id: str,
    segment_id: str,
    component_type: ThesisComponentType,
    parent_pain_id: Optional[str] = None,
    parent_aspiration_id: Optional[str] = None,
    feature_key: Optional[str] = None,
) -> Dict[str, float]:
    query = (
        db.query(ThesisComponentPrior)
        .filter(
            ThesisComponentPrior.product_id == product_id,
            ThesisComponentPrior.segment_id == segment_id,
            ThesisComponentPrior.component_type == component_type,
        )
    )
    if parent_pain_id is not None:
        query = query.filter(ThesisComponentPrior.parent_pain_id == parent_pain_id)
    else:
        query = query.filter(ThesisComponentPrior.parent_pain_id.is_(None))
    if parent_aspiration_id is not None:
        query = query.filter(ThesisComponentPrior.parent_aspiration_id == parent_aspiration_id)
    else:
        query = query.filter(ThesisComponentPrior.parent_aspiration_id.is_(None))
    if feature_key is not None:
        query = query.filter(ThesisComponentPrior.feature_key == feature_key)
    else:
        query = query.filter(ThesisComponentPrior.feature_key.is_(None))
    prior = query.first()
    return prior.alphas if prior and prior.alphas else {}


def _compute_distribution_metrics(
    prior_alphas: Dict[str, float], posterior_alphas: Dict[str, float]
) -> Tuple[Dict[str, float], Dict[str, float], float, str]:
    prior_mean = dirichlet_mean(prior_alphas)
    posterior_mean = dirichlet_mean(posterior_alphas)
    if not prior_mean and posterior_mean:
        prior_mean = dict(posterior_mean)
    drift = total_variation_distance(prior_mean, posterior_mean)
    drift_level = drift_level_from_tv(drift)
    return prior_mean, posterior_mean, drift, drift_level


def _build_component_posters(
    db: Session,
    product_id: str,
    segment_id: str,
    component_type: ThesisComponentType,
    *,
    model_cls: Any,
    include_feature_key: bool = False,
    include_parent_pain: bool = False,
    include_parent_aspiration: bool = False,
) -> List[Any]:
    rows = (
        db.query(ThesisComponentPosterior)
        .filter(
            ThesisComponentPosterior.product_id == product_id,
            ThesisComponentPosterior.segment_id == segment_id,
            ThesisComponentPosterior.component_type == component_type,
        )
        .all()
    )
    entries = []
    for row in rows:
        prior_alphas = _get_prior_alphas(
            db,
            product_id,
            segment_id,
            component_type,
            parent_pain_id=row.parent_pain_id,
            parent_aspiration_id=row.parent_aspiration_id,
            feature_key=row.feature_key,
        )
        prior_mean, posterior_mean, drift, drift_level = _compute_distribution_metrics(
            prior_alphas, row.alphas or {}
        )
        payload = {
            "prior": prior_mean,
            "posterior": posterior_mean,
            "drift": drift,
            "drift_level": drift_level,
        }
        if include_feature_key:
            payload["feature_key"] = row.feature_key or ""
        if include_parent_pain:
            payload["parent_pain_id"] = row.parent_pain_id
        if include_parent_aspiration:
            payload["parent_aspiration_id"] = row.parent_aspiration_id
        entries.append(model_cls(**payload))
    return entries


def _get_or_create_posterior(
    db: Session,
    product_id: str,
    segment_id: str,
    component_type: ThesisComponentType,
    *,
    parent_pain_id: Optional[str] = None,
    parent_aspiration_id: Optional[str] = None,
    feature_key: Optional[str] = None,
) -> ThesisComponentPosterior:
    query = (
        db.query(ThesisComponentPosterior)
        .filter(
            ThesisComponentPosterior.product_id == product_id,
            ThesisComponentPosterior.segment_id == segment_id,
            ThesisComponentPosterior.component_type == component_type,
        )
    )
    if parent_pain_id is not None:
        query = query.filter(ThesisComponentPosterior.parent_pain_id == parent_pain_id)
    else:
        query = query.filter(ThesisComponentPosterior.parent_pain_id.is_(None))
    if parent_aspiration_id is not None:
        query = query.filter(ThesisComponentPosterior.parent_aspiration_id == parent_aspiration_id)
    else:
        query = query.filter(ThesisComponentPosterior.parent_aspiration_id.is_(None))
    if feature_key is not None:
        query = query.filter(ThesisComponentPosterior.feature_key == feature_key)
    else:
        query = query.filter(ThesisComponentPosterior.feature_key.is_(None))
    row = query.first()
    if row:
        return row
    new_row = ThesisComponentPosterior(
        id=str(uuid.uuid4()),
        product_id=product_id,
        segment_id=segment_id,
        component_type=component_type,
        parent_pain_id=parent_pain_id,
        parent_aspiration_id=parent_aspiration_id,
        feature_key=feature_key,
        alphas={},
        last_updated_at=datetime.datetime.utcnow(),
    )
    db.add(new_row)
    db.flush()
    return new_row


def _normalize_intensity(value: float) -> float:
    return max(0.0, min(1.0, value))


def _update_component_distribution(
    db: Session,
    product_id: str,
    segment_id: str,
    component_type: ThesisComponentType,
    evidence: Dict[str, float],
    *,
    parent_pain_id: Optional[str] = None,
    parent_aspiration_id: Optional[str] = None,
    feature_key: Optional[str] = None,
) -> Tuple[Dict[str, float], Dict[str, float], float, str]:
    posterior = _get_or_create_posterior(
        db,
        product_id,
        segment_id,
        component_type,
        parent_pain_id=parent_pain_id,
        parent_aspiration_id=parent_aspiration_id,
        feature_key=feature_key,
    )
    prior_alphas = _get_prior_alphas(
        db,
        product_id,
        segment_id,
        component_type,
        parent_pain_id=parent_pain_id,
        parent_aspiration_id=parent_aspiration_id,
        feature_key=feature_key,
    )
    posterior.alphas = update_dirichlet(posterior.alphas or {}, evidence)
    posterior.last_updated_at = datetime.datetime.utcnow()
    db.add(posterior)
    metrics = _compute_distribution_metrics(prior_alphas, posterior.alphas or {})
    return metrics


def _alert_message_for_component(component_type: ThesisComponentType) -> str:
    if component_type == ThesisComponentType.PAIN:
        return "Primary pain distribution for this segment is drifting away from your original thesis."
    if component_type == ThesisComponentType.ZMOT:
        return "ZMOT / ZMOT evidence is drifting from your baseline pain narrative."
    if component_type == ThesisComponentType.ASPIRATION:
        return "Aspirations tied to this pain are shifting relative to what you believed."
    if component_type == ThesisComponentType.BELIEF:
        return "Belief levels are drifting compared to what your thesis predicts."
    return "Attribute distributions are drifting relative to the expected customer segment."


def _maybe_record_drift_alert(
    alerts: Dict[str, List[DriftAlert]],
    segment_id: str,
    component_type: ThesisComponentType,
    drift: float,
    drift_level: str,
    *,
    parent_pain_id: Optional[str] = None,
    parent_aspiration_id: Optional[str] = None,
    feature_key: Optional[str] = None,
) -> None:
    if drift_level == "stable":
        return
    alerts.setdefault(segment_id, [])
    alerts[segment_id].append(
        DriftAlert(
            component_type=component_type,
            parent_pain_id=parent_pain_id,
            parent_aspiration_id=parent_aspiration_id,
            feature_key=feature_key,
            drift=drift,
            drift_level=drift_level,
            message=_alert_message_for_component(component_type),
        )
    )


@router.get("/{product_id}/features", response_model=List[ThesisSegmentFeatureOut])
def list_features(product_id: str, db: Session = Depends(get_db)):
    features = (
        db.query(ThesisSegmentFeatureDefinition)
        .filter(ThesisSegmentFeatureDefinition.product_id == product_id)
        .all()
    )
    return [_serialize_feature(feature) for feature in features]


@router.post("/{product_id}/features", response_model=ThesisSegmentFeatureOut)
def create_feature(product_id: str, payload: FeatureCreate, db: Session = Depends(get_db)):
    feature = ThesisSegmentFeatureDefinition(
        id=str(uuid.uuid4()),
        product_id=product_id,
        name=payload.name,
        key=payload.key,
        type=payload.type,
        bins=[bin.dict() for bin in payload.bins] if payload.bins else None,
        categories=payload.categories,
    )
    db.add(feature)
    db.commit()
    db.refresh(feature)
    return _serialize_feature(feature)


@router.get("/{product_id}/segments", response_model=List[ThesisSegmentOut])
def list_segments(product_id: str, db: Session = Depends(get_db)):
    segments = (
        db.query(ThesisSegment)
        .filter(ThesisSegment.product_id == product_id)
        .order_by(ThesisSegment.created_at.desc())
        .all()
    )
    return [_serialize_segment(segment) for segment in segments]


@router.post("/{product_id}/segments", response_model=ThesisSegmentOut)
def create_segment(product_id: str, payload: SegmentCreate, db: Session = Depends(get_db)):
    segment = ThesisSegment(
        id=str(uuid.uuid4()),
        product_id=product_id,
        name=payload.name,
        description=payload.description,
        rules=payload.rules.dict(),
        prior_weight=payload.prior_weight or 1.0,
    )
    db.add(segment)
    db.commit()
    db.refresh(segment)
    return _serialize_segment(segment)


@router.get(
    "/{product_id}/segments/{segment_id}/summary",
    response_model=SegmentThesisSummaryResponse,
)
def get_segment_summary(
    product_id: str, segment_id: str, db: Session = Depends(get_db)
):
    segment = _get_segment_or_404(db, product_id, segment_id)
    distributions = SegmentThesisDistributions(
        attributes=_build_component_posters(
            db,
            product_id,
            segment_id,
            ThesisComponentType.ATTRIBUTE,
            model_cls=AttributeDistributionSummary,
            include_feature_key=True,
        ),
        pains=_build_component_posters(
            db,
            product_id,
            segment_id,
            ThesisComponentType.PAIN,
            model_cls=PainDistributionSummary,
            include_parent_pain=True,
        ),
        zmots=_build_component_posters(
            db,
            product_id,
            segment_id,
            ThesisComponentType.ZMOT,
            model_cls=ZmotDistributionSummary,
            include_parent_pain=True,
        ),
        aspirations=_build_component_posters(
            db,
            product_id,
            segment_id,
            ThesisComponentType.ASPIRATION,
            model_cls=AspirationDistributionSummary,
            include_parent_pain=True,
        ),
        beliefs=_build_component_posters(
            db,
            product_id,
            segment_id,
            ThesisComponentType.BELIEF,
            model_cls=BeliefDistributionSummary,
            include_parent_pain=True,
            include_parent_aspiration=True,
        ),
    )
    return SegmentThesisSummaryResponse(segment=_serialize_segment(segment), distributions=distributions)


@router.post("/{product_id}/calls", response_model=CallIngestionResponse)
def ingest_thesis_call(
    product_id: str, payload: CallIngestionPayload, db: Session = Depends(get_db)
) -> CallIngestionResponse:
    segments = (
        db.query(ThesisSegment)
        .filter(ThesisSegment.product_id == product_id)
        .order_by(ThesisSegment.created_at.desc())
        .all()
    )
    segment_map = {segment.id: segment for segment in segments}
    canonical_bins = {
        key: canonicalize_category_id(str(value))
        for key, value in payload.attribute_bins.items()
    }
    assigned_segments = assign_segment_by_rules(segments, canonical_bins)
    target_segments: List[str] = []
    hint_segment = payload.canonical_summary.segment_hint_id
    if hint_segment and hint_segment in segment_map:
        target_segments.append(hint_segment)
    for segment_id in assigned_segments:
        if segment_id and segment_id not in target_segments and segment_id in segment_map:
            target_segments.append(segment_id)
    if not target_segments and segments:
        target_segments.append(segments[0].id)
    selected_segment_id = target_segments[0] if target_segments else None

    stakeholders = [
        stakeholder.dict(exclude_none=True) for stakeholder in payload.stakeholders
    ]
    raw_summary_store = dict(payload.raw_summary or {})
    raw_summary_store["canonical_summary"] = payload.canonical_summary.dict()

    call = ThesisCall(
        id=str(uuid.uuid4()),
        product_id=product_id,
        account_id=payload.account_id,
        external_call_id=payload.external_call_id,
        attribute_bins=payload.attribute_bins,
        stakeholders=stakeholders,
        raw_summary=raw_summary_store,
        segment_id=selected_segment_id,
        transcript_text=payload.transcript_text,
    )
    db.add(call)
    db.flush()

    alerts: Dict[str, List[DriftAlert]] = {}

    def _store_evidence(
        component_type: ThesisComponentType,
        category_id: str,
        *,
        parent_pain_id: Optional[str] = None,
        parent_aspiration_id: Optional[str] = None,
        feature_key: Optional[str] = None,
        intensity: float = 1.0,
    ) -> None:
        evidence = ThesisCallEvidence(
            id=str(uuid.uuid4()),
            call_id=call.id,
            product_id=product_id,
            segment_id=selected_segment_id,
            component_type=component_type,
            category_id=category_id,
            parent_pain_id=parent_pain_id,
            parent_aspiration_id=parent_aspiration_id,
            feature_key=feature_key,
            intensity=intensity,
        )
        db.add(evidence)

    def _apply_updates_for_segment(segment_id: str) -> None:
        for feature_key, canonical_value in canonical_bins.items():
            prior, posterior, drift, drift_level = _update_component_distribution(
                db,
                product_id,
                segment_id,
                ThesisComponentType.ATTRIBUTE,
                {canonical_value: 1.0},
                feature_key=feature_key,
            )
            _maybe_record_drift_alert(
                alerts,
                segment_id,
                ThesisComponentType.ATTRIBUTE,
                drift,
                drift_level,
                feature_key=feature_key,
            )
        for pain in payload.canonical_summary.pains:
            normalized_pain = canonicalize_category_id(pain.pain_id)
            intensity = _normalize_intensity(pain.intensity)
            _apply_and_alert(
                segment_id,
                ThesisComponentType.PAIN,
                normalized_pain,
                intensity,
            )
        for zmot in payload.canonical_summary.zmots:
            normalized_pain = canonicalize_category_id(zmot.pain_id)
            normalized_value = canonicalize_category_id(zmot.zmot_id)
            intensity = _normalize_intensity(zmot.intensity)
            _apply_and_alert(
                segment_id,
                ThesisComponentType.ZMOT,
                normalized_value,
                intensity,
                parent_pain_id=normalized_pain,
            )
        for aspiration in payload.canonical_summary.aspirations:
            normalized_pain = canonicalize_category_id(aspiration.pain_id)
            normalized_value = canonicalize_category_id(aspiration.aspiration_id)
            intensity = _normalize_intensity(aspiration.intensity)
            _apply_and_alert(
                segment_id,
                ThesisComponentType.ASPIRATION,
                normalized_value,
                intensity,
                parent_pain_id=normalized_pain,
            )
        for belief in payload.canonical_summary.beliefs:
            normalized_pain = canonicalize_category_id(belief.pain_id)
            normalized_aspiration = canonicalize_category_id(belief.aspiration_id)
            normalized_belief = canonicalize_category_id(belief.belief_level)
            intensity = _normalize_intensity(belief.intensity)
            _apply_and_alert(
                segment_id,
                ThesisComponentType.BELIEF,
                normalized_belief,
                intensity,
                parent_pain_id=normalized_pain,
                parent_aspiration_id=normalized_aspiration,
            )

    def _apply_and_alert(
        segment_id: str,
        component_type: ThesisComponentType,
        category_id: str,
        intensity: float,
        *,
        parent_pain_id: Optional[str] = None,
        parent_aspiration_id: Optional[str] = None,
    ) -> None:
        prior, posterior, drift, drift_level = _update_component_distribution(
            db,
            product_id,
            segment_id,
            component_type,
            {category_id: intensity},
            parent_pain_id=parent_pain_id,
            parent_aspiration_id=parent_aspiration_id,
        )
        _maybe_record_drift_alert(
            alerts,
            segment_id,
            component_type,
            drift,
            drift_level,
            parent_pain_id=parent_pain_id,
            parent_aspiration_id=parent_aspiration_id,
        )

    for segment_id in target_segments:
        _apply_updates_for_segment(segment_id)

    for feature_key, original_value in payload.attribute_bins.items():
        canonical_value = canonicalize_category_id(original_value)
        _store_evidence(
            ThesisComponentType.ATTRIBUTE,
            canonical_value,
            feature_key=feature_key,
            intensity=1.0,
        )
    for pain in payload.canonical_summary.pains:
        _store_evidence(
            ThesisComponentType.PAIN,
            canonicalize_category_id(pain.pain_id),
            intensity=_normalize_intensity(pain.intensity),
        )
    for zmot in payload.canonical_summary.zmots:
        _store_evidence(
            ThesisComponentType.ZMOT,
            canonicalize_category_id(zmot.zmot_id),
            parent_pain_id=canonicalize_category_id(zmot.pain_id),
            intensity=_normalize_intensity(zmot.intensity),
        )
    for aspiration in payload.canonical_summary.aspirations:
        _store_evidence(
            ThesisComponentType.ASPIRATION,
            canonicalize_category_id(aspiration.aspiration_id),
            parent_pain_id=canonicalize_category_id(aspiration.pain_id),
            intensity=_normalize_intensity(aspiration.intensity),
        )
    for belief in payload.canonical_summary.beliefs:
        _store_evidence(
            ThesisComponentType.BELIEF,
            canonicalize_category_id(belief.belief_level),
            parent_pain_id=canonicalize_category_id(belief.pain_id),
            parent_aspiration_id=canonicalize_category_id(belief.aspiration_id),
            intensity=_normalize_intensity(belief.intensity),
        )

    attribute_posteriors = {
        (row.segment_id, row.feature_key or ""): row.alphas or {}
        for row in db.query(ThesisComponentPosterior)
        .filter(
            ThesisComponentPosterior.product_id == product_id,
            ThesisComponentPosterior.component_type == ThesisComponentType.ATTRIBUTE,
        )
        .all()
        if row.feature_key
    }
    segment_priors = {
        segment.id: segment.prior_weight or 1.0 for segment in segments
    }
    segment_posterior = infer_segment_posterior(
        segments,
        segment_priors,
        canonical_bins,
        attribute_posteriors,
    )

    updated_segments = [
        UpdatedSegmentDrift(segment_id=segment_id, drift_alerts=alerts.get(segment_id, []))
        for segment_id in target_segments
        if alerts.get(segment_id)
    ]

    db.commit()

    return CallIngestionResponse(
        call_id=call.id,
        segment_posterior=segment_posterior,
        updated_segments=updated_segments,
    )


@router.get("/{product_id}/calls", response_model=List[CallSummaryItem])
def list_calls(product_id: str, db: Session = Depends(get_db)):
    calls = (
        db.query(ThesisCall)
        .filter(ThesisCall.product_id == product_id)
        .order_by(ThesisCall.created_at.desc())
        .limit(25)
        .all()
    )
    return [
        CallSummaryItem(
            call_id=call.id,
            account_id=call.account_id,
            external_call_id=call.external_call_id,
            attribute_bins=call.attribute_bins or {},
            segment_id=call.segment_id,
            created_at=call.created_at,
        )
        for call in calls
    ]


@router.post("/{product_id}/calls/transcript")
def save_call_transcript(
    product_id: str, payload: CallTranscriptPayload, db: Session = Depends(get_db)
):
    call = ThesisCall(
        id=str(uuid.uuid4()),
        product_id=product_id,
        account_id=payload.account_id,
        external_call_id=payload.external_call_id,
        attribute_bins=payload.attribute_bins or {},
        stakeholders=[],
        raw_summary={"transcript_only": True},
        transcript_text=payload.transcript_text,
    )
    db.add(call)
    db.commit()
    db.refresh(call)
    return {"call_id": call.id}


@router.post("/{product_id}/segments/{segment_id}/accept-drift")
def accept_drift(
    product_id: str,
    segment_id: str,
    payload: AcceptDriftRequest,
    db: Session = Depends(get_db),
):
    _get_segment_or_404(db, product_id, segment_id)
    updated_components = 0
    for component in payload.components:
        query = (
            db.query(ThesisComponentPosterior)
            .filter(
                ThesisComponentPosterior.product_id == product_id,
                ThesisComponentPosterior.segment_id == segment_id,
                ThesisComponentPosterior.component_type == component.component_type,
            )
        )
        if component.parent_pain_id is not None:
            query = query.filter(ThesisComponentPosterior.parent_pain_id == component.parent_pain_id)
        else:
            query = query.filter(ThesisComponentPosterior.parent_pain_id.is_(None))
        if component.parent_aspiration_id is not None:
            query = query.filter(
                ThesisComponentPosterior.parent_aspiration_id == component.parent_aspiration_id
            )
        else:
            query = query.filter(ThesisComponentPosterior.parent_aspiration_id.is_(None))
        if component.feature_key is not None:
            query = query.filter(ThesisComponentPosterior.feature_key == component.feature_key)
        else:
            query = query.filter(ThesisComponentPosterior.feature_key.is_(None))
        posterior = query.first()
        if not posterior:
            continue
        posterior.baseline_alphas = dict(posterior.alphas or {})
        posterior.baseline_at = datetime.datetime.utcnow()
        prior_query = (
            db.query(ThesisComponentPrior)
            .filter(
                ThesisComponentPrior.product_id == product_id,
                ThesisComponentPrior.segment_id == segment_id,
                ThesisComponentPrior.component_type == component.component_type,
            )
        )
        if component.parent_pain_id is not None:
            prior_query = prior_query.filter(ThesisComponentPrior.parent_pain_id == component.parent_pain_id)
        else:
            prior_query = prior_query.filter(ThesisComponentPrior.parent_pain_id.is_(None))
        if component.parent_aspiration_id is not None:
            prior_query = prior_query.filter(
                ThesisComponentPrior.parent_aspiration_id == component.parent_aspiration_id
            )
        else:
            prior_query = prior_query.filter(ThesisComponentPrior.parent_aspiration_id.is_(None))
        if component.feature_key is not None:
            prior_query = prior_query.filter(ThesisComponentPrior.feature_key == component.feature_key)
        else:
            prior_query = prior_query.filter(ThesisComponentPrior.feature_key.is_(None))
        prior = prior_query.first()
        if not prior:
            prior = ThesisComponentPrior(
                id=str(uuid.uuid4()),
                product_id=product_id,
                segment_id=segment_id,
                component_type=component.component_type,
                parent_pain_id=component.parent_pain_id,
                parent_aspiration_id=component.parent_aspiration_id,
                feature_key=component.feature_key,
                alphas=dict(posterior.alphas or {}),
            )
            db.add(prior)
        else:
            prior.alphas = dict(posterior.alphas or {})
        db.add(prior)
        updated_components += 1
    db.commit()
    return {"status": "ok", "updated_components": updated_components}
