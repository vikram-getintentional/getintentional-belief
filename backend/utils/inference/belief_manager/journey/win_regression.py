from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

WIN_MODEL_CATEGORICAL = [
    "industry",
    "revenue_range",
    "employee_range",
    "geography",
    "funding_stage",
    "primary_segment",
    "dominant_asset_category",
    "primary_channel",
]

WIN_MODEL_NUMERIC = [
    "critical_transition_count",
    "off_path_ratio",
    "journey_duration_days",
    "persona_activation",
    "asset_touch_ratio",
    "channel_diversity",
    "unique_stage_count",
    "asset_success_score",
    "engagement_depth",
    "segment_count",
]


def _normalize_category_value(value: Any) -> str:
    if value is None:
        return "Unknown"
    if isinstance(value, str):
        cleaned = value.strip()
        return cleaned or "Unknown"
    return str(value)


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def build_win_regression_summary(
    feature_rows: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    if not feature_rows:
        return None

    feature_cols = WIN_MODEL_CATEGORICAL + WIN_MODEL_NUMERIC
    records: List[Dict[str, Any]] = []
    for row in feature_rows:
        account_id = row.get("account_id")
        if not account_id:
            continue
        entry = {
            "account_id": str(account_id),
            "account_name": row.get("account_name"),
            "label": row.get("label"),
            "label_raw": row.get("label_raw"),
        }
        features = row.get("feature_values") or {}
        for col in feature_cols:
            entry[col] = features.get(col)
        records.append(entry)

    if not records:
        return None

    df = pd.DataFrame(records)
    if df.empty:
        return None

    for col in WIN_MODEL_CATEGORICAL:
        df[col] = df[col].apply(_normalize_category_value)
    for col in WIN_MODEL_NUMERIC:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    train_mask = df["label"].notna()
    train_df = df[train_mask]
    sample_size = int(train_df.shape[0])
    wins = int((train_df["label"] == 1).sum())
    losses = int((train_df["label"] == 0).sum())

    model_payload: Optional[Dict[str, Any]] = None
    metrics_payload: Optional[Dict[str, Any]] = None
    feature_coeffs: List[Dict[str, float]] = []

    if train_df["label"].nunique() >= 2 and sample_size >= 8:
        categorical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        numeric_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        preprocessor = ColumnTransformer(
            transformers=[
                ("cat", categorical_transformer, WIN_MODEL_CATEGORICAL),
                ("num", numeric_transformer, WIN_MODEL_NUMERIC),
            ]
        )
        clf = LogisticRegression(max_iter=500, class_weight="balanced")
        pipeline = Pipeline(
            steps=[
                ("pre", preprocessor),
                ("clf", clf),
            ]
        )
        pipeline.fit(train_df[feature_cols], train_df["label"])
        probabilities = pipeline.predict_proba(df[feature_cols])[:, 1]
        df["win_probability"] = probabilities

        if sample_size >= 5:
            train_pred = pipeline.predict_proba(train_df[feature_cols])[:, 1]
            accuracy = accuracy_score(
                train_df["label"], (train_pred >= 0.5).astype(int)
            )
            try:
                auc = roc_auc_score(train_df["label"], train_pred)
            except ValueError:
                auc = None
            metrics_payload = {
                "accuracy": round(float(accuracy), 4),
                "roc_auc": round(float(auc), 4) if auc is not None else None,
            }

        pre = pipeline.named_steps["pre"]
        feature_names = list(pre.get_feature_names_out())
        coefficients = pipeline.named_steps["clf"].coef_[0]
        feature_coeffs = [
            {"feature": name, "coefficient": float(coef)}
            for name, coef in zip(feature_names, coefficients)
        ]
        feature_importances = sorted(
            feature_coeffs,
            key=lambda item: abs(item["coefficient"]),
            reverse=True,
        )

        numeric_info: Dict[str, Dict[str, float]] = {}
        cat_info: Dict[str, Dict[str, Any]] = {}
        num_transformer = pre.named_transformers_.get("num")
        if num_transformer:
            num_imputer = num_transformer.named_steps.get("imputer")
            scaler = num_transformer.named_steps.get("scaler")
            stats = list(getattr(num_imputer, "statistics_", [0.0] * len(WIN_MODEL_NUMERIC)))
            means = list(getattr(scaler, "mean_", [0.0] * len(WIN_MODEL_NUMERIC)))
            scales = list(getattr(scaler, "scale_", [1.0] * len(WIN_MODEL_NUMERIC)))
            for idx, col in enumerate(WIN_MODEL_NUMERIC):
                numeric_info[col] = {
                    "impute": float(stats[idx]) if idx < len(stats) else 0.0,
                    "mean": float(means[idx]) if idx < len(means) else 0.0,
                    "scale": float(scales[idx]) if idx < len(scales) and scales[idx] not in (0, None) else 1.0,
                }
        cat_transformer = pre.named_transformers_.get("cat")
        if cat_transformer:
            cat_imputer = cat_transformer.named_steps.get("imputer")
            encoder = cat_transformer.named_steps.get("encoder")
            stats = list(getattr(cat_imputer, "statistics_", []))
            categories = list(getattr(encoder, "categories_", []))
            for idx, col in enumerate(WIN_MODEL_CATEGORICAL):
                impute_val = None
                if idx < len(stats):
                    impute_val = _normalize_category_value(stats[idx])
                cats = []
                if idx < len(categories):
                    cats = [_normalize_category_value(cat) for cat in categories[idx]]
                cat_info[col] = {
                    "impute": impute_val,
                    "categories": cats,
                }

        model_payload = {
            "coefficients": feature_coeffs,
            "intercept": float(pipeline.named_steps["clf"].intercept_[0]),
            "numeric_features": numeric_info,
            "categorical_features": cat_info,
        }
        feature_ranking = feature_importances
    else:
        df["win_probability"] = None
        feature_ranking = []

    account_feature_payload: Dict[str, Dict[str, Any]] = {}
    for record in df.to_dict(orient="records"):
        account_id = record.get("account_id")
        if not account_id:
            continue
        feature_values = {col: record.get(col) for col in feature_cols}
        probability = record.get("win_probability")
        account_feature_payload[str(account_id)] = {
            "account_id": str(account_id),
            "account_name": record.get("account_name"),
            "label": record.get("label_raw"),
            "win_probability": float(probability) if probability is not None else None,
            "feature_values": feature_values,
        }

    return {
        "sample_size": sample_size,
        "class_balance": {"won": wins, "lost": losses},
        "metrics": metrics_payload,
        "model": model_payload,
        "feature_importances": feature_ranking,
        "account_features": account_feature_payload,
    }


class WinProbabilityCalibrator:
    def __init__(
        self,
        model_payload: Dict[str, Any],
        account_features: Dict[str, Dict[str, Any]],
    ) -> None:
        self._coefficients: Dict[str, float] = {
            str(item.get("feature")): float(item.get("coefficient", 0.0))
            for item in model_payload.get("coefficients") or []
        }
        self._intercept = float(model_payload.get("intercept") or 0.0)
        self._numeric = model_payload.get("numeric_features") or {}
        self._categorical = model_payload.get("categorical_features") or {}
        self._account_features = account_features or {}

    @classmethod
    def from_summary(cls, summary: Dict[str, Any]) -> Optional["WinProbabilityCalibrator"]:
        if not summary:
            return None
        model_payload = summary.get("model")
        account_features = summary.get("account_features") or {}
        if not model_payload:
            return None
        return cls(model_payload, account_features)

    def has_account(self, account_id: str) -> bool:
        return str(account_id) in self._account_features

    def predict_probability(self, feature_values: Dict[str, Any]) -> Optional[float]:
        if not self._coefficients:
            return None
        score = self._intercept
        for col, info in self._numeric.items():
            coeff = self._coefficients.get(f"num__{col}")
            if coeff is None:
                continue
            value = feature_values.get(col)
            impute_val = info.get("impute", 0.0)
            mean_val = info.get("mean", 0.0)
            scale_val = info.get("scale") or 1.0
            try:
                numeric_value = float(value)
            except (TypeError, ValueError):
                numeric_value = float(impute_val or 0.0)
            score += coeff * ((numeric_value - float(mean_val or 0.0)) / float(scale_val or 1.0))
        for col, info in self._categorical.items():
            raw_value = feature_values.get(col)
            if raw_value in (None, "", []):
                raw_value = info.get("impute")
            normalized = _normalize_category_value(raw_value)
            for category in info.get("categories") or []:
                coeff = self._coefficients.get(f"cat__{col}_{category}")
                if coeff is not None and normalized == category:
                    score += coeff
        try:
            return 1.0 / (1.0 + math.exp(-score))
        except OverflowError:
            return 0.0 if score < 0 else 1.0

    def estimate_delta_bp(
        self,
        account_id: str,
        *,
        stage_index: int,
        stage_key: Optional[str],
        asset_signal: Optional[float] = None,
        base_delta_bp: Optional[float] = None,
    ) -> Optional[Dict[str, float]]:
        account_entry = self._account_features.get(str(account_id))
        if not account_entry:
            return None
        features = dict(account_entry.get("feature_values") or {})
        baseline = account_entry.get("win_probability")
        if baseline is None:
            baseline = self.predict_probability(features)
        if baseline is None:
            return None
        mutated = dict(features)
        mutated["critical_transition_count"] = (
            (mutated.get("critical_transition_count") or 0) + 1
        )
        mutated["unique_stage_count"] = max(
            mutated.get("unique_stage_count") or 0,
            max(stage_index + 1, 1),
        )
        mutated["asset_touch_ratio"] = min(
            1.0,
            (mutated.get("asset_touch_ratio") or 0.0) + 0.02,
        )
        if asset_signal is not None:
            mutated["asset_success_score"] = (
                mutated.get("asset_success_score") or 0.0
            ) + asset_signal
        simulated = self.predict_probability(mutated)
        if simulated is None:
            return None
        delta_bp = (simulated - baseline) * 10000.0
        if base_delta_bp is not None and abs(delta_bp) < 1.0:
            delta_bp = base_delta_bp
        delta_bp = max(-2000.0, min(4000.0, delta_bp))
        return {
            "delta_bp": float(delta_bp),
            "baseline_probability": float(baseline),
            "simulated_probability": float(simulated),
        }
