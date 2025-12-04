import React, { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import {
  Alert,
  Box,
  Button,
  Card,
  CardContent,
  Chip,
  CircularProgress,
  Collapse,
  Divider,
  FormControl,
  Grid,
  InputLabel,
  MenuItem,
  Popover,
  Paper,
  Select,
  IconButton,
  Stack,
  Tab,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Tabs,
  Tooltip,
  Typography,
} from "@mui/material";
import type { ChipProps } from "@mui/material";
import InfoOutlinedIcon from "@mui/icons-material/InfoOutlined";
import PersonaEngagementCadence, {
  type PersonaEngagementPlan,
} from "../components/PersonaEngagementCadence";
import AssetCadenceTable, {
  type AssetCadenceRow,
} from "../components/AssetCadenceTable";

// ---------------------------------------------------------------------------
// Types matching backend payload
// ---------------------------------------------------------------------------

type PhaseProbabilities = Record<string, number>;

type PersonMatch = {
  match_id: string;
  person_id?: string | null;
  person_name?: string | null;
  person_title?: string | null;
  person_department?: string | null;
  person_seniority?: string | null;
  match_confidence?: number | null;
  source?: string | null;
  notes?: string | null;
  display_name?: string | null;
  last_seen_at?: string | null;
  engagement_count?: number | null;
  person_belief_level?: number | null;
  person_involvement_score?: number | null;
  committee_probability?: number | null;
  belief_phase_probs?: PhaseProbabilities | null;
  dominant_phase?: string | null;
};

type SegmentFilters = Record<string, string>;

type AccountZmotKeyword = {
  id?: string;
  label?: string;
  keyword?: string;
};

type AccountZmotObservable = {
  id?: string;
  label?: string;
  description?: string;
  channels?: string[];
};

type AccountZmotEvent = {
  zmot_event_id?: string;
  zmot_label?: string;
  pain_trigger_id?: string;
  pain_trigger_label?: string;
  pain_id?: string;
  pain_label?: string;
  boost?: number;
  trigger_boost?: number;
  observable_moments?: AccountZmotObservable[];
  keywords?: AccountZmotKeyword[];
  already_observed?: boolean;
};

type AccountZmotWatchPersona = {
  persona_id: string;
  persona_label?: string | null;
  events: AccountZmotEvent[];
};

type AccountZmotPortfolioEvent = AccountZmotEvent & {
  personas?: Array<{
    persona_id: string;
    persona_label?: string | null;
    already_observed?: boolean;
  }>;
};

type PersonaMeta = {
  id?: string;
  label?: string;
  title?: string;
  department?: string;
  seniority?: string;
  perceptibility?: number | null;
  proximity?: number | null;
  involvement?: number | null;
  activation?: number | null;
  belief_level?: number | null;
  expected_next_prob?: number | null;
  fatigue?: number | null;
  fatigue_reason?: string | null;
  priority_rank?: number | null;
  priority_score?: number | null;
  expected_in_deal_prob?: number | null;
  expected_in_deal_pct?: number | null;
  phase_probs?: PhaseProbabilities | null;
  dominant_phase?: string | null;
};

type PersonaTopPerson = {
  person_id?: string | null;
  display_name?: string | null;
  person_involvement_score?: number | null;
  person_belief_level?: number | null;
  committee_probability?: number | null;
  belief_phase_probs?: PhaseProbabilities | null;
  dominant_phase?: string | null;
};

type PersonaSummary = PersonaMeta & {
  id: string;
  order: number;
  path_probability?: number | null;
  matched_people?: PersonMatch[];
  matched_people_count?: number;
  people_names?: string[];
  has_person_match?: boolean;
  zmot_events?: AccountZmotEvent[];
  top_people?: PersonaTopPerson[];
};

type PersonaPath = {
  id: string;
  probability?: number | null;
  score?: number | null;
  is_primary?: boolean;
  personas: PersonaSummary[];
};

type JourneyStep = {
  t?: number;
  bucket?: string;
  predicted: string[];
  observed?: string | null;
  win_likelihood?: number | null;
  hit_at_1?: boolean;
  hit_at_3?: boolean;
};

type TypicalPathStep = {
  step: number;
  persona: string;
  highlight?: string;
  impact?: string;
  confidence?: number;
  reason?: string;
};

type CanonicalJourney = {
  typical_path?: TypicalPathStep[];
  average_duration_days?: number | null;
  common_paths?: string[];
  deviations?: string | null;
};

type ProductInsights = {
  journey_structure?: CanonicalJourney;
};

type Transition = {
  persona: PersonaSummary;
  stage_index: number;
  job?: { id: string; label: string };
  pains?: Array<{ id: string; label: string }>;
  capabilities?: Array<{ id: string; label: string }>;
  belief_transition?: {
    persona?: NodeRef;
    problem?: NodeRef;
    execution?: NodeRef;
    pain?: NodeRef | null;
    resolution?: NodeRef | null;
    narrative?: string;
    from_to?: BeliefEdge[];
    stage?: string;
    stage_label?: string;
  };
};

type AssetRec = {
  id?: string;
  name?: string;
  title?: string | null;
  format?: string | null;
  category?: string | null;
  category_label?: string | null;
  time_to_effect_days?: number | null;
  base_lift_bp?: number | null;
  metadata_complete?: boolean;
  target_personas?: string[];
  target_belief_stages?: string[];
  target_concerns?: string[];
  usage?: Record<string, any>;
  cost_tier?: string | null;
};

type ChannelRec = {
  id?: string;
  name?: string;
  type?: string | null;
  channel_type?: string | null;
  channel_type_label?: string | null;
  reach_score?: number | null;
  breadth_multiplier?: number | null;
  cadence_hint?: string | null;
  delivery_mode?: string | null;
  target_personas?: string[];
  target_belief_stages?: string[];
  target_concerns?: string[];
  usage?: Record<string, any>;
};

type ArsenalRationaleSegment = {
  asset_matches?: string[];
  channel_matches?: string[];
  account_segments?: string[];
};

type ArsenalRationaleBelief = {
  belief_tokens?: string[];
  pain_match?: string | null;
  channel_concern_match?: string | null;
  stage_tokens?: string[];
};

type ArsenalRationale = {
  segment?: ArsenalRationaleSegment;
  belief?: ArsenalRationaleBelief;
};

type NodeRef = {
  type?: string;
  id?: string;
  label?: string;
};

type BeliefEdge = {
  from_type?: string | null;
  from_id?: string | null;
  from_label?: string | null;
  to_type?: string | null;
  to_id?: string | null;
  to_label?: string | null;
};

type BeliefTransitionDetail = {
  persona?: NodeRef;
  problem?: NodeRef;
  execution?: NodeRef;
  pain?: NodeRef | null;
  resolution?: NodeRef | null;
  narrative?: string;
  from_to?: BeliefEdge[];
  stage?: string;
  stage_label?: string;
};

type ConversionFocus = {
  persona_label?: string;
  persona_focus?: string;
  stage?: string;
  stage_label?: string;
  focus_label?: string;
  label?: string;
  expected_delta_bp?: number;
  confidence?: number;
  belief_transition?: BeliefTransitionDetail;
  asset_table?: AggregatedFocusAssetRow[];
  avg_duration_days?: number | null;
  time_to_effect_days?: number | null;
  avg_effort_required?: number | null;
  avg_effort_pct?: number | null;
  avg_belief_conversion?: number | null;
  segment_filters?: SegmentFilters;
  segment_summary?: string | null;
  persona_meta?: PersonaMeta;
  persona_descriptor?: string;
  persona_descriptor_counts?: PersonaDescriptorCount[];
  expected_outcome_summary?: string;
  plays?: Play[];
  people?: string[];
};

type CampaignAssetChannel = ChannelRec & {
  engagement_score?: number | null;
  expected_delta_bp?: number;
  belief_conversion_likelihood?: number | null;
  average_confidence?: number | null;
  evidence_count?: number | null;
  mode?: string | null;
  modes?: string[];
  exploration_weight?: number | null;
  persona_aligned?: boolean;
  avg_duration_days?: number | null;
};

type CampaignAssetRecommendation = {
  asset: AssetRec;
  asset_type?: string | null;
  asset_type_label?: string | null;
  asset_fit_score?: number | null;
  asset_expected_delta_bp?: number | null;
  channel: CampaignAssetChannel;
  channel_type?: string | null;
  channel_type_label?: string | null;
  expected_delta_bp: number;
  belief_conversion_likelihood?: number | null;
  confidence?: number | null;
  evidence_count?: number | null;
  play_count?: number | null;
};

type Play = {
  persona_id: string;
  persona_label: string;
  stage_index: number;
  belief_transition?: Transition["belief_transition"];
  asset: AssetRec;
  channel: ChannelRec;
  asset_label?: string | null;
  asset_type?: string | null;
  asset_type_label?: string | null;
  channel_id?: string | null;
  channel_type?: string | null;
  channel_type_label?: string | null;
  channel_label?: string | null;
  expected_delta_bp: number;
  expected_delta_pct: number;
  confidence: number;
  asset_score?: number;
  channel_score?: number;
  start_day?: number;
  end_day?: number;
  quarter?: string;
  theme?: string;
  mode?: "broad" | "focused";
  exploration_weight?: number;
  path_probability?: number;
  path_id?: string;
  belief_conversion_likelihood?: number | null;
  avg_belief_conversion?: number | null;
  evidence_count?: number | null;
  impact_strength?: number | null;
  impact_stages?: Record<string, number>;
  people?: PersonMatch[];
  people_names?: string[];
  has_person_match?: boolean;
  primary_person?: string | null;
  timeline_index?: number;
  timeline_label?: string;
  timeline_days?: number;
  effort_required?: number;
  effort_pct?: number;
  account_meta?: Record<string, any>;
  persona_meta?: PersonaMeta;
  persona_descriptor?: string;
  segment_filters?: SegmentFilters;
  segment_summary?: string | null;
  expected_outcome_summary?: string;
  accounts?: Array<{ id?: string | null; name?: string | null }> | null;
  rationale?: ArsenalRationale | null;
  reasons?: Record<string, any> | null;
  time_to_effect_days?: number | null;
  duration_days?: number | null;
  persona_focus?: string | null;
  target_stage?: string | null;
  target_concern?: string | null;
  segment_fit_score?: number | null;
  belief_probability?: number | null;
  evidence?: Record<string, any> | null;
};

type Campaign = {
  quarter: string;
  theme: string;
  persona_focus?: string | null;
  conversion_summary?: string | null;
  conversion_focuses?: ConversionFocus[];
  focus_personas: string[];
  focus_pains: string[];
  focus_people: string[];
  start_day: number;
  end_day: number;
  total_delta_bp: number;
  confidence: number;
  plays: Play[];
  play_count?: number | null;
  mode_mix: { broad: number; focused: number };
  belief_transitions?: BeliefTransitionDetail[];
  asset_table?: CampaignAssetRecommendation[];
  segment_filters?: SegmentFilters;
  segment_summary?: string | null;
  persona_descriptor_counts?: PersonaDescriptorCount[];
  expected_outcome_summary?: string;
  accounts?: Array<{ id: string; name: string }>;
  avg_confidence?: number | null;
  focus_count?: number;
  broad_count?: number;
  time_to_effect_days?: number | null;
};

type RandomizationPolicy = {
  path_allocation?: Array<{
    path_id: string;
    probability: number;
    exploration_weight: number;
    persona_ids: string[];
  }>;
  belief_spread?: {
    early_cycle_avg?: number;
    late_cycle_avg?: number;
  };
  notes?: string;
};

type AccountPlan = {
  account_id: string;
  account_name: string;
  deal_status?: string;
  meta?: Record<string, any>;
  prediction: {
    persona_paths: PersonaPath[];
    fit: Record<string, any>;
    expected_next?: any;
    expected_next_personas?: Array<{
      persona_id: string;
      persona_label?: string | null;
      probability?: number | null;
      band?: string | null;
      journey_stage?: string | null;
      reason?: string | null;
    }>;
    observed_personas: string[];
    journey: {
      steps: JourneyStep[];
      total_steps: number;
    };
  };
  scatter?: { personas: PersonaSummary[] };
  transitions?: Transition[];
  execution?: {
    plays: Play[];
    conversion_sequence?: ConversionSequenceEntry[];
    persona_engagements?: PersonaEngagementPlan[];
    asset_cadence?: AssetCadenceRow[];
  };
  campaigns?: Campaign[];
  randomization?: RandomizationPolicy;
  learning?: any;
  enrichment?: AccountEnrichment;
  zmot?: {
    watchlist: AccountZmotWatchPersona[];
    events: AccountZmotPortfolioEvent[];
  };
  thesis?: {
    persona_likelihoods?: Array<{
      persona_id: string;
      persona_label?: string | null;
      probability?: number | null;
      band?: string | null;
    }>;
    person_likelihoods?: Array<{
      person_id?: string | null;
      display_name?: string | null;
      probability?: number | null;
      role_band?: string | null;
    }>;
    expected_next_personas?: Array<{
      persona_id: string;
      persona_label?: string | null;
      probability?: number | null;
      band?: string | null;
      journey_stage?: string | null;
      reason?: string | null;
    }>;
  };
  error?: string;
};

type PortfolioTheme = {
  quarter: string;
  theme: string;
  accounts: Array<{ id: string; name: string }>;
  focus_personas: string[];
  focus_people: string[];
  focus_pains: string[];
  total_delta_bp: number;
  avg_confidence: number;
  play_count: number;
  focus_keys?: string[];
  conversion_focuses?: PortfolioConversionFocus[];
  segment_filters?: SegmentFilters;
  segment_summary?: string | null;
  persona_descriptor_counts?: PersonaDescriptorCount[];
  expected_outcome_summary?: string;
};

type PortfolioPlan = {
  campaign_themes: PortfolioTheme[];
  conversion_focuses: PortfolioConversionFocus[];
  scatter: Array<{
    id?: string | null;
    label?: string;
    perceptibility?: number | null;
    proximity?: number | null;
    involvement?: number | null;
    path_probability?: number | null;
    belief_level?: number | null;
    expected_next_prob?: number | null;
    activation?: number | null;
  }>;
  broad_focus_mix: {
    broad: number;
    focused: number;
    broad_ratio: number;
  };
  randomization: {
    avg_path_probability: number;
    path_count_sampled: number;
  };
  expected_outcomes?: {
    portfolio: ExpectedOutcomeSummary;
    by_quarter: ExpectedOutcomeSummary[];
  };
  typical_path?: TypicalPathStep[];
  asset_cadence?: AssetCadenceRow[];
  arsenal_table?: BackendArsenalRow[];
  execution_matrix?: BackendExecutionMatrix | null;
};

type AccountPersonaRequirement = {
  persona_id: string;
  persona_label?: string | null;
  persona_title?: string | null;
  persona_department?: string | null;
  persona_seniority?: string | null;
  expected_in_deal: number;
  expected_in_deal_pct: number;
  stage_index: number;
  stage_label: string;
  matched_people: PersonMatch[];
  match_count?: number;
  has_match?: boolean;
  people_names?: string[];
};

type AccountEnrichmentSummary = {
  required_personas: number;
  personas_with_matches: number;
  matched_people: number;
  coverage_ratio: number;
  coverage_pct: number;
  total_people: number;
};

type AccountEnrichment = {
  persona_requirements: AccountPersonaRequirement[];
  summary: AccountEnrichmentSummary;
  unmatched_personas?: AccountPersonaRequirement[];
};

type PlanSummary = {
  account_count: number;
  avg_best_path_probability: number;
  avg_accuracy: number;
  top_personas: Array<{ label: string; count: number }>;
  top_pains: Array<{ label: string; count: number }>;
  top_capabilities: Array<{ label: string; count: number }>;
  total_expected_delta_bp: number;
  avg_people_coverage?: number | null;
  unmatched_persona_count?: number;
  total_persona_requirements?: number;
  typical_path?: TypicalPathStep[];
  persona_engagements?: PersonaEngagementPlan[];
  portfolio_summary_report?: BackendPortfolioSummary | null;
  portfolio_thesis?: BackendPortfolioThesis | null;
};

type StageDistributionRow = {
  stage: string;
  count: number;
};

type ClusterPersonaMatch = {
  name?: string;
  title?: string | null;
  department?: string | null;
  seniority?: string | null;
  accountId?: string | null;
  accountName?: string | null;
  confidence?: number | null;
  source?: string | null;
  persona?: string;
  stage?: string | null;
};

type ClusterPersonaExpectation = {
  persona: string;
  stage?: string | null;
  share?: number | null;
  matchRate?: number | null;
  requiredPersonas?: number | null;
  samplePeople: ClusterPersonaMatch[];
};

type ClusterCoalition = {
  sequence: string[];
  score?: number | null;
  share?: number | null;
};

type ClusterTransition = {
  persona?: string | null;
  stage?: string | null;
  narrative?: string | null;
  pain?: string | null;
  resolution?: string | null;
  deltaBp?: number | null;
  accountCount?: number | null;
};

type ClusterPlayPlan = {
  asset?: string | null;
  assetType?: string | null;
  channel?: string | null;
  channelType?: string | null;
  personaLabels: string[];
  stageLabels: string[];
  deltaBp?: number | null;
  accountCount?: number | null;
};

type ClusterFallbackPlan = {
  persona?: string | null;
  stage?: string | null;
  reason?: string | null;
  accounts: string[];
};

type ClusterLiftAnalysis = {
  clusterDeltaBp?: number | null;
  genericAllocationBp?: number | null;
  incrementalVsGenericBp?: number | null;
  perAccountDeltaBp?: number | null;
  peerPerAccountDeltaBp?: number | null;
};

type AccountCluster = {
  label: string;
  token: string;
  accountCount: number;
  shareOfExpectedLift: number;
  accounts: Array<{ id: string; name: string }>;
  personaExpectations: ClusterPersonaExpectation[];
  peopleMatches: ClusterPersonaMatch[];
  beliefCoalitions: ClusterCoalition[];
  beliefTransitions: ClusterTransition[];
  primaryPlays: ClusterPlayPlan[];
  fallbackPlan: ClusterFallbackPlan[];
  liftAnalysis?: ClusterLiftAnalysis | null;
};

type PortfolioSummaryReport = {
  totalAccounts: number;
  coveragePct: number | null;
  enrichedAccounts: number;
  beliefScore: number | null;
  maturityScore: number | null;
  assetsPlanned: number;
  channelCount: number;
  totalDeltaBp: number;
  expectedConversionRate: number | null;
  expectedTimeToConversionDays: number | null;
  stageDistribution: StageDistributionRow[];
  highRiskAccounts: Array<{ id: string; name: string; reason: string; score: number }>;
  highOpportunityAccounts: Array<{ id: string; name: string; reason: string; score: number }>;
  personaMix: Array<{ label: string; count: number }>;
  painMix: Array<{ label: string; count: number }>;
  capabilityMix: Array<{ label: string; count: number }>;
  fatigueAlerts: number;
  assetsPerPersona: Array<{ persona: string; count: number }>;
  accountClusters: AccountCluster[];
};

type BackendAccountCluster = {
  label?: string | null;
  token?: string | null;
  account_count?: number;
  share_of_expected_lift?: number;
  accounts?: Array<{ id?: string | null; name?: string | null }>;
  persona_expectations?: BackendClusterPersonaExpectation[] | null;
  people_matches?: BackendClusterPersonaMatch[] | null;
  belief_coalitions?: BackendClusterCoalition[] | null;
  belief_transitions?: BackendClusterTransition[] | null;
  primary_plays?: BackendClusterPlayPlan[] | null;
  fallback_plan?: BackendClusterFallbackPlan[] | null;
  lift_analysis?: BackendClusterLiftAnalysis | null;
};

type BackendClusterPersonaMatch = {
  name?: string | null;
  title?: string | null;
  department?: string | null;
  seniority?: string | null;
  account_id?: string | null;
  account_name?: string | null;
  confidence?: number | null;
  source?: string | null;
  persona?: string | null;
  stage?: string | null;
};

type BackendClusterPersonaExpectation = {
  persona?: string | null;
  stage?: string | null;
  share?: number | null;
  match_rate?: number | null;
  required_personas?: number | null;
  sample_people?: BackendClusterPersonaMatch[] | null;
};

type BackendClusterCoalition = {
  sequence?: string[] | null;
  score?: number | null;
  share?: number | null;
};

type BackendClusterTransition = {
  persona?: string | null;
  stage?: string | null;
  narrative?: string | null;
  pain?: string | null;
  resolution?: string | null;
  delta_bp?: number | null;
  account_count?: number | null;
};

type BackendClusterPlayPlan = {
  asset?: string | null;
  asset_type?: string | null;
  channel?: string | null;
  channel_type?: string | null;
  persona_labels?: string[] | null;
  stage_labels?: string[] | null;
  delta_bp?: number | null;
  account_count?: number | null;
};

type BackendClusterFallbackPlan = {
  persona?: string | null;
  stage?: string | null;
  reason?: string | null;
  accounts?: string[] | null;
};

type BackendClusterLiftAnalysis = {
  cluster_delta_bp?: number | null;
  generic_allocation_bp?: number | null;
  incremental_vs_generic_bp?: number | null;
  per_account_delta_bp?: number | null;
  peer_per_account_delta_bp?: number | null;
};

type BackendPortfolioSummary = {
  total_accounts?: number;
  coverage_pct?: number | null;
  enriched_accounts?: number;
  belief_score?: number | null;
  maturity_score?: number | null;
  assets_planned?: number;
  channel_count?: number;
  total_delta_bp?: number;
  expected_conversion_rate?: number | null;
  expected_time_to_conversion_days?: number | null;
  stage_distribution?: StageDistributionRow[];
  high_risk_accounts?: Array<{ id: string; name: string; reason: string; score: number }>;
  high_opportunity_accounts?: Array<{ id: string; name: string; reason: string; score: number }>;
  persona_mix?: Array<{ label: string; count: number }>;
  pain_mix?: Array<{ label: string; count: number }>;
  capability_mix?: Array<{ label: string; count: number }>;
  fatigue_alerts?: number;
  assets_per_persona?: Array<{ persona: string; count: number }>;
  account_clusters?: BackendAccountCluster[] | null;
};

type ThesisQuarterOutcome = {
  quarter: string;
  summary: string;
  deltaBp?: number;
  conversionReadyCount?: number;
  personaDescriptors?: Array<{ descriptor: string; count: number }>;
  segmentSummary?: string | null;
};

type ThesisData = {
  strategyHighlights: string[];
  personaSequence: string[];
  blockers: Array<{ id: string; name: string; reason: string }>;
  beliefTargets: StageDistributionRow[];
  segments: string[];
  quarterOutcomes: ThesisQuarterOutcome[];
  personaCadence: PersonaEngagementPlan[];
  clusterBriefs: AccountCluster[];
};

type BackendThesisQuarterOutcome = {
  quarter?: string;
  summary?: string;
  delta_bp?: number;
  conversion_ready_count?: number;
  persona_descriptors?: Array<{ descriptor: string; count: number }>;
  segment_summary?: string | null;
};

type BackendPortfolioThesis = {
  strategy_highlights?: string[];
  persona_sequence?: string[];
  blockers?: Array<{ id: string; name: string; reason: string }>;
  belief_targets?: StageDistributionRow[];
  segments?: string[];
  quarter_outcomes?: BackendThesisQuarterOutcome[];
  persona_cadence?: PersonaEngagementPlan[];
  cluster_briefs?: BackendAccountCluster[] | null;
};

type ArsenalRow = {
  theme: string;
  quarter: string;
  asset: string;
  channel: string;
  persona: string;
  accounts: string[];
  deltaBp: number;
  confidence: number | null;
  beliefConversion: number | null;
  timeToImpactDays: number | null;
  mode?: string | null;
  stageLabel?: string | null;
  stageIndex?: number | null;
  beliefTransition?: BeliefTransitionDetail | null;
  assetMeta?: AssetRec | null;
  channelMeta?: ChannelRec | null;
  rationale?: ArsenalRationale | null;
  reasons?: Record<string, any> | null;
  segmentFilters?: SegmentFilters;
  segmentSummary?: string | null;
  personaId?: string | null;
  accountId?: string | null;
  accountName?: string | null;
};

type AggregatedArsenalEntry = {
  key: string;
  assetLabel: string;
  assetFormat?: string | null;
  channelLabel: string;
  segments: string[];
  segmentSummary?: string | null;
  personas: string[];
  deltaBp: number;
  confidence: number | null;
  beliefConversion: number | null;
  avgDurationDays: number | null;
  themes: string[];
  quarters: string[];
  sampleRow: ArsenalRow;
};

type BackendArsenalRow = {
  theme?: string;
  quarter?: string;
  asset?: string;
  channel?: string;
  persona?: string;
  accounts?: string[];
  delta_bp?: number;
  confidence?: number | null;
  belief_conversion?: number | null;
  time_to_impact_days?: number | null;
  mode?: string | null;
  stage_label?: string | null;
  stage_index?: number | null;
  persona_id?: string | null;
  belief_transition?: BeliefTransitionDetail | null;
  asset_detail?: AssetRec | null;
  channel_detail?: ChannelRec | null;
  reasons?: Record<string, any> | null;
  rationale?: ArsenalRationale | null;
  segment_filters?: SegmentFilters;
  segment_summary?: string | null;
  account_id?: string | null;
  account_name?: string | null;
};

type ExecutionCell = {
  theme: string;
  deltaBp: number;
  mode: "broad" | "focused";
  accountCount: number;
  fatigue?: number | null;
  belief?: number | null;
  assets?: number;
  campaigns?: Array<{
    id: string;
    theme: string;
    description?: string | null;
    deltaBp?: number | null;
    assets?: number | null;
    fatigue?: number | null;
    belief?: number | null;
  }>;
};

type ExecutionMatrix = {
  quarters: string[];
  personas: string[];
  cells: Record<string, ExecutionCell[]>;
};

type BackendExecutionCell = {
  theme?: string;
  delta_bp?: number;
  mode?: "broad" | "focused";
  account_count?: number;
  fatigue?: number | null;
  belief?: number | null;
  assets?: number | null;
};

type BackendExecutionMatrix = {
  quarters?: string[];
  personas?: string[];
  cells?: Record<string, BackendExecutionCell[]>;
};

type MarketingPlan = {
  product_id: string;
  generated_at: string;
  summary: PlanSummary;
  accounts: AccountPlan[];
  portfolio_plan: PortfolioPlan;
  canonical_journey?: CanonicalJourney;
  canonical_persona_path?: TypicalPathStep[];
  product_insights?: ProductInsights;
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const fmtPercent = (value?: number | null) => {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  return `${Math.round(value * 100)}%`;
};

const fmtNumber = (value?: number | null, digits = 2) => {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  return Number(value).toFixed(digits);
};

const fmtBasisPoints = (value?: number | null) => {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  return `${Math.round(value)} bps`;
};

const dominantPhaseFromProbs = (
  probs?: PhaseProbabilities | null,
  fallback?: string | null
) => {
  if (fallback) return fallback;
  if (!probs) return null;
  const entries = Object.entries(probs);
  if (!entries.length) return null;
  entries.sort((a, b) => (b[1] ?? 0) - (a[1] ?? 0));
  return entries[0]?.[0] ?? null;
};

const avgOrNull = (values: Array<number | null | undefined>) => {
  const numeric = values.filter(
    (value): value is number => typeof value === "number" && !Number.isNaN(value)
  );
  if (numeric.length === 0) return null;
  return numeric.reduce((sum, value) => sum + value, 0) / numeric.length;
};

const toWeeks = (days?: number) => {
  if (days === null || days === undefined || Number.isNaN(days)) return "—";
  return `${Math.round(days / 7)}w`;
};

const titleize = (value?: string | null) => {
  if (!value) return undefined;
  return value
    .replace(/[_|-]+/g, " ")
    .split(" ")
    .filter(Boolean)
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1).toLowerCase())
    .join(" ");
};

const formatList = (items: string[], conjunction = "and") => {
  if (!items.length) return "";
  if (items.length === 1) return items[0];
  if (items.length === 2) return `${items[0]} ${conjunction} ${items[1]}`;
  return `${items[0]}, ${items[1]} ${conjunction} +${items.length - 2} more`;
};

const mapDayToQuarter = (day?: number | null): string | null => {
  if (typeof day !== "number" || Number.isNaN(day)) return null;
  const clamped = Math.max(0, day);
  const quarterIndex = Math.min(3, Math.floor(clamped / 90));
  return `Q${quarterIndex + 1}`;
};

const STAGE_LABELS: Record<string, string> = {
  problem: "Problem Realization",
  execution: "Execution Guidance",
  pain: "Pain Realization",
  resolution: "Resolution Discovery",
};

const STAGE_ORDER: Record<string, number> = {
  "zero moment of truth": 0,
  "zmot": 0,
  "problem realization": 1,
  "pain realization": 2,
  "resolution discovery": 3,
  "barrier-breaking": 4,
  "execution guidance": 5,
};

const quarterOrder = (q: string) => {
  const match = /(\d+)/.exec(q);
  return match ? parseInt(match[1], 10) : Number.MAX_SAFE_INTEGER;
};

const themePalette = [
  "#2E86AB",
  "#88498F",
  "#E76F51",
  "#2A9D8F",
  "#C8553D",
  "#3D348B",
  "#577590",
  "#BC6C25",
];

const themeColorFor = (theme: string) => {
  if (!theme) return themePalette[0];
  const normalized = theme.toLowerCase();
  const hash = normalized.split("").reduce((sum, char) => sum + char.charCodeAt(0), 0);
  return themePalette[Math.abs(hash) % themePalette.length];
};

const stageLabelFromTransition = (bt?: BeliefTransitionDetail | null) => {
  if (!bt) return null;
  if (bt.stage_label) return bt.stage_label;
  if (bt.stage && STAGE_LABELS[bt.stage]) return STAGE_LABELS[bt.stage];
  if (bt.pain) return STAGE_LABELS.pain;
  if (bt.resolution) return STAGE_LABELS.resolution;
  if (bt.problem) return STAGE_LABELS.problem;
  return STAGE_LABELS.execution;
};

const normalizeStageLabel = (label?: string | null) => {
  if (!label) return "Next Stage";
  const cleaned = label.trim();
  if (!cleaned) return "Next Stage";
  const lower = cleaned.toLowerCase();
  for (const [key, value] of Object.entries(STAGE_LABELS)) {
    if (lower.includes(key)) return value;
    if (value.toLowerCase() === lower) return value;
  }
  return cleaned;
};

const mapClusterMatches = (
  matches?: BackendClusterPersonaMatch[] | null
): ClusterPersonaMatch[] => {
  if (!matches?.length) return [];
  return matches.map((match, idx) => {
    const fallbackName =
      match.name ??
      (match as any).person_name ??
      (match as any).display_name ??
      `Person ${idx + 1}`;
    return {
      name: fallbackName,
      title: match.title ?? (match as any).person_title ?? null,
      department: match.department ?? (match as any).person_department ?? null,
      seniority: match.seniority ?? (match as any).person_seniority ?? null,
      accountId: match.account_id ?? null,
      accountName: match.account_name ?? null,
      confidence:
        typeof match.confidence === "number"
          ? match.confidence
          : typeof (match as any).match_confidence === "number"
          ? (match as any).match_confidence
          : null,
      source: match.source ?? null,
      persona: match.persona ?? undefined,
      stage: match.stage ?? null,
    };
  });
};

const mapClusterPersonaExpectations = (
  expectations?: BackendClusterPersonaExpectation[] | null
): ClusterPersonaExpectation[] => {
  if (!expectations?.length) return [];
  return expectations.map((entry, idx) => ({
    persona: entry.persona ?? `Persona ${idx + 1}`,
    stage: entry.stage ?? null,
    share: entry.share ?? null,
    matchRate: entry.match_rate ?? null,
    requiredPersonas: entry.required_personas ?? null,
    samplePeople: mapClusterMatches(entry.sample_people),
  }));
};

const mapClusterCoalitions = (
  coalitions?: BackendClusterCoalition[] | null
): ClusterCoalition[] =>
  (coalitions ?? []).map((coalition) => ({
    sequence: coalition.sequence ?? [],
    score: coalition.score ?? null,
    share: coalition.share ?? null,
  }));

const mapClusterTransitions = (
  transitions?: BackendClusterTransition[] | null
): ClusterTransition[] =>
  (transitions ?? []).map((transition) => ({
    persona: transition.persona ?? null,
    stage: transition.stage ?? null,
    narrative: transition.narrative ?? null,
    pain: transition.pain ?? null,
    resolution: transition.resolution ?? null,
    deltaBp: transition.delta_bp ?? null,
    accountCount: transition.account_count ?? null,
  }));

const mapClusterPlays = (
  plays?: BackendClusterPlayPlan[] | null
): ClusterPlayPlan[] =>
  (plays ?? []).map((play) => ({
    asset: play.asset ?? null,
    assetType: play.asset_type ?? null,
    channel: play.channel ?? null,
    channelType: play.channel_type ?? null,
    personaLabels: play.persona_labels ?? [],
    stageLabels: play.stage_labels ?? [],
    deltaBp: play.delta_bp ?? null,
    accountCount: play.account_count ?? null,
  }));

const mapClusterFallbackPlan = (
  items?: BackendClusterFallbackPlan[] | null
): ClusterFallbackPlan[] =>
  (items ?? []).map((item) => ({
    persona: item.persona ?? null,
    stage: item.stage ?? null,
    reason: item.reason ?? null,
    accounts: item.accounts ?? [],
  }));

const mapClusterLiftAnalysis = (
  entry?: BackendClusterLiftAnalysis | null
): ClusterLiftAnalysis | null => {
  if (!entry) return null;
  return {
    clusterDeltaBp: entry.cluster_delta_bp ?? null,
    genericAllocationBp: entry.generic_allocation_bp ?? null,
    incrementalVsGenericBp: entry.incremental_vs_generic_bp ?? null,
    perAccountDeltaBp: entry.per_account_delta_bp ?? null,
    peerPerAccountDeltaBp: entry.peer_per_account_delta_bp ?? null,
  };
};

const mapBackendAccountClusters = (
  clusters?: BackendAccountCluster[] | null
): AccountCluster[] => {
  if (!clusters?.length) return [];
  return clusters.map((cluster, idx) => {
    const accounts = (cluster.accounts ?? []).map((acct, accountIdx) => ({
      id: acct.id ?? `cluster-account-${idx}-${accountIdx}`,
      name: acct.name ?? acct.id ?? "Account",
    }));
    const fallbackLabel = cluster.token || `Cluster ${idx + 1}`;
    return {
      label: cluster.label || fallbackLabel,
      token: cluster.token || fallbackLabel,
      accountCount: cluster.account_count ?? accounts.length,
      shareOfExpectedLift:
        typeof cluster.share_of_expected_lift === "number"
          ? cluster.share_of_expected_lift
          : 0,
      accounts,
      personaExpectations: mapClusterPersonaExpectations(cluster.persona_expectations),
      peopleMatches: mapClusterMatches(cluster.people_matches),
      beliefCoalitions: mapClusterCoalitions(cluster.belief_coalitions),
      beliefTransitions: mapClusterTransitions(cluster.belief_transitions),
      primaryPlays: mapClusterPlays(cluster.primary_plays),
      fallbackPlan: mapClusterFallbackPlan(cluster.fallback_plan),
      liftAnalysis: mapClusterLiftAnalysis(cluster.lift_analysis),
    };
  });
};

const mapBackendPortfolioSummary = (
  data?: BackendPortfolioSummary | null
): PortfolioSummaryReport | null => {
  if (!data) return null;
  return {
    totalAccounts: data.total_accounts ?? 0,
    coveragePct: data.coverage_pct ?? null,
    enrichedAccounts: data.enriched_accounts ?? 0,
    beliefScore: data.belief_score ?? null,
    maturityScore: data.maturity_score ?? null,
    assetsPlanned: data.assets_planned ?? 0,
    channelCount: data.channel_count ?? 0,
    totalDeltaBp: data.total_delta_bp ?? 0,
    expectedConversionRate: data.expected_conversion_rate ?? null,
    expectedTimeToConversionDays: data.expected_time_to_conversion_days ?? null,
    stageDistribution: data.stage_distribution ?? [],
    highRiskAccounts: data.high_risk_accounts ?? [],
    highOpportunityAccounts: data.high_opportunity_accounts ?? [],
    personaMix: data.persona_mix ?? [],
    painMix: data.pain_mix ?? [],
    capabilityMix: data.capability_mix ?? [],
    fatigueAlerts: data.fatigue_alerts ?? 0,
    assetsPerPersona: data.assets_per_persona ?? [],
    accountClusters: mapBackendAccountClusters(data.account_clusters),
  };
};

const mapBackendThesisData = (
  data?: BackendPortfolioThesis | null,
  fallbackCadence?: PersonaEngagementPlan[]
): ThesisData | null => {
  if (!data) return null;
  const cadence = data.persona_cadence ?? fallbackCadence ?? [];
  const clusterBriefs =
    mapBackendAccountClusters(data.cluster_briefs ?? null) ?? [];
  return {
    strategyHighlights: data.strategy_highlights ?? [],
    personaSequence: data.persona_sequence ?? [],
    blockers: data.blockers ?? [],
    beliefTargets: data.belief_targets ?? [],
    segments: data.segments ?? [],
    quarterOutcomes: (data.quarter_outcomes ?? []).map((entry) => ({
      quarter: entry.quarter ?? "Quarter",
      summary: entry.summary ?? "Outcome pending",
      deltaBp: entry.delta_bp,
      conversionReadyCount: entry.conversion_ready_count,
      personaDescriptors: entry.persona_descriptors,
      segmentSummary: entry.segment_summary ?? null,
    })),
    personaCadence: cadence,
    clusterBriefs,
  };
};

const mapBackendArsenalRows = (rows?: BackendArsenalRow[] | null): ArsenalRow[] => {
  if (!rows) return [];
  return rows.map((row) => {
    const beliefTransition = row.belief_transition ?? null;
    const stageLabel =
      row.stage_label || stageLabelFromTransition(beliefTransition) || null;
    const segmentFilters = row.segment_filters;
    const segmentSummary = row.segment_summary ?? summarizeSegmentFilters(segmentFilters);
    return {
      theme: row.theme || "Campaign",
      quarter: row.quarter || "Q1",
      asset: row.asset || "Asset",
      channel: row.channel || "Channel",
      persona: row.persona || "Target persona",
      accounts: row.accounts ?? [],
      deltaBp: row.delta_bp ?? 0,
      confidence:
        typeof row.confidence === "number" && !Number.isNaN(row.confidence)
          ? row.confidence
          : null,
      beliefConversion:
        typeof row.belief_conversion === "number" && !Number.isNaN(row.belief_conversion)
          ? row.belief_conversion
          : null,
      timeToImpactDays:
        typeof row.time_to_impact_days === "number" ? row.time_to_impact_days : null,
      mode: row.mode ?? undefined,
      stageLabel,
      stageIndex: typeof row.stage_index === "number" ? row.stage_index : null,
      beliefTransition,
      assetMeta: (row.asset_detail ?? null) as AssetRec | null,
      channelMeta: (row.channel_detail ?? null) as ChannelRec | null,
      rationale: row.rationale ?? null,
      reasons: row.reasons ?? null,
      segmentFilters,
      segmentSummary,
      personaId: row.persona_id ?? null,
      accountId: row.account_id ?? null,
      accountName: row.account_name ?? null,
    };
  });
};

const mapBackendExecutionMatrix = (
  data?: BackendExecutionMatrix | null
): ExecutionMatrix | null => {
  if (!data || !data.quarters || !data.personas) return null;
  const mappedCells: Record<string, ExecutionCell[]> = {};
  Object.entries(data.cells ?? {}).forEach(([key, entries]) => {
    mappedCells[key] = entries.map((entry) => ({
      theme: entry.theme || "Campaign",
      deltaBp: entry.delta_bp ?? 0,
      mode: entry.mode === "broad" ? "broad" : "focused",
      accountCount: entry.account_count ?? 0,
      fatigue:
        typeof entry.fatigue === "number" && !Number.isNaN(entry.fatigue)
          ? entry.fatigue
          : undefined,
      belief:
        typeof entry.belief === "number" && !Number.isNaN(entry.belief)
          ? entry.belief
          : undefined,
      assets:
        typeof entry.assets === "number" && !Number.isNaN(entry.assets)
          ? entry.assets
          : undefined,
    }));
  });
  return {
    quarters: data.quarters,
    personas: data.personas,
    cells: mappedCells,
  };
};

const segmentChipsFromFilters = (filters?: SegmentFilters) => {
  if (!filters) return [] as Array<[string, string]>;
  return Object.entries(filters).filter(([, value]) => value && value !== "any");
};

const summarizeSegmentFilters = (
  filters?: SegmentFilters,
  fallback?: string | null
): string | null => {
  const entries = segmentChipsFromFilters(filters);
  if (!entries.length) return fallback ?? null;
  return entries
    .map(([key, value]) => `${titleize(key) ?? key}: ${value}`)
    .join(", ");
};

const renderSegmentSummary = (filters?: SegmentFilters, summary?: string | null) => {
  const entries = segmentChipsFromFilters(filters);
  const effectiveSummary = summary ?? summarizeSegmentFilters(filters);
  if (!entries.length && !effectiveSummary) return null;
  return (
    <Stack direction="row" spacing={0.5} sx={{ flexWrap: "wrap", mt: 0.75 }}>
      {(effectiveSummary ? [effectiveSummary] : []).map((text) => (
        <Chip key={`segment-summary-${text}`} label={text} size="small" variant="outlined" />
      ))}
      {entries.map(([key, value]) => (
        <Chip
          key={`segment-${key}-${value}`}
          label={`${titleize(key)}: ${value}`}
          size="small"
          variant="outlined"
        />
      ))}
    </Stack>
  );
};

const STAGE_INTENSITY_LABELS: Record<string, string> = {
  probe: "Light discovery taps",
  ramp: "High-intent push",
  sustain: "Steady pressure",
  decay: "Back off & monitor",
  park: "Monitor only",
};

const STAGE_COLOR_MAP: Record<string, string> = {
  probe: "#90caf9",
  ramp: "#42a5f5",
  sustain: "#1e88e5",
  decay: "#ffb74d",
  park: "#b0bec5",
  unaware: "#bdbdbd",
  discovery: "#64b5f6",
  "problem realization": "#ffcc80",
  "execution guidance": "#aed581",
  "pain realization": "#ffa726",
  "resolution discovery": "#80cbc4",
};

const stageColorFor = (stage?: string | null) => {
  if (!stage) return STAGE_COLOR_MAP.probe;
  const normalized = stage.toLowerCase();
  return (
    STAGE_COLOR_MAP[normalized] ||
    STAGE_COLOR_MAP[normalized.replace(/_/g, " ")] ||
    STAGE_COLOR_MAP.probe
  );
};

const computeMeanStd = (values: number[]): { mean: number; std: number } => {
  if (!values.length) return { mean: 0, std: 1 };
  const mean = values.reduce((sum, value) => sum + value, 0) / values.length;
  const variance =
    values.reduce((sum, value) => sum + (value - mean) ** 2, 0) / values.length;
  const std = Math.sqrt(variance) || 1;
  return { mean, std };
};

const classifyPersonaRole = (percZ: number, proxZ: number): string => {
  const percHigh = percZ >= 0;
  const proxHigh = proxZ >= 0;
  if (percHigh && proxHigh) return "Potential Champions";
  if (percHigh && !proxHigh) return "Key Drivers";
  if (!percHigh && proxHigh) return "Operators";
  return "Potential Blockers";
};

const FOUR_PHASE_KEYS = ["probe", "ramp", "sustain", "decay", "park"] as const;
type CadencePhase = (typeof FOUR_PHASE_KEYS)[number];

type PortfolioCadenceStats = {
  touches: Record<CadencePhase, number>;
  people: Record<CadencePhase, number>;
  totalTouches: number;
  totalPeople: number;
};

const normalizeCadencePhase = (phase?: string | null): CadencePhase => {
  const normalized = (phase || "").toLowerCase() as CadencePhase;
  if ((FOUR_PHASE_KEYS as readonly string[]).includes(normalized)) {
    return normalized;
  }
  return "probe";
};

const buildPortfolioCadenceStats = (
  engagements: PersonaEngagementPlan[]
): PortfolioCadenceStats => {
  const touches = FOUR_PHASE_KEYS.reduce(
    (acc, phase) => ({ ...acc, [phase]: 0 }),
    {} as Record<CadencePhase, number>
  );
  const people = FOUR_PHASE_KEYS.reduce(
    (acc, phase) => ({ ...acc, [phase]: 0 }),
    {} as Record<CadencePhase, number>
  );

  engagements.forEach((engagement) => {
    const cadencePhase = normalizeCadencePhase(
      engagement.engagement_state || engagement.cadence?.[0]?.phase
    );
    people[cadencePhase] += 1;
    (engagement.cadence || []).forEach((row) => {
      const phase = normalizeCadencePhase(row.phase);
      if (typeof row.frequency_per_week === "number" && row.frequency_per_week > 0) {
        touches[phase] += row.frequency_per_week;
      }
    });
  });

  const totalTouches = Object.values(touches).reduce((sum, value) => sum + value, 0);
  const totalPeople = Object.values(people).reduce((sum, value) => sum + value, 0);

  return {
    touches,
    people,
    totalTouches,
    totalPeople,
  };
};

const aggregateArsenalEntries = (rows: ArsenalRow[]): AggregatedArsenalEntry[] => {
  const map = new Map<
    string,
    {
      entry: AggregatedArsenalEntry;
      personasSet: Set<string>;
      segmentsSet: Set<string>;
      themesSet: Set<string>;
      quartersSet: Set<string>;
      deltaValues: number[];
      confidenceValues: number[];
      beliefValues: number[];
      durationValues: number[];
    }
  >();

  const ensureSegmentLabels = (row: ArsenalRow) => {
    const chips = segmentChipsFromFilters(row.segmentFilters);
    if (chips.length) {
      return chips.map(([key, value]) => `${titleize(key)}: ${value}`);
    }
    return row.segmentSummary ? [row.segmentSummary] : ["Any segment"];
  };

  rows.forEach((row, idx) => {
    const assetLabel =
      row.asset ||
      row.assetMeta?.name ||
      row.assetMeta?.title ||
      row.assetMeta?.category_label ||
      "Asset";
    const assetFormat =
      row.assetMeta?.format || row.assetMeta?.category_label || row.assetMeta?.type || null;
    const channelLabel =
      row.channel ||
      row.channelMeta?.name ||
      row.channelMeta?.channel_type_label ||
      "Channel";
    const segmentSummary =
      summarizeSegmentFilters(row.segmentFilters, row.segmentSummary) || "Any segment";
    const key = `${assetLabel}__${channelLabel}__${segmentSummary}`;
    const segments = ensureSegmentLabels(row);
    if (!map.has(key)) {
      map.set(key, {
        entry: {
          key,
          assetLabel,
          assetFormat,
          channelLabel,
          segments: [],
          segmentSummary,
          personas: [],
          deltaBp: 0,
          confidence: null,
          beliefConversion: null,
          avgDurationDays: null,
          themes: [],
          quarters: [],
          sampleRow: row,
        },
        personasSet: new Set(row.persona ? [row.persona] : []),
        segmentsSet: new Set(segments),
        themesSet: new Set(row.theme ? [row.theme] : []),
        quartersSet: new Set(row.quarter ? [row.quarter] : []),
        deltaValues: Number.isFinite(row.deltaBp) ? [row.deltaBp] : [],
        confidenceValues:
          typeof row.confidence === "number" ? [row.confidence] : [],
        beliefValues:
          typeof row.beliefConversion === "number" ? [row.beliefConversion] : [],
        durationValues:
          typeof row.timeToImpactDays === "number" ? [row.timeToImpactDays] : [],
      });
    } else {
      const bucket = map.get(key)!;
      if (row.persona) bucket.personasSet.add(row.persona);
      segments.forEach((segment) => bucket.segmentsSet.add(segment));
      if (row.theme) bucket.themesSet.add(row.theme);
      if (row.quarter) bucket.quartersSet.add(row.quarter);
      if (Number.isFinite(row.deltaBp)) bucket.deltaValues.push(row.deltaBp);
      if (typeof row.confidence === "number") bucket.confidenceValues.push(row.confidence);
      if (typeof row.beliefConversion === "number")
        bucket.beliefValues.push(row.beliefConversion);
      if (typeof row.timeToImpactDays === "number")
        bucket.durationValues.push(row.timeToImpactDays);
    }
  });

  const average = (values: number[]): number | null =>
    values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : null;

  return Array.from(map.values())
    .map((bucket) => {
      const deltaMax = bucket.deltaValues.length
        ? Math.max(...bucket.deltaValues)
        : 0;
      return {
        ...bucket.entry,
        deltaBp: deltaMax,
        confidence: average(bucket.confidenceValues),
        beliefConversion: average(bucket.beliefValues),
        avgDurationDays: average(bucket.durationValues),
        personas: Array.from(bucket.personasSet),
        segments: Array.from(bucket.segmentsSet),
        themes: Array.from(bucket.themesSet),
        quarters: Array.from(bucket.quartersSet),
      };
    })
    .sort((a, b) => b.deltaBp - a.deltaBp);
};

type FourQuestionsPortfolioProps = {
  plan: MarketingPlan;
  thesis: ThesisData | null;
  summary: PortfolioSummaryReport | null;
  aggregatedArsenal: AggregatedArsenalEntry[];
  personaEngagements: PersonaEngagementPlan[];
  cadenceStats: PortfolioCadenceStats;
};

const FourQuestionsPortfolio: React.FC<FourQuestionsPortfolioProps> = ({
  plan,
  thesis,
  summary,
  aggregatedArsenal,
  personaEngagements,
  cadenceStats,
}) => {
  const [showAllPlays, setShowAllPlays] = useState(false);

  const personaRoleRows = useMemo(() => {
    const engagements = personaEngagements || [];
    if (!engagements.length) {
      return (plan.summary?.top_personas || []).slice(0, 6).map((item, idx) => ({
        persona: item.label || `Persona ${idx + 1}`,
        stage: "—",
        role: "Needs signals",
        probability: item.count ?? 0,
      }));
    }
    const buckets = new Map<
      string,
      {
        label: string;
        stage: string | null;
        perc: Array<number | null>;
        prox: Array<number | null>;
        probability: Array<number | null>;
      }
    >();

    engagements.forEach((engagement) => {
      const key = engagement.persona_id || engagement.persona_label || `persona-${buckets.size}`;
      if (!buckets.has(key)) {
        buckets.set(key, {
          label: engagement.persona_label || engagement.persona_id || key,
          stage:
            dominantPhaseFromProbs(engagement.phase_probs, engagement.dominant_phase) ||
            engagement.journey_phase ||
            null,
          perc: [],
          prox: [],
          probability: [],
        });
      }
      const bucket = buckets.get(key)!;
      if (!bucket.stage) {
        bucket.stage =
          dominantPhaseFromProbs(engagement.phase_probs, engagement.dominant_phase) ||
          engagement.journey_phase ||
          bucket.stage;
      }
      const metrics = engagement.belief_metrics;
      if (metrics) {
        bucket.perc.push(metrics.perceptibility ?? null);
        bucket.prox.push(metrics.proximity ?? null);
      }
      bucket.probability.push(engagement.committee_probability ?? engagement.priority_score ?? null);
    });

    const aggregated = Array.from(buckets.values()).map((bucket) => ({
      label: bucket.label,
      stage: bucket.stage,
      avgPerc: avgOrNull(bucket.perc),
      avgProx: avgOrNull(bucket.prox),
      probability: avgOrNull(bucket.probability),
    }));

    const percStats = computeMeanStd(
      aggregated
        .map((entry) => entry.avgPerc)
        .filter((value): value is number => typeof value === "number")
    );
    const proxStats = computeMeanStd(
      aggregated
        .map((entry) => entry.avgProx)
        .filter((value): value is number => typeof value === "number")
    );

    return aggregated
      .map((entry) => {
        const percZ =
          entry.avgPerc !== null ? (entry.avgPerc - percStats.mean) / percStats.std : null;
        const proxZ =
          entry.avgProx !== null ? (entry.avgProx - proxStats.mean) / proxStats.std : null;
        const role =
          percZ !== null && proxZ !== null
            ? classifyPersonaRole(percZ, proxZ)
            : "Needs signals";
        return {
          persona: entry.label,
          stage: titleize(entry.stage) || entry.stage || "—",
          role,
          probability: entry.probability ?? 0,
        };
      })
      .sort((a, b) => (b.probability ?? 0) - (a.probability ?? 0))
      .slice(0, 8);
  }, [personaEngagements, plan.summary]);

  const stageDistribution =
    summary?.stageDistribution?.length
      ? summary.stageDistribution
      : thesis?.beliefTargets || [];
  const avgBeliefScore =
    summary?.beliefScore ??
    (thesis?.beliefTargets?.length
      ? thesis.beliefTargets.reduce((sum, row) => sum + row.count, 0) /
        (thesis.beliefTargets.length * 5)
      : null);
  const totalStageCount = stageDistribution.reduce((sum, row) => sum + (row.count || 0), 0);

  const cadenceTableRows = useMemo(
    () =>
      FOUR_PHASE_KEYS.map((phase) => {
        const people = cadenceStats.people[phase] || 0;
        const share =
          cadenceStats.totalTouches > 0
            ? cadenceStats.touches[phase] / cadenceStats.totalTouches
            : null;
        return {
          phase,
          people,
          share,
          intensity: STAGE_INTENSITY_LABELS[phase],
        };
      }),
    [cadenceStats]
  );

  const assetsPreview = showAllPlays
    ? aggregatedArsenal
    : aggregatedArsenal.slice(0, 5);
  const hasMoreAssets = aggregatedArsenal.length > 5;

  const sectionHeading = (label: string, helper: string) => (
    <Stack direction="row" spacing={0.5} alignItems="center" sx={{ mb: 1 }}>
      <Typography variant="subtitle1">{label}</Typography>
      <Tooltip title={helper}>
        <InfoOutlinedIcon fontSize="small" color="action" />
      </Tooltip>
    </Stack>
  );

  return (
    <Stack spacing={2}>
      <Card variant="outlined">
        <CardContent>
          {sectionHeading(
            "1. Who is likely to be involved?",
            "Top committee personas across target accounts, with their expected roles."
          )}
          <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
            Role derived from perceptibility vs proximity signals.
          </Typography>
          <TableContainer component={Paper} variant="outlined">
            <Table size="small">
              <TableHead>
                <TableRow>
                  <TableCell>Persona</TableCell>
                  <TableCell>Expected Stage</TableCell>
                  <TableCell>Expected Role</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {personaRoleRows.length ? (
                  personaRoleRows.map((row) => (
                    <TableRow key={`role-${row.persona}`}>
                      <TableCell>{row.persona}</TableCell>
                      <TableCell>{row.stage}</TableCell>
                      <TableCell>{row.role}</TableCell>
                    </TableRow>
                  ))
                ) : (
                  <TableRow>
                    <TableCell colSpan={3}>
                      <Typography variant="body2" color="text.secondary">
                        Persona coverage will populate once journey paths refresh.
                      </Typography>
                    </TableCell>
                  </TableRow>
                )}
              </TableBody>
            </Table>
          </TableContainer>
        </CardContent>
      </Card>

      <Card variant="outlined">
        <CardContent>
          {sectionHeading(
            "2. What is the current belief level?",
            "Belief score and confidence per stage."
          )}
          <Typography variant="h4" sx={{ mb: 1 }}>
            {avgBeliefScore !== null ? `${Math.round(avgBeliefScore * 100)}` : "—"}
            <Typography component="span" variant="body2" color="text.secondary" sx={{ ml: 0.5 }}>
              / 100 portfolio belief
            </Typography>
          </Typography>
          {stageDistribution.length ? (
            <>
              <Box
                sx={{
                  display: "flex",
                  borderRadius: 1,
                  overflow: "hidden",
                  border: "1px solid",
                  borderColor: "divider",
                  minHeight: 32,
                }}
              >
                {stageDistribution.map((row) => {
                  const share = totalStageCount ? row.count / totalStageCount : 0;
                  return (
                    <Box
                      key={row.stage}
                      sx={{
                        flex: share || 0.1,
                        minWidth: 40,
                        backgroundColor: stageColorFor(row.stage),
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "center",
                        color: "common.white",
                        px: 1,
                      }}
                    >
                      <Typography variant="caption" sx={{ fontWeight: 600 }}>
                        {row.stage}
                      </Typography>
                    </Box>
                  );
                })}
              </Box>
              <Stack direction="row" spacing={1} sx={{ flexWrap: "wrap", mt: 1 }}>
                {stageDistribution.map((row) => (
                  <Chip
                    key={`dist-${row.stage}`}
                    size="small"
                    label={`${row.stage}: ${row.count} personas`}
                    sx={{ backgroundColor: `${stageColorFor(row.stage)}22` }}
                  />
                ))}
              </Stack>
            </>
          ) : (
            <Typography variant="body2" color="text.secondary">
              Stage distribution will appear once the thesis refreshes.
            </Typography>
          )}
        </CardContent>
      </Card>

      <Card variant="outlined">
        <CardContent>
          {sectionHeading(
            "3. How often will we engage them?",
            "Cadence mix driven by belief stage, fatigue, and response."
          )}
          <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
            Portfolio-wide view of cadence policy by phase.
          </Typography>
          <TableContainer component={Paper} variant="outlined">
            <Table size="small">
              <TableHead>
                <TableRow>
                  <TableCell>Stage</TableCell>
                  <TableCell align="right">People in stage</TableCell>
                  <TableCell align="right">Share of touches</TableCell>
                  <TableCell>Intensity</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {cadenceTableRows.map((row) => (
                  <TableRow key={`cadence-${row.phase}`}>
                    <TableCell>{titleize(row.phase)}</TableCell>
                    <TableCell align="right">{row.people}</TableCell>
                    <TableCell align="right">{fmtPercent(row.share)}</TableCell>
                    <TableCell>{row.intensity}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        </CardContent>
      </Card>

      <Card variant="outlined">
        <CardContent>
          {sectionHeading(
            "4. What assets & channels will we deploy?",
            "Uniqued plays with the strongest expected lift."
          )}
          <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
            {showAllPlays
              ? "Full arsenal preview with deduplicated plays."
              : "Top unique plays across the portfolio."}
          </Typography>
          {assetsPreview.length ? (
            <TableContainer component={Paper} variant="outlined">
              <Table size="small">
                <TableHead>
                  <TableRow>
                    <TableCell>Asset</TableCell>
                    <TableCell>Channel</TableCell>
                    <TableCell>Segments</TableCell>
                    <TableCell>Personas</TableCell>
                    <TableCell align="right">Δ belief</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {assetsPreview.map((entry) => (
                    <TableRow key={entry.key}>
                      <TableCell>
                        <Typography variant="body2">{entry.assetLabel}</Typography>
                        {entry.assetFormat && (
                          <Typography variant="caption" color="text.secondary">
                            {entry.assetFormat}
                          </Typography>
                        )}
                      </TableCell>
                      <TableCell>{entry.channelLabel}</TableCell>
                      <TableCell>
                        <Stack spacing={0.5}>
                          {entry.segments.slice(0, 3).map((segment) => (
                            <Chip key={`${entry.key}-${segment}`} label={segment} size="small" />
                          ))}
                          {entry.segments.length > 3 && (
                            <Typography variant="caption" color="text.secondary">
                              +{entry.segments.length - 3} more
                            </Typography>
                          )}
                        </Stack>
                      </TableCell>
                      <TableCell>
                        <Stack spacing={0.5}>
                          {entry.personas.slice(0, 2).map((persona) => (
                            <Chip key={`${entry.key}-${persona}`} label={persona} size="small" variant="outlined" />
                          ))}
                          {entry.personas.length > 2 && (
                            <Typography variant="caption" color="text.secondary">
                              +{entry.personas.length - 2} more
                            </Typography>
                          )}
                        </Stack>
                      </TableCell>
                      <TableCell align="right">{fmtBasisPoints(entry.deltaBp)}</TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          ) : (
            <Typography variant="body2" color="text.secondary">
              Arsenal data will populate once campaigns are generated.
            </Typography>
          )}
          {hasMoreAssets && (
            <Button
              size="small"
              sx={{ mt: 1.5 }}
              onClick={() => setShowAllPlays((prev) => !prev)}
            >
              {showAllPlays ? "Hide extra plays" : "See all plays"}
            </Button>
          )}
        </CardContent>
      </Card>
    </Stack>
  );
};

const formatObservables = (observables?: AccountZmotObservable[]) => {
  if (!observables?.length) return "—";
  return observables
    .map((observable) => observable.label || observable.description)
    .filter(Boolean)
    .join(", ");
};

const formatKeywords = (keywords?: AccountZmotKeyword[]) => {
  if (!keywords?.length) return "—";
  return keywords.map((item) => item.label || item.keyword).filter(Boolean).join(", ");
};

const incrementCounter = (counter: Map<string, number>, label?: string | null) => {
  if (!label) return;
  counter.set(label, (counter.get(label) || 0) + 1);
};

const renderTypicalPathSteps = (
  steps: TypicalPathStep[],
  placeholder = "Persona journey will populate once insights refresh."
) => {
  if (!steps.length) {
    return (
      <Typography variant="body2" color="text.secondary">
        {placeholder}
      </Typography>
    );
  }
  return (
    <Stack spacing={1}>
      {steps.map((step, idx) => {
        const stepNumber = step.step ?? idx + 1;
        const personaLabel = step.persona || step.label || step.name || "Persona";
        const tooltipLines = [
          step.impact ? `Impact: ${step.impact}` : null,
          step.reason ? `Why: ${step.reason}` : null,
          typeof step.confidence === "number"
            ? `Confidence ${fmtPercent(step.confidence)}`
            : null,
        ].filter(Boolean) as string[];
        const tooltip =
          tooltipLines.length > 0 ? (
            <Box>
              {tooltipLines.map((line, idx) => (
                <Typography key={`${step.persona}-detail-${idx}`} variant="body2">
                  {line}
                </Typography>
              ))}
            </Box>
          ) : null;
        return (
          <Paper key={`${stepNumber}-${personaLabel}`} variant="outlined" sx={{ p: 1.25 }}>
            <Stack direction="row" spacing={1} alignItems="center">
              <Chip size="small" label={`Step ${stepNumber}`} />
              <Typography variant="subtitle2">{personaLabel}</Typography>
              {tooltip && (
                <Tooltip title={tooltip} arrow>
                  <IconButton size="small">
                    <InfoOutlinedIcon fontSize="inherit" />
                  </IconButton>
                </Tooltip>
              )}
            </Stack>
            <Typography variant="caption" color="text.secondary">
              {step.highlight || step.reason || "Key activation in the canonical path"}
            </Typography>
          </Paper>
        );
      })}
    </Stack>
  );
};

const personaTooltip = (persona: PersonaSummary) => {
  const phaseLabel = dominantPhaseFromProbs(persona.phase_probs, persona.dominant_phase);
  const pieces = [
    persona.perceptibility !== null && persona.perceptibility !== undefined
      ? `Perceptibility: ${fmtNumber(persona.perceptibility)}`
      : null,
    persona.proximity !== null && persona.proximity !== undefined
      ? `Proximity: ${fmtNumber(persona.proximity)}`
      : null,
    persona.involvement !== null && persona.involvement !== undefined
      ? `Involvement: ${fmtNumber(persona.involvement)}`
      : null,
    persona.path_probability !== null && persona.path_probability !== undefined
      ? `Path probability: ${fmtPercent(persona.path_probability)}`
      : null,
    persona.belief_level !== null && persona.belief_level !== undefined
      ? `Belief: ${fmtPercent(persona.belief_level)}`
      : null,
    persona.expected_next_prob !== null && persona.expected_next_prob !== undefined
      ? `Next persona prob: ${fmtPercent(persona.expected_next_prob)}`
      : null,
    persona.expected_in_deal_prob !== null &&
    persona.expected_in_deal_prob !== undefined
      ? `Committee likelihood: ${fmtPercent(persona.expected_in_deal_prob)}`
      : null,
    persona.fatigue !== null && persona.fatigue !== undefined
      ? `Fatigue: ${fmtPercent(persona.fatigue)}`
      : null,
    phaseLabel ? `Belief phase: ${phaseLabel}` : null,
  ].filter(Boolean);
  return pieces.join(" · ") || persona.label;
};

const personMatchLabel = (match: PersonMatch) => {
  const label =
    match.display_name ||
    match.person_name ||
    match.notes ||
    match.person_id ||
    "Unassigned person";
  const confidenceValue =
    typeof match.match_confidence === "number" && !Number.isNaN(match.match_confidence)
      ? match.match_confidence
      : null;
  const involvementValue =
    typeof match.person_involvement_score === "number" && !Number.isNaN(match.person_involvement_score)
      ? match.person_involvement_score
      : null;
  const probability =
    match.committee_probability ??
    match.person_involvement_score ??
    match.person_belief_level ??
    null;
  const dominantPhase = dominantPhaseFromProbs(
    match.belief_phase_probs,
    match.dominant_phase
  );
  const decoratedLabel = dominantPhase ? `${label} · ${dominantPhase}` : label;
  if (probability !== null && probability !== undefined) {
    return `${decoratedLabel} (${fmtPercent(probability)})`;
  }
  if (confidenceValue !== null && involvementValue !== null) {
    return `${decoratedLabel} (${fmtPercent(confidenceValue)}, ${fmtPercent(involvementValue)})`;
  }
  if (confidenceValue !== null) {
    return `${decoratedLabel} (${fmtPercent(confidenceValue)})`;
  }
  if (involvementValue !== null) {
    return `${decoratedLabel} (${fmtPercent(involvementValue)})`;
  }
  return decoratedLabel;
};

const personMatchTooltip = (match: PersonMatch) => {
  const phaseLabel = dominantPhaseFromProbs(
    match.belief_phase_probs,
    match.dominant_phase
  );
  const parts = [
    match.committee_probability !== null &&
    match.committee_probability !== undefined
      ? `Deal likelihood ${fmtPercent(match.committee_probability)}`
      : "",
    match.person_belief_level !== null &&
    match.person_belief_level !== undefined
      ? `Belief readiness ${fmtPercent(match.person_belief_level)}`
      : "",
    phaseLabel ? `Belief phase ${phaseLabel}` : "",
    match.person_involvement_score !== null &&
    match.person_involvement_score !== undefined
      ? `Engagement priority ${fmtPercent(match.person_involvement_score)}`
      : "",
    match.person_title,
    match.person_department,
    match.person_seniority,
    match.notes,
  ]
    .map((part) => (part ? part.trim() : ""))
    .filter(Boolean);
  if (typeof match.match_confidence === "number" && !Number.isNaN(match.match_confidence)) {
    parts.unshift(`Confidence ${fmtPercent(match.match_confidence)}`);
  }
  if (match.source) {
    parts.push(`Source: ${match.source}`);
  }
  return parts.join(" · ") || undefined;
};

const SummaryMetric: React.FC<{ label: string; value: string | number }> = ({
  label,
  value,
}) => (
  <Box sx={{ minWidth: 120 }}>
    <Typography variant="caption" color="text.secondary">
      {label}
    </Typography>
    <Typography variant="h6">{value}</Typography>
  </Box>
);

type AggregatedFocusAssetRow = {
  asset: AssetRec;
  asset_type?: string | null;
  asset_type_label?: string | null;
  asset_fit_score?: number | null;
  asset_expected_delta_bp?: number | null;
  channel: CampaignAssetChannel;
  channel_type?: string | null;
  channel_type_label?: string | null;
  expected_delta_bp: number;
  belief_conversion_likelihood?: number | null;
  confidence?: number | null;
  evidence_count?: number | null;
  play_count?: number | null;
  engagement_score?: number | null;
  exploration_weight?: number | null;
  avg_duration_days?: number | null;
};

type PersonaDescriptorCount = {
  descriptor: string;
  count: number;
};

type ConversionSequenceEntry = {
  timeline_index: number;
  timeline_label?: string;
  timeline_days?: number;
  persona?: PersonaMeta;
  persona_descriptor?: string;
  stage_label?: string | null;
  belief_transition_meta: {
    stage?: string | null;
    stage_label?: string | null;
    problem?: string | null;
    pain?: string | null;
    resolution?: string | null;
    from_to: BeliefEdge[];
  };
  expected_delta_bp?: number | null;
  avg_confidence?: number | null;
  avg_belief_conversion?: number | null;
  effort_required?: number | null;
  effort_pct?: number | null;
  time_to_effect_days?: number | null;
  segment_filters: SegmentFilters;
  segment_summary?: string | null;
  expected_outcome_summary: string;
  asset_table: AggregatedFocusAssetRow[];
  people?: string[];
  matched_people?: PersonMatch[];
  plays?: Play[];
};

type PortfolioConversionFocus = {
  key: string;
  label?: string | null;
  focus_label?: string | null;
  persona_focus?: string | null;
  persona_label?: string | null;
  stage_label?: string | null;
  campaign_theme?: string | null;
  expected_delta_bp: number;
  confidence?: number | null;
  accounts: Array<{ id: string; name: string }>;
  account_count: number;
  people?: string[];
  asset_table?: AggregatedFocusAssetRow[];
  avg_duration_days: number;
  coverage_ratio?: number | null;
  avg_effort_required?: number | null;
  avg_effort_pct?: number | null;
  avg_belief_conversion?: number | null;
  time_to_effect_days?: number | null;
  timeline_index?: number | null;
  timeline_label?: string | null;
  segment_filters?: SegmentFilters;
  segment_summary?: string | null;
  persona_descriptor?: string | null;
  persona_descriptor_counts?: PersonaDescriptorCount[];
  expected_outcome_summary?: string;
};

type PortfolioThemeGroup = PortfolioTheme & {
  focuses: PortfolioConversionFocus[];
};

type ExpectedOutcomeSummary = {
  segment_filters: SegmentFilters;
  segment_summary?: string | null;
  persona_descriptor_counts?: PersonaDescriptorCount[];
  expected_outcome_summary: string;
  total_delta_bp: number;
  quarter?: string;
  expected_delta_bp?: number;
  conversion_ready_accounts?: number;
};

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------

const MarketingPlanner: React.FC = () => {
  const token =
    typeof window !== "undefined" ? localStorage.getItem("token") : null;

  const [products, setProducts] = useState<Array<{ id: string; name: string }>>(
    []
  );
  const [selectedProductId, setSelectedProductId] = useState<string>("");
  const [plan, setPlan] = useState<MarketingPlan | null>(null);
  const [viewAccountId, setViewAccountId] = useState<string>("portfolio");
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [statusMsg, setStatusMsg] = useState<string>("");
  const [plannerPanel, setPlannerPanel] = useState<"overview" | "timeline" | "arsenal">("overview");
  const [arsenalPreset, setArsenalPreset] = useState<string | null>(null);
  const [openArsenalReasoning, setOpenArsenalReasoning] = useState<string | null>(null);
  const [accountPopover, setAccountPopover] = useState<{
    anchorEl: HTMLElement | null;
    focusId: string | null;
  }>({ anchorEl: null, focusId: null });

  useEffect(() => {
    if (token) {
      (async () => {
        try {
          const meRes = await fetch("http://localhost:8000/me", {
            headers: { Authorization: `Bearer ${token}` },
          });
          if (!meRes.ok) throw new Error(await meRes.text());
          const me = await meRes.json();

          const prodRes = await fetch(
            `http://localhost:8000/get-products/${me.company_id}`,
            { headers: { Authorization: `Bearer ${token}` } }
          );
          if (!prodRes.ok) throw new Error(await prodRes.text());
          const prodData = await prodRes.json();
          const list = prodData.products ?? [];
          if (!list.length) {
            setStatusMsg("No products found. Please run Value Prop first.");
            return;
          }
          setProducts(list);
          setSelectedProductId((prev) => prev || list[0].id);
        } catch (err: any) {
          console.error(err);
          setError(err.message || "Failed to load company or products.");
        }
      })();
    }
  }, [token]);

  useEffect(() => {
    if (!selectedProductId || !token) return;
    setLoading(true);
    setError(null);
    (async () => {
      try {
        const res = await fetch(
          `http://localhost:8000/get-comprehensive-execution-plan/${selectedProductId}`,
          { headers: { Authorization: `Bearer ${token}` } }
        );
        if (!res.ok) throw new Error(await res.text());
        const data: MarketingPlan = await res.json();
        setPlan(data);
        setStatusMsg(
          `Marketing blueprint generated ${new Date(
            data.generated_at
          ).toLocaleString()}`
        );
      } catch (err: any) {
        console.error(err);
        setError(err.message || "Failed to build marketing plan.");
        setPlan(null);
      } finally {
        setLoading(false);
      }
    })();
  }, [selectedProductId, token]);

  useEffect(() => {
    if (!plan) {
      if (viewAccountId !== "portfolio") {
        setViewAccountId("portfolio");
      }
      return;
    }
    if (viewAccountId === "portfolio") return;
    const exists = plan.accounts?.some(
      (account) => account.account_id === viewAccountId
    );
    if (!exists) {
      setViewAccountId("portfolio");
    }
  }, [plan, viewAccountId]);

  const accounts = plan?.accounts ?? [];

  const selectedAccountPlan = useMemo(() => {
    if (!plan || viewAccountId === "portfolio") {
      return null;
    }
    return (
      plan.accounts.find((account) => account.account_id === viewAccountId) ||
      null
    );
  }, [plan, viewAccountId]);

  const summaryTitle = selectedAccountPlan
    ? `${selectedAccountPlan.account_name || "Account"} Summary`
    : "Portfolio Summary";

  const canonicalTypicalPath = useMemo(() => {
    if (!plan) return [];
    return (
      plan.canonical_persona_path ||
      plan.portfolio_plan?.typical_path ||
      plan.summary?.typical_path ||
      []
    );
  }, [plan]);

  const derivedPortfolioSummary = useMemo<PortfolioSummaryReport | null>(() => {
    if (!plan) return null;
    const planAccounts = plan.accounts || [];
    if (!planAccounts.length) return null;

    const coverageRatios: number[] = [];
    const beliefScores: number[] = [];
    const maturityScores: number[] = [];
    const stageCounts: Record<string, number> = {};
    let totalAssets = 0;
    let totalDelta = 0;
    const channelSet = new Set<string>();
    const timelineDays: number[] = [];
    const personaAssetCounter: Record<string, number> = {};
    let fatigueAlerts = 0;

    planAccounts.forEach((account) => {
      const coverage = account.enrichment?.summary?.coverage_ratio;
      if (typeof coverage === "number" && !Number.isNaN(coverage)) {
        coverageRatios.push(coverage);
      }

      const primaryProb = account.prediction?.persona_paths?.[0]?.probability;
      if (typeof primaryProb === "number" && !Number.isNaN(primaryProb)) {
        beliefScores.push(primaryProb);
      }

      const totalSteps = account.prediction?.journey?.total_steps || 0;
      const observedSteps = (account.prediction?.journey?.steps || []).length;
      if (totalSteps > 0) {
        maturityScores.push(Math.min(1, observedSteps / totalSteps));
      }

      (account.execution?.conversion_sequence || []).forEach((entry) => {
        const stage =
          normalizeStageLabel(entry.belief_transition_meta?.stage_label) ||
          normalizeStageLabel(entry.stage_label) ||
          "Next Stage";
        stageCounts[stage] = (stageCounts[stage] || 0) + 1;
      });

      const plays = account.execution?.plays || [];
      totalAssets += plays.length;
      plays.forEach((play) => {
        const delta = typeof play.expected_delta_bp === "number" ? play.expected_delta_bp : 0;
        totalDelta += delta;
        const channelId =
          play.channel?.id ||
          play.channel_id ||
          play.channel_label ||
          play.channel?.name ||
          play.channel_type ||
          play.channel_type_label;
        if (channelId) channelSet.add(channelId);
        const personaLabel = play.persona_label || play.persona_descriptor || play.persona_focus;
        if (personaLabel) {
          personaAssetCounter[personaLabel] = (personaAssetCounter[personaLabel] || 0) + 1;
        }
      });

      const lastTimelineEntry = (account.execution?.conversion_sequence || []).slice(-1)[0];
      if (lastTimelineEntry && typeof lastTimelineEntry.timeline_days === "number") {
        timelineDays.push(lastTimelineEntry.timeline_days);
      }

      (account.execution?.persona_engagements || []).forEach((entry) => {
        if ((entry.fatigue ?? 0) >= 0.6) {
          fatigueAlerts += 1;
        }
      });
    });

    const stageDistribution: StageDistributionRow[] = Object.entries(stageCounts)
      .map(([stage, count]) => ({ stage, count }))
      .sort((a, b) => {
        const orderDiff =
          (STAGE_ORDER[a.stage.toLowerCase()] ?? Number.MAX_SAFE_INTEGER) -
          (STAGE_ORDER[b.stage.toLowerCase()] ?? Number.MAX_SAFE_INTEGER);
        if (orderDiff !== 0) return orderDiff;
        return b.count - a.count;
      });

    const personaMix = plan.summary?.top_personas || [];
    const painMix = plan.summary?.top_pains || [];
    const capabilityMix = plan.summary?.top_capabilities || [];

    const highRiskAccounts = planAccounts
      .map((account) => {
        const coverage = account.enrichment?.summary?.coverage_ratio ?? 0;
        const fatigueValues =
          account.execution?.persona_engagements?.map((entry) => entry.fatigue ?? 0) || [];
        const avgFatigue = avgOrNull(fatigueValues) ?? 0;
        const blockerCount =
          account.enrichment?.unmatched_personas?.length || 0;
        const score = (1 - coverage) + avgFatigue + blockerCount * 0.2;
        const reasons: string[] = [];
        if (coverage < 0.6) reasons.push(`Coverage ${fmtPercent(coverage)}`);
        if (avgFatigue > 0.45)
          reasons.push(`Fatigue ${Math.round(avgFatigue * 100)}%`);
        if (blockerCount)
          reasons.push(`${blockerCount} persona gap${blockerCount === 1 ? "" : "s"}`);
        return {
          id: account.account_id,
          name: account.account_name || account.account_id,
          reason: reasons.join(" · ") || "Monitoring",
          score,
        };
      })
      .sort((a, b) => b.score - a.score)
      .slice(0, 4);

    const highOpportunityAccounts = planAccounts
      .map((account) => {
        const coverage = account.enrichment?.summary?.coverage_ratio ?? 0;
        const primaryProb = account.prediction?.persona_paths?.[0]?.probability ?? 0;
        const timeline = (account.execution?.conversion_sequence || []).slice(-1)[0];
        const daysRemaining = timeline?.timeline_days ?? 90;
        const opportunityScore =
          coverage * 0.4 + primaryProb * 0.4 + (1 - Math.min(daysRemaining / 120, 1)) * 0.2;
        const reasons = [
          `Coverage ${fmtPercent(coverage)}`,
          `Path p=${fmtPercent(primaryProb)}`,
          `ETA ${toWeeks(daysRemaining)}`,
        ];
        return {
          id: account.account_id,
          name: account.account_name || account.account_id,
          reason: reasons.join(" · "),
          score: opportunityScore,
        };
      })
      .sort((a, b) => b.score - a.score)
      .slice(0, 4);

    const assetsPerPersona = Object.entries(personaAssetCounter)
      .map(([persona, count]) => ({ persona, count }))
      .sort((a, b) => b.count - a.count)
      .slice(0, 8);

    return {
      totalAccounts: planAccounts.length,
      coveragePct: avgOrNull(coverageRatios),
      enrichedAccounts: coverageRatios.filter((ratio) => ratio >= 0.65).length,
      beliefScore: avgOrNull(beliefScores),
      maturityScore: avgOrNull(maturityScores),
      assetsPlanned: totalAssets,
      channelCount: channelSet.size,
      totalDeltaBp: totalDelta,
      expectedConversionRate:
        plan.portfolio_plan?.randomization?.avg_path_probability ??
        avgOrNull(beliefScores),
      expectedTimeToConversionDays:
        timelineDays.length > 0
          ? timelineDays.reduce((sum, value) => sum + value, 0) / timelineDays.length
          : null,
      stageDistribution,
      highRiskAccounts,
      highOpportunityAccounts,
      personaMix,
      painMix,
      capabilityMix,
      fatigueAlerts,
      assetsPerPersona,
    };
  }, [plan]);

  const backendPortfolioSummary = plan?.summary?.portfolio_summary_report ?? null;
  const portfolioSummaryReport = useMemo<PortfolioSummaryReport | null>(() => {
    const mapped = mapBackendPortfolioSummary(backendPortfolioSummary);
    if (mapped) return mapped;
    return derivedPortfolioSummary;
  }, [backendPortfolioSummary, derivedPortfolioSummary]);
  const portfolioAvgAccuracy = plan?.summary?.avg_accuracy ?? null;
  const portfolioEnrichmentRatio =
    portfolioSummaryReport && portfolioSummaryReport.totalAccounts
      ? portfolioSummaryReport.enrichedAccounts / portfolioSummaryReport.totalAccounts
      : null;

  const bestPathValue = fmtPercent(
    selectedAccountPlan
      ? selectedAccountPlan.prediction?.persona_paths?.[0]?.probability ?? null
      : portfolioSummaryReport?.beliefScore ??
        plan?.summary?.avg_best_path_probability ??
        null
  );

  const accuracyValue = fmtPercent(
    selectedAccountPlan
      ? selectedAccountPlan.prediction?.fit?.accuracy ?? null
      : portfolioAvgAccuracy ?? null
  );

  const accountLiftBp = selectedAccountPlan
    ? (selectedAccountPlan.campaigns || []).reduce(
        (sum, campaign) => sum + (campaign.total_delta_bp || 0),
        0
      )
    : plan?.summary?.total_expected_delta_bp ?? 0;
  const liftValue = fmtBasisPoints(accountLiftBp);

  const coverageRatio = selectedAccountPlan
    ? selectedAccountPlan.enrichment?.summary?.coverage_ratio ?? null
    : plan?.summary?.avg_people_coverage ?? null;
  const coverageValue = fmtPercent(coverageRatio);
  const matchedPeopleCount =
    selectedAccountPlan?.enrichment?.summary?.personas_with_matches ?? 0;
  const requiredPersonaCount =
    selectedAccountPlan?.enrichment?.summary?.required_personas ?? 0;

  const accountMix = useMemo(() => {
    if (!selectedAccountPlan) return null;
    let broad = 0;
    let focused = 0;
    (selectedAccountPlan.execution?.plays || []).forEach((play) => {
      if (play.mode === "broad") {
        broad += 1;
      } else {
        focused += 1;
      }
    });
    const total = broad + focused;
    return {
      broad,
      focused,
      ratio: total ? broad / total : null,
    };
  }, [selectedAccountPlan]);

  const unmatchedPersonaReqs =
    selectedAccountPlan?.enrichment?.persona_requirements?.filter(
      (req) => !req.has_match
    ) || [];

  const zmotWatchlist = selectedAccountPlan?.zmot?.watchlist || [];
  const zmotEventPortfolio = selectedAccountPlan?.zmot?.events || [];

  const portfolioMix = plan?.portfolio_plan?.broad_focus_mix;
  const portfolioRandomization = plan?.portfolio_plan?.randomization;
  const accountConversionSequence =
    selectedAccountPlan?.execution?.conversion_sequence || [];
  const accountAssetCadence =
    selectedAccountPlan?.execution?.asset_cadence || [];
  const accountCampaigns = selectedAccountPlan?.campaigns || [];

  const portfolioFocuses = useMemo<PortfolioConversionFocus[]>(() => {
    const focuses = plan?.portfolio_plan?.conversion_focuses ?? [];
    return focuses
      .map((focus) => {
        const avgDuration =
          typeof focus.avg_duration_days === "number" && !Number.isNaN(focus.avg_duration_days)
            ? focus.avg_duration_days
            : 14;
        const timeToEffect =
          typeof focus.time_to_effect_days === "number" && !Number.isNaN(focus.time_to_effect_days)
            ? focus.time_to_effect_days
            : avgDuration;
        return {
          ...focus,
          segment_filters: focus.segment_filters ?? {},
          avg_duration_days: avgDuration,
          time_to_effect_days: timeToEffect,
        };
      })
      .sort(
        (a, b) =>
          (a.time_to_effect_days ?? a.avg_duration_days ?? 14) -
          (b.time_to_effect_days ?? b.avg_duration_days ?? 14)
      );
  }, [plan?.portfolio_plan?.conversion_focuses]);

  const portfolioFocusByKey = useMemo(() => {
    const map = new Map<string, PortfolioConversionFocus>();
    portfolioFocuses.forEach((focus) => {
      if (focus.key) {
        map.set(focus.key, focus);
      }
    });
    return map;
  }, [portfolioFocuses]);

  const portfolioFocusMap = useMemo(() => {
    const map = new Map<string, PortfolioConversionFocus>();
    portfolioFocuses.forEach((focus) => {
      map.set(focus.key, focus);
    });
    return map;
  }, [portfolioFocuses]);

  const portfolioThemes = useMemo<PortfolioThemeGroup[]>(() => {
    const rawThemes = plan?.portfolio_plan?.campaign_themes ?? [];
    const assigned = new Set<string>();

    const themed = rawThemes.map((theme) => {
      const focusKeys = theme.focus_keys ?? [];
      let focuses = focusKeys
        .map((id) => (id ? portfolioFocusByKey.get(id) : undefined))
        .filter((focus): focus is PortfolioConversionFocus => Boolean(focus));

      if (!focuses.length && Array.isArray(theme.conversion_focuses)) {
        focuses = theme.conversion_focuses
          .map((focus) => {
            if (focus.key && portfolioFocusByKey.has(focus.key)) {
              return portfolioFocusByKey.get(focus.key)!;
            }
            if (focus.key) {
              return focus;
            }
            const fallbackKey =
              focus.persona_focus ||
              focus.persona_label ||
              focus.focus_label ||
              focus.label ||
              "focus";
            return {
              ...focus,
              key: `${theme.quarter}-${theme.theme}-${fallbackKey}`,
            };
          })
          .filter((focus): focus is PortfolioConversionFocus => Boolean(focus));
      }

      focuses.forEach((focus) => assigned.add(focus.key));
      return { ...theme, focuses };
    });

    const unassigned = portfolioFocuses.filter((focus) => !assigned.has(focus.key));
    if (unassigned.length) {
      const accountMap = new Map<string, { id: string; name: string }>();
      unassigned.forEach((focus) => {
        focus.accounts.forEach((acct) => {
          if (acct.id && !accountMap.has(acct.id)) {
            accountMap.set(acct.id, acct);
          }
        });
      });
      themed.push({
        quarter: "All",
        theme: "General Conversion Focuses",
        accounts: Array.from(accountMap.values()),
        focus_personas: [],
        focus_people: [],
        focus_pains: [],
        total_delta_bp: unassigned.reduce(
          (sum, focus) => sum + (focus.expected_delta_bp || 0),
          0
        ),
        avg_confidence:
          avgOrNull(unassigned.map((focus) => focus.confidence ?? null)) ?? 0,
        play_count: unassigned.length,
        focus_keys: unassigned.map((focus) => focus.key),
        focuses: unassigned,
        segment_filters: {},
        segment_summary: undefined,
        persona_descriptor_counts: [],
        expected_outcome_summary: undefined,
      });
    }

    const themeOrder = (theme: PortfolioThemeGroup) => {
      if (!theme.focuses.length) return Number.MAX_SAFE_INTEGER;
      return Math.min(...theme.focuses.map((focus) => focus.avg_duration_days ?? 14));
    };

    return themed.sort((a, b) => {
      const orderDiff = themeOrder(a) - themeOrder(b);
      if (orderDiff !== 0) return orderDiff;
      const quarterDiff =
        parseInt(String(a.quarter).replace(/\D/g, "") || "0", 10) -
        parseInt(String(b.quarter).replace(/\D/g, "") || "0", 10);
      if (quarterDiff !== 0) return quarterDiff;
      return a.theme.localeCompare(b.theme);
    });
  }, [plan?.portfolio_plan?.campaign_themes, portfolioFocusByKey, portfolioFocuses]);

  const portfolioJourneySteps = useMemo(() => {
    if (canonicalTypicalPath.length) {
      return canonicalTypicalPath;
    }
    return portfolioFocuses.slice(0, 6).map((focus, idx) => ({
      step: idx + 1,
      persona:
        focus.persona_label ||
        focus.persona_focus ||
        focus.label ||
        `Persona ${idx + 1}`,
      highlight:
        focus.expected_outcome_summary ||
        focus.segment_summary ||
        focus.campaign_theme ||
        undefined,
      reason: focus.stage_label || undefined,
      confidence: focus.confidence || undefined,
    }));
  }, [canonicalTypicalPath, portfolioFocuses]);

  const portfolioFocusContext = useMemo(() => {
    const map = new Map<
      string,
      { focus: PortfolioConversionFocus; theme: PortfolioThemeGroup | null }
    >();

    const attach = (
      focus: PortfolioConversionFocus,
      theme: PortfolioThemeGroup | null
    ) => {
      const descriptor =
        focus.persona_descriptor ||
        focus.persona_focus ||
        focus.focus_label ||
        focus.label;
      const key = focusMatchKey(descriptor, focus.stage_label);
      if (!key) return;
      if (!map.has(key)) {
        map.set(key, { focus, theme });
      }
    };

    portfolioThemes.forEach((theme) => {
      theme.focuses.forEach((focus) => attach(focus, theme));
    });
    portfolioFocuses.forEach((focus) => attach(focus, null));

    return map;
  }, [portfolioThemes, portfolioFocuses]);

  const derivedThesisData = useMemo<ThesisData | null>(() => {
    if (!plan || !portfolioSummaryReport) return null;
    const typicalPath =
      plan.canonical_persona_path ||
      plan.portfolio_plan?.typical_path ||
      plan.summary?.typical_path ||
      [];
    const personaSequence = typicalPath.map((step, idx) => {
      const label = (step as any).label || step.persona || (step as any).title;
      return label ? `${idx + 1}. ${label}` : `Step ${idx + 1}`;
    });

    const startingPains = (plan.summary?.top_pains || [])
      .slice(0, 3)
      .map((item) => item.label);
    const dominantCapabilities = (plan.summary?.top_capabilities || [])
      .slice(0, 3)
      .map((item) => item.label);
    const personaThemes = (plan.summary?.top_personas || [])
      .slice(0, 3)
      .map((item) => item.label);
    const blockers = portfolioSummaryReport.highRiskAccounts.map((acct) => ({
      id: acct.id,
      name: acct.name,
      reason: acct.reason,
    }));
    const beliefTargets = portfolioSummaryReport.stageDistribution;
    const segments = portfolioThemes
      .map((theme) => theme.segment_summary || "")
      .filter((text): text is string => Boolean(text));

    const strategyHighlights = [
      startingPains.length
        ? `Starting pains concentrate on ${startingPains.join(", ")}.`
        : "Starting pains remain distributed; prioritize discovery interviews.",
      dominantCapabilities.length
        ? `Dominant capabilities/themes: ${dominantCapabilities.join(", ")}.`
        : "Capabilities need definition across campaigns.",
      personaThemes.length
        ? `Most engaged personas: ${personaThemes.join(", ")}.`
        : "Persona engagement data is still sparse.",
      beliefTargets.length
        ? `Belief load concentrates in ${beliefTargets[0].stage} with ${beliefTargets[0].count} personas queued.`
        : "Belief progression is evenly distributed.",
      blockers.length
        ? `High-risk accounts (${blockers.length}) show fatigue or coverage gaps—pre-wire legal/IT early.`
        : "No portfolio-level blockers surfaced yet.",
    ];

    const quarterOutcomes =
      plan.portfolio_plan?.expected_outcomes?.by_quarter?.map((outcome) => ({
        quarter: outcome.quarter || "Quarter",
        summary: outcome.expected_outcome_summary || "Outcome pending",
        deltaBp: outcome.expected_delta_bp,
        conversionReadyCount: outcome.conversion_ready_accounts,
        personaDescriptors: outcome.persona_descriptor_counts,
        segmentSummary: outcome.segment_summary,
      })) || [];

    return {
      strategyHighlights,
      personaSequence,
      blockers,
      beliefTargets,
      segments,
      quarterOutcomes,
      personaCadence: plan.summary?.persona_engagements || [],
    };
  }, [plan, portfolioSummaryReport, portfolioThemes]);

  const backendThesis = plan?.summary?.portfolio_thesis ?? null;
  const personaCadenceFallback = plan?.summary?.persona_engagements;
  const thesisData = useMemo<ThesisData | null>(() => {
    const mapped = mapBackendThesisData(backendThesis, personaCadenceFallback);
    if (mapped) return mapped;
    return derivedThesisData;
  }, [backendThesis, personaCadenceFallback, derivedThesisData]);

  const derivedArsenalRows = useMemo<ArsenalRow[]>(() => {
    if (!plan) return [];
    const rows: ArsenalRow[] = [];
    plan.accounts?.forEach((account) => {
      const accountName = account.account_name || account.account_id;
      account.campaigns?.forEach((campaign) => {
        const theme = campaign.theme || "Campaign";
        const quarter = campaign.quarter || "Q1";
        (campaign.plays || []).forEach((play) => {
          const assetName =
            play.asset?.name ||
            play.asset_label ||
            play.asset?.title ||
            "Asset";
          const channelName =
            play.channel?.name ||
            play.channel_label ||
            play.channel_type ||
            "Channel";
        const persona =
          play.persona_label ||
          play.persona_descriptor ||
          campaign.persona_focus ||
          "Target persona";
        const accountsInPlay =
          play.accounts?.map((acc: any) => acc.name).filter(Boolean) || [];
        const beliefTransition = (play.belief_transition ?? null) as
          | BeliefTransitionDetail
          | null;
        const stageLabel = stageLabelFromTransition(beliefTransition);
        const segmentFilters = (play.segment_filters ?? undefined) as
          | SegmentFilters
          | undefined;
        const segmentSummary = summarizeSegmentFilters(segmentFilters);
        const assetMeta = (play.asset ?? null) as AssetRec | null;
        const channelMeta = (play.channel ?? null) as ChannelRec | null;
        const rationale = (play.rationale ?? null) as ArsenalRationale | null;
        const reasons = (play.reasons ?? null) as Record<string, any> | null;
        const row: ArsenalRow = {
          theme,
          quarter,
          asset: assetName,
          channel: channelName,
          persona,
          accounts: Array.from(
            new Set([accountName, ...accountsInPlay].filter(Boolean))
          ),
          deltaBp: play.expected_delta_bp ?? 0,
          confidence:
            typeof play.confidence === "number" ? play.confidence : null,
          beliefConversion:
            play.belief_conversion_likelihood ??
            play.avg_belief_conversion ??
            null,
          timeToImpactDays:
            play.time_to_effect_days ??
            campaign.time_to_effect_days ??
            play.duration_days ??
            null,
          mode: play.mode,
          stageLabel: stageLabel || undefined,
          stageIndex:
            typeof play.stage_index === "number" ? play.stage_index : null,
          beliefTransition,
          assetMeta,
          channelMeta,
          rationale,
          reasons,
          segmentFilters,
          segmentSummary,
          personaId: (play.persona_id as string | null) ?? null,
          accountId: (account.account_id as string | null) ?? null,
          accountName:
            account.account_name || account.account_id || accountName || null,
        };
        rows.push(row);
      });
    });
  });
    return rows.sort((a, b) => {
      const quarterDiff = quarterOrder(a.quarter) - quarterOrder(b.quarter);
      if (quarterDiff !== 0) return quarterDiff;
      if (a.theme !== b.theme) return a.theme.localeCompare(b.theme);
      return b.deltaBp - a.deltaBp;
    });
  }, [plan]);
  const backendArsenalRows = useMemo(
    () => mapBackendArsenalRows(plan?.portfolio_plan?.arsenal_table),
    [plan?.portfolio_plan?.arsenal_table]
  );
  const portfolioArsenalRows =
    backendArsenalRows.length > 0 ? backendArsenalRows : derivedArsenalRows;
  const portfolioPersonaEngagements = useMemo<PersonaEngagementPlan[]>(() => {
    if (!plan?.accounts?.length) return [];
    return plan.accounts.flatMap(
      (account) => account.execution?.persona_engagements || []
    );
  }, [plan]);

  const aggregatedPortfolioArsenal = useMemo(
    () => aggregateArsenalEntries(portfolioArsenalRows),
    [portfolioArsenalRows]
  );
  const filteredArsenalEntries = useMemo(() => {
    return aggregatedPortfolioArsenal.filter((entry) => {
      if (arsenalPreset === "legalExec") {
        return entry.personas.some((persona) =>
          persona.toLowerCase().includes("legal") ||
          persona.toLowerCase().includes("compliance")
        );
      }
      if (arsenalPreset === "analyticsOps") {
        return entry.personas.some((persona) =>
          persona.toLowerCase().includes("analyst") ||
          persona.toLowerCase().includes("data")
        );
      }
      if (arsenalPreset === "highLift") {
        return entry.deltaBp >= 300;
      }
      if (arsenalPreset === "shortDuration") {
        return (entry.avgDurationDays ?? 999) <= 14;
      }
      return true;
    });
  }, [aggregatedPortfolioArsenal, arsenalPreset]);

  const portfolioCadenceStats = useMemo(
    () => buildPortfolioCadenceStats(portfolioPersonaEngagements),
    [portfolioPersonaEngagements]
  );

  const derivedExecutionMatrix = useMemo<ExecutionMatrix | null>(() => {
    if (!plan) return null;

    const personaMetrics = new Map<
      string,
      { fatigueValues: number[]; beliefValues: number[] }
    >();
    plan.accounts?.forEach((account) => {
      (account.execution?.persona_engagements || []).forEach((entry) => {
        const label = entry.persona_label || entry.persona_id;
        if (!label) return;
        if (!personaMetrics.has(label)) {
          personaMetrics.set(label, { fatigueValues: [], beliefValues: [] });
        }
        const bucket = personaMetrics.get(label)!;
        if (typeof entry.fatigue === "number" && !Number.isNaN(entry.fatigue)) {
          bucket.fatigueValues.push(entry.fatigue);
        }
        if (
          typeof entry.belief_level === "number" &&
          !Number.isNaN(entry.belief_level)
        ) {
          bucket.beliefValues.push(entry.belief_level);
        }
      });
    });

    const quarterSet = new Set<string>();
    const personaSet = new Set<string>();
    const personaQuarterBuckets = new Map<
      string,
      Map<
        string,
        {
          accounts: number;
          campaignIds: Set<string>;
          campaigns: ExecutionCell["campaigns"];
          totalDelta: number;
          totalAssets: number;
          broadTouches: number;
          focusedTouches: number;
          fatigueValues: number[];
          beliefValues: number[];
        }
      >
    >();

    plan.accounts?.forEach((account) => {
      account.campaigns?.forEach((campaign, idx) => {
        const quarter =
          campaign.quarter ||
          mapDayToQuarter(campaign.start_day) ||
          `Q${(idx % 4) + 1}`;
        quarterSet.add(quarter);
        const personas = campaign.focus_personas?.length
          ? campaign.focus_personas
          : [campaign.persona_focus || "Multi-persona"];
        const accountCount = Math.max(1, campaign.accounts?.length ?? 0);
        const campaignId = [
          campaign.theme || "campaign",
          campaign.belief_transitions?.[0]?.stage_label || "",
          quarter,
        ].join("|");
        const mode =
          (campaign.mode_mix?.focused ?? 0) >= (campaign.mode_mix?.broad ?? 0)
            ? "focused"
            : "broad";

        personas.forEach((personaLabel) => {
          personaSet.add(personaLabel);
          if (!personaQuarterBuckets.has(personaLabel)) {
            personaQuarterBuckets.set(personaLabel, new Map());
          }
          const quarterMap = personaQuarterBuckets.get(personaLabel)!;
          if (!quarterMap.has(quarter)) {
            quarterMap.set(quarter, {
              accounts: 0,
              campaignIds: new Set(),
              campaigns: [],
              totalDelta: 0,
              totalAssets: 0,
              broadTouches: 0,
              focusedTouches: 0,
              fatigueValues: [],
              beliefValues: [],
            });
          }
          const bucket = quarterMap.get(quarter)!;
          bucket.accounts += accountCount;

          if (!bucket.campaignIds.has(campaignId)) {
            bucket.campaignIds.add(campaignId);
            bucket.campaigns.push({
              id: campaignId,
              theme: campaign.theme || "Campaign",
              description: campaign.expected_outcome_summary,
              deltaBp: campaign.total_delta_bp ?? 0,
              assets: campaign.plays?.length ?? 0,
              belief: campaign.avg_confidence ?? null,
            });
            bucket.totalDelta += campaign.total_delta_bp || 0;
            bucket.totalAssets += campaign.plays?.length || 0;
            bucket.broadTouches += campaign.mode_mix?.broad ?? 0;
            bucket.focusedTouches += campaign.mode_mix?.focused ?? 0;
          }

          const metrics = personaMetrics.get(personaLabel);
          if (metrics) {
            bucket.fatigueValues.push(...metrics.fatigueValues);
            bucket.beliefValues.push(...metrics.beliefValues);
          }
        });
      });
    });

    const cells: Record<string, ExecutionCell[]> = {};
    personaQuarterBuckets.forEach((quarterMap, personaLabel) => {
      quarterMap.forEach((bucket, quarter) => {
        const fatigue =
          bucket.fatigueValues.length > 0
            ? bucket.fatigueValues.reduce((sum, value) => sum + value, 0) /
              bucket.fatigueValues.length
            : null;
        const belief =
          bucket.beliefValues.length > 0
            ? bucket.beliefValues.reduce((sum, value) => sum + value, 0) /
              bucket.beliefValues.length
            : null;
        const key = `${quarter}|${personaLabel}`;
        const dominantMode =
          bucket.focusedTouches >= bucket.broadTouches ? "focused" : "broad";
        const campaignsSorted = bucket.campaigns
          .slice()
          .sort((a, b) => (b.deltaBp ?? 0) - (a.deltaBp ?? 0));
        cells[key] = [
          {
            theme: campaignsSorted[0]?.theme || "Campaign mix",
            deltaBp: bucket.totalDelta,
            mode: dominantMode,
            accountCount: bucket.accounts,
            fatigue,
            belief,
            assets: bucket.totalAssets,
            campaigns: campaignsSorted,
          },
        ];
      });
    });

    const quarters = Array.from(quarterSet).sort((a, b) => quarterOrder(a) - quarterOrder(b));
    const personas = Array.from(personaSet).sort((a, b) => a.localeCompare(b));
    if (!quarters.length || !personas.length) return null;
    return { quarters, personas, cells };
  }, [plan]);

  const planExecutionMatrix = plan?.portfolio_plan?.execution_matrix ?? null;
  const backendExecutionMatrix = useMemo(
    () => mapBackendExecutionMatrix(planExecutionMatrix),
    [planExecutionMatrix]
  );
  const executionMatrix = backendExecutionMatrix ?? derivedExecutionMatrix;
  const accountExecutionMatrix = useMemo<ExecutionMatrix | null>(() => {
    if (!selectedAccountPlan) return null;
    const quarterSet = new Set<string>();
    const personaSet = new Set<string>();
    const cells: Record<string, ExecutionCell[]> = {};
    const fatigueByPersona = new Map<string, number>();
    (selectedAccountPlan.execution?.persona_engagements || []).forEach((entry) => {
      if (entry.persona_label) {
        fatigueByPersona.set(entry.persona_label, entry.fatigue ?? 0);
      }
    });
    (selectedAccountPlan.campaigns || []).forEach((campaign) => {
      const quarter = campaign.quarter || "Q1";
      quarterSet.add(quarter);
      const personas = campaign.focus_personas?.length
        ? campaign.focus_personas
        : [campaign.persona_focus || "Multi-persona"];
      personas.forEach((persona) => {
        personaSet.add(persona);
        const key = `${quarter}|${persona}`;
        if (!cells[key]) cells[key] = [];
        const modeMix = campaign.mode_mix || { broad: 0, focused: 0 };
        cells[key].push({
          theme: campaign.theme || "Campaign",
          deltaBp: campaign.total_delta_bp || 0,
          mode: (modeMix.focused || 0) >= (modeMix.broad || 0) ? "focused" : "broad",
          accountCount: 1,
          fatigue: fatigueByPersona.get(persona),
          belief: campaign.confidence ?? null,
          assets: campaign.plays?.length || 0,
        });
      });
    });
    const quarters = Array.from(quarterSet).sort((a, b) => quarterOrder(a) - quarterOrder(b));
    const personas = Array.from(personaSet).sort((a, b) => a.localeCompare(b));
    if (!quarters.length || !personas.length) return null;
    return { quarters, personas, cells };
  }, [selectedAccountPlan]);
  const executionMatrixForView = selectedAccountPlan ? accountExecutionMatrix : executionMatrix;

  const personaLabelLookup = useMemo(() => {
    const lookup: Record<string, string> = {};
    const register = (id?: string | null, label?: string | null) => {
      if (!id || !label) return;
      lookup[id] = label;
    };
    plan?.summary?.persona_engagements?.forEach((entry) =>
      register(entry.persona_id, entry.persona_label)
    );
    plan?.accounts?.forEach((account) => {
      account.prediction?.persona_paths?.forEach((path) => {
        path.personas.forEach((persona) => register(persona.id, persona.label));
      });
      account.execution?.persona_engagements?.forEach((entry) =>
        register(entry.persona_id, entry.persona_label)
      );
    });
    return lookup;
  }, [plan]);

const counterToItems = (
  counter: Map<string, number>,
  limit = 6
): Array<{ label: string; count: number }> =>
  Array.from(counter.entries())
    .map(([label, count]) => ({ label, count }))
    .sort((a, b) => b.count - a.count)
    .slice(0, limit);

type RationaleChipData = {
  key: string;
  label: string;
  color?: ChipProps["color"];
  tooltip?: string;
};

const buildRationaleChips = (row: ArsenalRow): RationaleChipData[] => {
  const chips: RationaleChipData[] = [];
  const pushChip = (
    id: string,
    label?: string | null,
    color?: ChipProps["color"],
    tooltip?: string
  ) => {
    if (!label) return;
    const key = `${id}-${chips.length}-${label}`;
    chips.push({ key, label, color, tooltip });
  };

  const rationale = row.rationale;
  rationale?.segment?.asset_matches?.forEach((match, idx) =>
    pushChip(`asset-${idx}`, `Asset fit: ${match}`, "info")
  );
  rationale?.segment?.channel_matches?.forEach((match, idx) =>
    pushChip(`channel-${idx}`, `Channel fit: ${match}`, "success")
  );
  rationale?.segment?.account_segments?.forEach((segment, idx) =>
    pushChip(`segment-${idx}`, `Segment: ${segment}`, "default")
  );

  const belief = rationale?.belief;
  if (belief?.pain_match) {
    pushChip("pain", `Pain: ${belief.pain_match}`, "warning");
  }
  if (belief?.channel_concern_match) {
    pushChip("channel-pain", `Channel pain: ${belief.channel_concern_match}`, "warning");
  }
  belief?.belief_tokens?.slice(0, 3).forEach((token, idx) =>
    pushChip(`belief-${idx}`, `Belief: ${token}`, "secondary")
  );
  belief?.stage_tokens?.slice(0, 2).forEach((token, idx) =>
    pushChip(`stage-${idx}`, `Stage cue: ${token}`, "secondary")
  );

  const reasons = row.reasons || {};
  if (reasons.stage_alignment) pushChip("reason-stage", "Stage aligned", "success");
  if (reasons.persona_alignment) pushChip("reason-persona", "Persona fit", "primary");
  if (reasons.concern_alignment) pushChip("reason-pain", "Pain fit", "warning");
  if (reasons.channel_persona_alignment)
    pushChip("reason-channel-persona", "Channel persona fit", "primary");

  const segmentFit = (reasons.segment_fit as string[] | undefined) ?? [];
  segmentFit.forEach((match, idx) =>
    pushChip(`reason-seg-${idx}`, `Segment fit: ${match}`, "info")
  );
  const channelSegment = (reasons.channel_segment_fit as string[] | undefined) ?? [];
  channelSegment.forEach((match, idx) =>
    pushChip(`reason-channel-seg-${idx}`, `Channel segment: ${match}`, "info")
  );

  return chips;
};

const renderRationaleCell = (row: ArsenalRow) => {
  const chips = buildRationaleChips(row);
  if (!chips.length) {
    return (
      <Typography variant="body2" color="text.secondary">
        —
      </Typography>
    );
  }
  return (
    <Stack direction="row" spacing={0.5} sx={{ flexWrap: "wrap" }}>
      {chips.map((chip) => {
        const chipNode = (
          <Chip
            key={chip.key}
            size="small"
            label={chip.label}
            color={chip.color}
            variant={chip.color ? "filled" : "outlined"}
          />
        );
        if (chip.tooltip) {
          return (
            <Tooltip title={chip.tooltip} key={chip.key}>
              {chipNode}
            </Tooltip>
          );
        }
        return chipNode;
      })}
    </Stack>
  );
};

const ExecutionPlanMatrixView = ({ matrix }: { matrix: ExecutionMatrix | null }) => {
  if (!matrix || !matrix.quarters.length || !matrix.personas.length) {
    return (
      <Card variant="outlined">
        <CardContent>
          <Typography variant="h6" sx={{ mb: 1 }}>
            Execution Plan
          </Typography>
          <Typography variant="body2" color="text.secondary">
            Execution plan will appear once campaigns assign personas and timelines.
          </Typography>
        </CardContent>
      </Card>
    );
  }
  const columnTemplate = `160px repeat(${matrix.quarters.length}, minmax(200px, 1fr))`;
  return (
    <Card variant="outlined">
      <CardContent>
        <Typography variant="h6" sx={{ mb: 1 }}>
          Execution Plan (Personas vs Time)
        </Typography>
        <Typography variant="body2" color="text.secondary">
          Personas run down the rows, quarters move left to right. Color intensity reflects cadence pressure; hashes indicate fatigue.
        </Typography>
        <Box sx={{ overflowX: "auto", mt: 2 }}>
          <Box
            sx={{
              display: "grid",
              gridTemplateColumns: columnTemplate,
              gap: 1,
              minWidth: matrix.quarters.length * 200 + 200,
            }}
          >
            <Box />
            {matrix.quarters.map((quarter) => (
              <Paper
                key={`quarter-${quarter}`}
                variant="outlined"
                sx={{ p: 1.25, backgroundColor: "grey.50" }}
              >
                <Typography variant="subtitle2">{quarter}</Typography>
              </Paper>
            ))}
            {matrix.personas.map((persona) => (
              <React.Fragment key={`persona-row-${persona}`}>
                <Paper variant="outlined" sx={{ p: 1.25, backgroundColor: "grey.50" }}>
                  <Typography variant="subtitle2">{persona}</Typography>
                </Paper>
                {matrix.quarters.map((quarter) => {
                  const key = `${quarter}|${persona}`;
                  const blocks = matrix.cells[key] || [];
                  if (!blocks.length) {
                    return (
                      <Paper
                        key={key}
                        variant="outlined"
                        sx={{ p: 1.25, minHeight: 96, backgroundColor: "grey.50" }}
                      >
                        <Typography variant="caption" color="text.secondary">
                          — idle —
                        </Typography>
                      </Paper>
                    );
                  }
                  return (
                    <Stack key={key} spacing={0.75} sx={{ minHeight: 96 }}>
                      {blocks.map((block, idx) => {
                        const color = themeColorFor(block.theme);
                        const fatigueValue =
                          typeof block.fatigue === "number" && !Number.isNaN(block.fatigue)
                            ? block.fatigue
                            : null;
                        const beliefValue =
                          typeof block.belief === "number" && !Number.isNaN(block.belief)
                            ? block.belief
                            : null;
                        const hashed = fatigueValue !== null && fatigueValue > 0.6;
                        const glow = beliefValue !== null && beliefValue >= 0.6;
                        return (
                          <Paper
                            key={`${key}-${idx}`}
                            variant="outlined"
                            sx={{
                              p: 1,
                              backgroundColor: `${color}22`,
                              borderColor: glow ? color : "divider",
                              borderWidth: glow ? 2 : 1,
                              backgroundImage: hashed
                                ? "repeating-linear-gradient(-45deg, rgba(0,0,0,0.08), rgba(0,0,0,0.08) 8px, transparent 8px, transparent 16px)"
                                : undefined,
                            }}
                          >
                            <Typography variant="body2" sx={{ fontWeight: 600 }}>
                              {block.theme}
                            </Typography>
                            <Typography variant="caption" color="text.secondary">
                              {block.mode === "focused" ? "Focused" : "Broad"} · Δ {fmtBasisPoints(block.deltaBp)}
                            </Typography>
                          <Typography
                            variant="caption"
                            color="text.secondary"
                            sx={{ display: "block" }}
                          >
                            Accounts: {block.accountCount} · Assets: {block.assets ?? 0}
                          </Typography>
                          {fatigueValue !== null && (
                            <Typography variant="caption" color="text.secondary">
                              Fatigue {Math.round(fatigueValue * 100)}%
                            </Typography>
                          )}
                          {beliefValue !== null && (
                            <Typography variant="caption" color="text.secondary">
                              Belief {fmtPercent(beliefValue)}
                            </Typography>
                          )}
                          {block.campaigns?.length ? (
                            <Box sx={{ mt: 0.5 }}>
                              {block.campaigns.slice(0, 2).map((campaign) => (
                                <Typography
                                  key={`${key}-${campaign.id}`}
                                  variant="caption"
                                  color="text.secondary"
                                  sx={{ display: "block" }}
                                >
                                  • {campaign.theme}: Δ {fmtBasisPoints(campaign.deltaBp)}
                                </Typography>
                              ))}
                              {block.campaigns.length > 2 && (
                                <Typography variant="caption" color="text.secondary">
                                  +{block.campaigns.length - 2} more plays
                                </Typography>
                              )}
                            </Box>
                          ) : null}
                        </Paper>
                      );
                    })}
                  </Stack>
                );
                })}
              </React.Fragment>
            ))}
          </Box>
        </Box>
      </CardContent>
    </Card>
  );
};

function focusMatchKey(
  descriptor?: string | null | undefined,
  stageLabel?: string | null | undefined
): string | null {
  if (!descriptor) return null;
  const normalizedDescriptor = descriptor.trim().toLowerCase();
  const normalizedStage = (stageLabel || "").trim().toLowerCase();
  return `${normalizedDescriptor}::${normalizedStage}`;
}

const renderAssetTable = (rows: AggregatedFocusAssetRow[], keyPrefix: string) => {
  if (!rows || rows.length === 0) {
    return (
      <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
        No asset recommendations yet.
      </Typography>
    );
  }

  return (
    <TableContainer component={Paper} variant="outlined" sx={{ mt: 1.5 }}>
      <Table size="small">
        <TableHead>
          <TableRow>
            <TableCell>Asset</TableCell>
            <TableCell>Channel</TableCell>
            <TableCell align="right">Δ (bps)</TableCell>
            <TableCell align="right">Confidence</TableCell>
            <TableCell align="right">Belief Conversion</TableCell>
            <TableCell align="right">Avg Time</TableCell>
          </TableRow>
        </TableHead>
        <TableBody>
          {rows.map((row, idx) => {
            const assetName =
              row.asset?.name ||
              row.asset?.title ||
              row.asset_type_label ||
              row.asset_type ||
              "Asset";
            const channelName =
              row.channel?.name ||
              row.channel_type_label ||
              row.channel_type ||
              "Channel";
            const avgDuration = row.avg_duration_days ?? row.channel?.avg_duration_days ?? 14;

            return (
              <TableRow key={`${keyPrefix}-asset-${idx}`}>
                <TableCell>
                  <Typography variant="body2">{assetName}</Typography>
                  {row.asset_type_label && (
                    <Typography variant="caption" color="text.secondary">
                      {row.asset_type_label}
                    </Typography>
                  )}
                </TableCell>
                <TableCell>
                  <Typography variant="body2">{channelName}</Typography>
                  {row.channel?.mode && (
                    <Typography variant="caption" color="text.secondary">
                      Mode: {row.channel.mode}
                    </Typography>
                  )}
                </TableCell>
                <TableCell align="right">
                  {fmtBasisPoints(row.expected_delta_bp)}
                </TableCell>
                <TableCell align="right">
                  {fmtPercent(row.confidence ?? null)}
                </TableCell>
                <TableCell align="right">
                  {fmtPercent(row.belief_conversion_likelihood ?? null)}
                </TableCell>
                <TableCell align="right">{toWeeks(avgDuration)}</TableCell>
              </TableRow>
            );
          })}
        </TableBody>
      </Table>
    </TableContainer>
  );
};

const renderFocusAssetTable = (focus: PortfolioConversionFocus) =>
  renderAssetTable(focus.asset_table || [], focus.key);

const describeBeliefTransitionMeta = (
  meta: ConversionSequenceEntry["belief_transition_meta"] | undefined
) => {
  if (!meta) return "Belief shift";
  const focus =
    meta.pain ||
    meta.resolution ||
    meta.problem ||
    (meta.stage_label ? meta.stage_label : undefined);
  if (meta.stage_label && focus) {
    return meta.stage_label === focus ? meta.stage_label : `${meta.stage_label}: ${focus}`;
  }
  return focus || "Belief shift";
};

const renderMatchedPeopleChips = (
  matches: PersonMatch[] | undefined,
  keyPrefix: string
) => {
  if (!matches?.length) return null;
  const sortedMatches = [...matches].sort((a, b) => {
    const scoreA =
      a.committee_probability ?? a.person_involvement_score ?? a.person_belief_level ?? 0;
    const scoreB =
      b.committee_probability ?? b.person_involvement_score ?? b.person_belief_level ?? 0;
    return scoreB - scoreA;
  });
  return (
    <Stack direction="row" spacing={0.5} sx={{ flexWrap: "wrap", mt: 0.75 }}>
      {sortedMatches.map((match) => (
        <Tooltip key={`${keyPrefix}-match-${match.match_id}`} title={personMatchTooltip(match)}>
          <Chip label={personMatchLabel(match)} size="small" variant="outlined" />
        </Tooltip>
      ))}
    </Stack>
  );
};

const renderConversionSequenceEntry = (entry: ConversionSequenceEntry) => {
  const key = `sequence-${entry.timeline_index}`;
  const personaLabel =
    entry.persona_descriptor || entry.persona?.label || "Target persona";
  const stageDescription = describeBeliefTransitionMeta(entry.belief_transition_meta);
  const segmentEntries = segmentChipsFromFilters(entry.segment_filters);
  const tooltipLines: string[] = [];
  if (entry.expected_outcome_summary) {
    tooltipLines.push(entry.expected_outcome_summary);
  }
  if (stageDescription) {
    tooltipLines.push(stageDescription);
  }
  if (entry.segment_summary) {
    tooltipLines.push(entry.segment_summary);
  }
  segmentEntries.forEach(([key, value]) => {
    tooltipLines.push(`${titleize(key)}: ${value}`);
  });
  const tooltipContent =
    tooltipLines.length > 0 ? (
      <Box>
        {tooltipLines.map((line, idx) => (
          <Typography key={`${key}-tooltip-${idx}`} variant="body2" sx={{ display: "block" }}>
            {line}
          </Typography>
        ))}
      </Box>
    ) : null;
  const chips: Array<{ label: string; outlined?: boolean }> = [
    { label: `Δ ${fmtBasisPoints(entry.expected_delta_bp)}` },
    {
      label: `Confidence ${fmtPercent(entry.avg_confidence ?? null)}`,
      outlined: true,
    },
  ];
  if (entry.avg_belief_conversion !== null && entry.avg_belief_conversion !== undefined) {
    chips.push({
      label: `Belief ${fmtPercent(entry.avg_belief_conversion)}`,
      outlined: true,
    });
  }
  const effortPct =
    entry.effort_pct ??
    (typeof entry.effort_required === "number"
      ? Math.round(entry.effort_required * 100)
      : undefined);
  if (effortPct !== undefined && !Number.isNaN(effortPct)) {
    chips.push({ label: `Effort ${effortPct}%`, outlined: true });
  }
  if (entry.time_to_effect_days !== undefined && entry.time_to_effect_days !== null) {
    chips.push({
      label: `Time ${toWeeks(entry.time_to_effect_days)}`,
      outlined: true,
    });
  }

  const matchKey =
    focusMatchKey(entry.persona_descriptor, entry.belief_transition_meta?.stage_label) ||
    focusMatchKey(entry.persona?.label, entry.belief_transition_meta?.stage_label);
  const portfolioMatch = matchKey ? portfolioFocusContext.get(matchKey) : undefined;
  const portfolioThemeLabel = portfolioMatch?.theme?.theme;
  const portfolioThemeQuarter = portfolioMatch?.theme?.quarter;
  const portfolioAggregatedFocus = portfolioMatch?.focus;

  return (
    <Box
      key={key}
      sx={{
        border: "1px solid",
        borderColor: "divider",
        borderRadius: 1,
        p: 2,
      }}
    >
      <Stack
        direction={{ xs: "column", md: "row" }}
        spacing={1}
        alignItems={{ xs: "flex-start", md: "center" }}
      >
        <Chip
          size="small"
          color="primary"
          label={entry.timeline_label || `T+${entry.timeline_index}`}
        />
        <Stack direction="row" spacing={1} alignItems="center">
          <Typography variant="subtitle1">{personaLabel}</Typography>
          {tooltipContent && (
            <Tooltip title={tooltipContent} arrow>
              <IconButton size="small">
                <InfoOutlinedIcon fontSize="inherit" />
              </IconButton>
            </Tooltip>
          )}
        </Stack>
      </Stack>
      <Typography variant="body2" color="text.secondary" sx={{ mt: 0.5 }}>
        {stageDescription}
      </Typography>
      {entry.expected_outcome_summary && (
        <Typography variant="body2" sx={{ mt: 1 }}>
          {entry.expected_outcome_summary}
        </Typography>
      )}
      {chips.length ? (
        <Stack direction="row" spacing={1} sx={{ flexWrap: "wrap", mt: 1 }}>
          {chips.map((chip, idx) => (
            <Chip
              key={`${key}-chip-${idx}`}
              size="small"
              label={chip.label}
              variant={chip.outlined ? "outlined" : "filled"}
            />
          ))}
        </Stack>
      ) : null}
      {portfolioThemeLabel && (
        <Typography variant="caption" color="text.secondary" sx={{ display: "block", mt: 0.75 }}>
          Campaign Theme: {portfolioThemeLabel}
          {portfolioThemeQuarter ? ` (${portfolioThemeQuarter})` : ""}
        </Typography>
      )}
      {portfolioAggregatedFocus ? (
        <Stack direction="row" spacing={0.75} sx={{ flexWrap: "wrap", mt: 0.5 }}>
          <Chip
            size="small"
            variant="outlined"
            label={`Portfolio Δ ${fmtBasisPoints(portfolioAggregatedFocus.expected_delta_bp)}`}
          />
          <Chip
            size="small"
            variant="outlined"
            label={`${portfolioAggregatedFocus.account_count} account${
              portfolioAggregatedFocus.account_count === 1 ? "" : "s"
            }`}
          />
        </Stack>
      ) : null}
      {renderSegmentSummary(entry.segment_filters, entry.segment_summary)}
      {entry.people?.length ? (
        <Stack direction="row" spacing={0.5} sx={{ flexWrap: "wrap", mt: 0.75 }}>
          {entry.people.map((person) => (
            <Chip
              key={`${key}-person-${person}`}
              label={person}
              size="small"
              variant="outlined"
            />
          ))}
        </Stack>
      ) : null}
      {renderMatchedPeopleChips(entry.matched_people, key)}
      {renderAssetTable(entry.asset_table || [], key)}
    </Box>
  );
};

const renderConversionFocusDetails = (focus: ConversionFocus, key: string) => {
  const personaLabel =
    focus.persona_descriptor ||
    focus.persona_meta?.label ||
    focus.persona_label ||
    "Focus Playbook";
  const chips: Array<{ label: string; outlined?: boolean }> = [
    { label: `Δ ${fmtBasisPoints(focus.expected_delta_bp)}` },
    {
      label: `Confidence ${fmtPercent(focus.confidence ?? null)}`,
      outlined: true,
    },
  ];
  if (focus.avg_belief_conversion !== undefined && focus.avg_belief_conversion !== null) {
    chips.push({
      label: `Belief ${fmtPercent(focus.avg_belief_conversion)}`,
      outlined: true,
    });
  }
  const effortPct =
    focus.avg_effort_pct ??
    (typeof focus.avg_effort_required === "number"
      ? Math.round(focus.avg_effort_required * 100)
      : undefined);
  if (effortPct !== undefined && !Number.isNaN(effortPct)) {
    chips.push({ label: `Effort ${effortPct}%`, outlined: true });
  }
  if (focus.time_to_effect_days !== undefined && focus.time_to_effect_days !== null) {
    chips.push({
      label: `Time ${toWeeks(focus.time_to_effect_days)}`,
      outlined: true,
    });
  }

  const matchKey =
    focusMatchKey(
      focus.persona_descriptor ||
        focus.persona_meta?.label ||
        focus.persona_label ||
        focus.persona_focus,
      focus.stage_label
    ) || null;
  const portfolioMatch = matchKey ? portfolioFocusContext.get(matchKey) : undefined;
  const portfolioThemeLabel = portfolioMatch?.theme?.theme;
  const portfolioThemeQuarter = portfolioMatch?.theme?.quarter;
  const portfolioAggregatedFocus = portfolioMatch?.focus;

  return (
    <Box
      key={key}
      sx={{
        border: "1px solid",
        borderColor: "divider",
        borderRadius: 1,
        p: 1.5,
      }}
    >
      <Typography variant="subtitle2">{personaLabel}</Typography>
      <Typography variant="body2" color="text.secondary">
        {focus.stage_label || "Belief shift"}
      </Typography>
      {focus.expected_outcome_summary && (
        <Typography variant="body2" sx={{ mt: 0.75 }}>
          {focus.expected_outcome_summary}
        </Typography>
      )}
      {chips.length ? (
        <Stack direction="row" spacing={0.75} sx={{ flexWrap: "wrap", mt: 0.75 }}>
          {chips.map((chip, idx) => (
            <Chip
              key={`${key}-focus-chip-${idx}`}
              size="small"
              label={chip.label}
              variant={chip.outlined ? "outlined" : "filled"}
            />
          ))}
        </Stack>
      ) : null}
      {portfolioThemeLabel && (
        <Typography variant="caption" color="text.secondary" sx={{ display: "block", mt: 0.75 }}>
          Portfolio Theme: {portfolioThemeLabel}
          {portfolioThemeQuarter ? ` (${portfolioThemeQuarter})` : ""}
        </Typography>
      )}
      {portfolioAggregatedFocus ? (
        <Stack direction="row" spacing={0.75} sx={{ flexWrap: "wrap", mt: 0.5 }}>
          <Chip
            size="small"
            variant="outlined"
            label={`Portfolio Δ ${fmtBasisPoints(portfolioAggregatedFocus.expected_delta_bp)}`}
          />
          <Chip
            size="small"
            variant="outlined"
            label={`${portfolioAggregatedFocus.account_count} account${
              portfolioAggregatedFocus.account_count === 1 ? "" : "s"
            }`}
          />
        </Stack>
      ) : null}
      {renderSegmentSummary(focus.segment_filters, focus.segment_summary)}
      {focus.people?.length ? (
        <Stack direction="row" spacing={0.5} sx={{ flexWrap: "wrap", mt: 0.75 }}>
          {focus.people.map((person) => (
            <Chip
              key={`${key}-focus-person-${person}`}
              label={person}
              size="small"
              variant="outlined"
            />
          ))}
        </Stack>
      ) : null}
      {renderAssetTable(focus.asset_table || [], key)}
    </Box>
  );
};

const renderCampaignCard = (campaign: Campaign) => {
  const key = `${campaign.quarter}-${campaign.theme}`;
  const primaryFocus = campaign.conversion_focuses?.[0];
  const campaignMatchKey =
    primaryFocus &&
    (focusMatchKey(
      primaryFocus.persona_descriptor ||
        primaryFocus.persona_meta?.label ||
        primaryFocus.persona_label ||
        primaryFocus.persona_focus,
      primaryFocus.stage_label
    ) ||
      null);
  const portfolioMatch = campaignMatchKey ? portfolioFocusContext.get(campaignMatchKey) : undefined;
  return (
    <Box
      key={key}
      sx={{
        border: "1px solid",
        borderColor: "divider",
        borderRadius: 1,
        p: 2,
      }}
    >
      <Stack
        direction={{ xs: "column", md: "row" }}
        spacing={1}
        justifyContent="space-between"
        alignItems={{ xs: "flex-start", md: "center" }}
      >
        <Box>
          <Typography variant="subtitle1">{campaign.theme}</Typography>
          <Typography variant="body2" color="text.secondary">
            Quarter {campaign.quarter}
          </Typography>
          {campaign.expected_outcome_summary && (
            <Typography variant="body2" sx={{ mt: 0.75 }}>
              {campaign.expected_outcome_summary}
            </Typography>
          )}
          {renderSegmentSummary(campaign.segment_filters, campaign.segment_summary)}
          {portfolioMatch?.theme && (
            <Typography variant="caption" color="text.secondary" sx={{ display: "block", mt: 0.5 }}>
              Portfolio Theme: {portfolioMatch.theme.theme}
              {portfolioMatch.theme.quarter ? ` (${portfolioMatch.theme.quarter})` : ""}
            </Typography>
          )}
          {campaign.persona_descriptor_counts?.length ? (
            <Stack direction="row" spacing={0.75} sx={{ flexWrap: "wrap", mt: 0.5 }}>
              {campaign.persona_descriptor_counts.map((item) => (
                <Chip
                  key={`${key}-descriptor-${item.descriptor}`}
                  label={`${item.descriptor} (${item.count})`}
                  size="small"
                  variant="outlined"
                />
              ))}
            </Stack>
          ) : null}
        </Box>
        <Stack direction="row" spacing={1} sx={{ flexWrap: "wrap" }}>
          <Chip size="small" label={`Δ ${fmtBasisPoints(campaign.total_delta_bp)}`} />
          <Chip
            size="small"
            variant="outlined"
            label={`Confidence ${fmtPercent(campaign.confidence)}`}
          />
          <Chip size="small" variant="outlined" label={`Plays ${campaign.play_count}`} />
          {portfolioMatch?.focus ? (
            <Chip
              size="small"
              variant="outlined"
              label={`Portfolio Δ ${fmtBasisPoints(portfolioMatch.focus.expected_delta_bp)}`}
            />
          ) : null}
        </Stack>
      </Stack>
      {campaign.conversion_focuses && campaign.conversion_focuses.length > 0 ? (
        <Stack spacing={1.5} sx={{ mt: 2 }}>
          {campaign.conversion_focuses.map((focus, idx) =>
            renderConversionFocusDetails(focus, `${key}-focus-${idx}`)
          )}
        </Stack>
      ) : null}
    </Box>
  );
};

  const renderAccountsPopover = () => {
    if (!accountPopover.anchorEl || !accountPopover.focusId) return null;
    const focus = portfolioFocusMap.get(accountPopover.focusId);
    const accounts = focus?.accounts ?? [];
    return (
      <Popover
        open
        anchorEl={accountPopover.anchorEl}
        onClose={() => setAccountPopover({ anchorEl: null, focusId: null })}
        anchorOrigin={{ vertical: "bottom", horizontal: "left" }}
      >
        <Box sx={{ p: 2, maxWidth: 260 }}>
          <Typography variant="subtitle2">Target Accounts</Typography>
          {accounts.length ? (
            <Stack spacing={0.5} sx={{ mt: 1 }}>
              {accounts.map((acct) => (
                <Typography key={`${focus?.key || "focus"}-acct-${acct.id}`} variant="body2">
                  {acct.name}
                </Typography>
              ))}
            </Stack>
          ) : (
            <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
              No accounts mapped yet.
            </Typography>
          )}
        </Box>
      </Popover>
    );
  };

const renderExecutionContent = () => {
  const currentQuarter = executionMatrixForView?.quarters?.[0] ?? "Q1";
  const topPersonasThisQuarter =
    executionMatrixForView?.personas.slice(0, 3) ?? [];

  return (
    <Stack spacing={2}>
      <Card variant="outlined">
        <CardContent>
          <Typography variant="h6" sx={{ mb: 1 }}>
            This Quarter Focus ({currentQuarter})
          </Typography>
          <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
            Rallying plays and personas prioritized for the current quarter.
          </Typography>
          <Stack direction={{ xs: "column", md: "row" }} spacing={2}>
            <SummaryMetric
              label="Headline"
              value={
                topPersonasThisQuarter.length
                  ? topPersonasThisQuarter.join(", ")
                  : "Pending updates"
              }
            />
            <SummaryMetric
              label="Plays scheduled"
              value={
                executionMatrixForView?.personas?.length
                  ? `${executionMatrixForView.personas.length * (executionMatrixForView.quarters?.length || 1)}+`
                  : "—"
              }
            />
            <SummaryMetric
              label="Next action"
              value="Generate Q briefing"
            />
          </Stack>
          <Stack direction="row" spacing={1} sx={{ mt: 2, flexWrap: "wrap" }}>
            <Button variant="contained" size="small">
              Download Q brief
            </Button>
            <Button variant="outlined" size="small">
              Share with Sales
            </Button>
          </Stack>
        </CardContent>
      </Card>
      <ExecutionPlanMatrixView matrix={executionMatrixForView} />
    </Stack>
  );
};

const renderOverviewContent = () => {
  if (!plan) return null;
  if (!selectedAccountPlan) {
    const totalAccounts = portfolioSummaryReport?.totalAccounts ?? accounts.length;
    const personaTotal =
      (portfolioSummaryReport?.personaMix || []).reduce(
        (sum, row) => sum + (row.count ?? 0),
        0
      ) || plan.summary?.total_persona_requirements || 0;
    const missingContacts =
      (plan.summary?.total_persona_requirements || 0) -
      (plan.summary?.avg_people_coverage ?? 0) *
        (plan.summary?.total_persona_requirements || 0);
    const timeframeWeeks = portfolioSummaryReport?.expectedTimeToConversionDays
      ? Math.round(
          (portfolioSummaryReport.expectedTimeToConversionDays || 180) / 7
        )
      : 26;
    const expectedLiftPts =
      (portfolioSummaryReport?.totalDeltaBp || 0) / 100;
    const accountClusters = portfolioSummaryReport?.accountClusters ?? [];
    const clusterLiftShare = accountClusters.reduce(
      (sum, cluster) => sum + (cluster.shareOfExpectedLift || 0),
      0
    );
    const thesisClusters =
      thesisData?.clusterBriefs?.length ? thesisData.clusterBriefs : accountClusters;
    const heroSentence = `Increase win likelihood for ${totalAccounts} target account${
      totalAccounts === 1 ? "" : "s"
    } by +${expectedLiftPts.toFixed(1)} pts over ${timeframeWeeks} weeks.`;
    const stageDistribution = portfolioSummaryReport?.stageDistribution ?? [];
    const dominantStageEntry =
      stageDistribution.length > 0
        ? stageDistribution.reduce(
            (top, row) => (row.count > (top?.count ?? 0) ? row : top),
            stageDistribution[0] ?? null
          )
        : null;
    const dominantStageLabel = dominantStageEntry?.stage
      ? normalizeStageLabel(dominantStageEntry.stage)
      : null;
    const conversionFocuses = plan?.portfolio_plan?.conversion_focuses ?? [];
    const targetFocus = conversionFocuses.find((focus) => focus.stage_label);
    const targetStageLabel = targetFocus?.stage_label
      ? normalizeStageLabel(targetFocus.stage_label)
      : null;
    const personaNames = (portfolioSummaryReport?.personaMix || [])
      .map((item) => item.label?.split("|")[0]?.trim() || item.label || null)
      .filter((label): label is string => Boolean(label))
      .slice(0, 2);
    const heroNarrative = (() => {
      if (!personaNames.length && !targetStageLabel && !plan.summary) {
        return "Portfolio strategy will refresh once telemetry syncs.";
      }
      const personaPhrase = personaNames.length ? formatList(personaNames) : "priority personas";
      let narrative = `We’ll move ${personaPhrase}`;
      if (dominantStageLabel && targetStageLabel) {
        narrative += ` from ${dominantStageLabel} to ${targetStageLabel}`;
      } else if (targetStageLabel) {
        narrative += ` toward ${targetStageLabel}`;
      }
      const proofFocus =
        targetFocus?.expected_outcome_summary ||
        targetFocus?.campaign_theme ||
        plan.summary?.top_pains?.[0]?.label ||
        plan.summary?.top_capabilities?.[0]?.label ||
        null;
      if (proofFocus) {
        narrative += ` by proving ${proofFocus}`;
      }
      if (!narrative.endsWith(".")) {
        narrative = `${narrative}.`;
      }
      return narrative;
    })();
    const personaSpotlights =
      conversionFocuses.length > 0
        ? conversionFocuses.slice(0, 3).map((focus) => ({
            persona:
              focus.persona_descriptor ||
              focus.persona_focus ||
              focus.label ||
              "Target persona",
            stage: focus.stage_label || focus.timeline_label || "Next stage",
            summary:
              focus.expected_outcome_summary ||
              focus.segment_summary ||
              focus.campaign_theme ||
              null,
            delta: typeof focus.expected_delta_bp === "number" ? focus.expected_delta_bp : null,
          }))
        : (portfolioSummaryReport?.personaMix || []).slice(0, 3).map((item, idx) => ({
            persona: item.label || `Persona ${idx + 1}`,
            stage: dominantStageLabel || "Next stage",
            summary:
              typeof item.count === "number"
                ? `${item.count} persona${item.count === 1 ? "" : "s"} in plan`
                : null,
            delta: null,
          }));

    const beliefTransitions =
      (thesisData?.strategyHighlights || []).slice(0, 4);

    const topPlays = aggregatedPortfolioArsenal.slice(0, 3);

    return (
      <Stack spacing={2}>
        <Card
          variant="outlined"
          sx={{
            background: "linear-gradient(135deg, #101935, #1b2a52)",
            color: "common.white",
          }}
        >
          <CardContent>
            <Stack spacing={2}>
              <Typography variant="overline" sx={{ color: "grey.300" }}>
                Portfolio Goal
              </Typography>
              <Typography variant="h5">{heroSentence}</Typography>
              <Typography variant="body2" sx={{ color: "grey.300" }}>
                {heroNarrative}
              </Typography>
              <Stack
                direction={{ xs: "column", md: "row" }}
                spacing={2}
                sx={{ flexWrap: "wrap" }}
              >
                <SummaryMetric label="Accounts" value={totalAccounts} />
                <SummaryMetric
                  label="Personas in play"
                  value={`${personaTotal}`}
                />
                <SummaryMetric
                  label="Timeframe"
                  value={`${timeframeWeeks} weeks`}
                />
                <SummaryMetric
                  label="Expected uplift"
                  value={`+${expectedLiftPts.toFixed(1)} pts`}
                />
              </Stack>
              <Stack direction="row" spacing={1} sx={{ flexWrap: "wrap" }}>
                <Button variant="contained" color="secondary" size="small">
                  Export as deck
                </Button>
                <Button variant="outlined" color="secondary" size="small">
                  Share with Sales
                </Button>
                <Button variant="text" color="inherit" size="small">
                  Assign missing contacts ({Math.max(0, Math.round(missingContacts))})
                </Button>
              </Stack>
            </Stack>
          </CardContent>
        </Card>

        {accountClusters.length ? (
          <Card variant="outlined">
            <CardContent>
              <Typography variant="subtitle2" gutterBottom>
                Portfolio clusters
              </Typography>
              <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                We clustered {totalAccounts} target account{totalAccounts === 1 ? "" : "s"} into{" "}
                {accountClusters.length} groups covering{" "}
                {fmtPercent(clusterLiftShare)} of expected lift.
              </Typography>
              <Stack spacing={1.5}>
                {accountClusters.map((cluster) => {
                  const sample = cluster.accounts.slice(0, 3).map((acct) => acct.name);
                  const remainder = Math.max(cluster.accountCount - sample.length, 0);
                  return (
                    <Paper key={cluster.token} variant="outlined" sx={{ p: 1.25 }}>
                      <Stack
                        direction={{ xs: "column", sm: "row" }}
                        spacing={1}
                        justifyContent="space-between"
                        alignItems={{ xs: "flex-start", sm: "center" }}
                      >
                        <Typography variant="subtitle1">{cluster.label}</Typography>
                        <Stack direction="row" spacing={1}>
                          <Chip
                            size="small"
                            label={`${cluster.accountCount} account${
                              cluster.accountCount === 1 ? "" : "s"
                            }`}
                          />
                          <Chip
                            size="small"
                            color="primary"
                            variant="outlined"
                            label={`${fmtPercent(cluster.shareOfExpectedLift)} lift`}
                          />
                        </Stack>
                      </Stack>
                      <Typography variant="body2" color="text.secondary" sx={{ mt: 0.5 }}>
                        {sample.length
                          ? `${sample.join(", ")}${remainder > 0 ? ` +${remainder} more` : ""}`
                          : "Accounts will populate after enrichment syncs."}
                      </Typography>
                    </Paper>
                  );
                })}
              </Stack>
            </CardContent>
          </Card>
        ) : null}

        {thesisClusters.length ? (
          <Card variant="outlined">
            <CardContent>
              <Typography variant="subtitle2" gutterBottom>
                Cluster theses — personas, belief paths, and plays
              </Typography>
              <Stack spacing={1.5}>
                {thesisClusters.map((cluster) => {
                  const lift = cluster.liftAnalysis;
                  const personaRows = cluster.personaExpectations ?? [];
                  const coalitions = cluster.beliefCoalitions ?? [];
                  const transitions = cluster.beliefTransitions ?? [];
                  const primaryPlays = cluster.primaryPlays ?? [];
                  const fallbackPlan = cluster.fallbackPlan ?? [];
                  return (
                    <Paper key={`cluster-story-${cluster.token}`} variant="outlined" sx={{ p: 1.5 }}>
                      <Stack spacing={1.25}>
                        <Stack
                          direction={{ xs: "column", md: "row" }}
                          spacing={1}
                          justifyContent="space-between"
                          alignItems={{ xs: "flex-start", md: "center" }}
                        >
                          <Box>
                            <Typography variant="subtitle1">{cluster.label}</Typography>
                            <Typography variant="body2" color="text.secondary">
                              {cluster.accountCount} account{cluster.accountCount === 1 ? "" : "s"} ·{" "}
                              {fmtPercent(cluster.shareOfExpectedLift)} of expected lift
                            </Typography>
                          </Box>
                          {lift ? (
                            <Stack direction="row" spacing={1} sx={{ flexWrap: "wrap" }}>
                              <Chip
                                size="small"
                                color="primary"
                                label={`Cluster Δ ${fmtBasisPoints(lift.clusterDeltaBp)}`}
                              />
                              <Chip
                                size="small"
                                label={`Vs generic ${fmtBasisPoints(lift.incrementalVsGenericBp)}`}
                              />
                              {lift.perAccountDeltaBp !== null && lift.perAccountDeltaBp !== undefined && (
                                <Chip
                                  size="small"
                                  variant="outlined"
                                  label={`Per acct ${fmtBasisPoints(lift.perAccountDeltaBp)}`}
                                />
                              )}
                            </Stack>
                          ) : null}
                        </Stack>

                        <Box>
                          <Typography variant="subtitle2">Expected personas & mapped people</Typography>
                          {personaRows.length ? (
                            <TableContainer component={Paper} variant="outlined" sx={{ mt: 1 }}>
                              <Table size="small">
                                <TableHead>
                                  <TableRow>
                                    <TableCell>Persona</TableCell>
                                    <TableCell>Stage</TableCell>
                                    <TableCell align="right">Match rate</TableCell>
                                    <TableCell>Sample people</TableCell>
                                  </TableRow>
                                </TableHead>
                                <TableBody>
                                  {personaRows.map((row) => (
                                    <TableRow key={`${cluster.token}-${row.persona}`}>
                                      <TableCell>{row.persona}</TableCell>
                                      <TableCell>{row.stage || "—"}</TableCell>
                                      <TableCell align="right">
                                        {row.matchRate !== null && row.matchRate !== undefined
                                          ? fmtPercent(row.matchRate)
                                          : "—"}
                                      </TableCell>
                                      <TableCell>
                                        {row.samplePeople.length ? (
                                          <Stack spacing={0.25}>
                                            {row.samplePeople.map((person, idx) => (
                                              <Typography key={`${row.persona}-${idx}`} variant="caption">
                                                {person.name}
                                                {person.title ? ` (${person.title})` : ""}{" "}
                                                {person.accountName ? `· ${person.accountName}` : ""}
                                              </Typography>
                                            ))}
                                          </Stack>
                                        ) : (
                                          <Typography variant="caption" color="text.secondary">
                                            Awaiting matched contacts
                                          </Typography>
                                        )}
                                      </TableCell>
                                    </TableRow>
                                  ))}
                                </TableBody>
                              </Table>
                            </TableContainer>
                          ) : (
                            <Typography variant="body2" color="text.secondary" sx={{ mt: 0.5 }}>
                              Persona expectations will appear once journey data refreshes.
                            </Typography>
                          )}
                        </Box>

                        {coalitions.length ? (
                          <Box>
                            <Typography variant="subtitle2">Belief coalitions</Typography>
                            <Stack direction="row" spacing={1} sx={{ flexWrap: "wrap", mt: 0.5 }}>
                              {coalitions.map((coalition, idx) => (
                                <Chip
                                  key={`${cluster.token}-coalition-${idx}`}
                                  label={`${(coalition.sequence || []).join(" → ")} · ${fmtPercent(
                                    coalition.share
                                  )}`}
                                  variant="outlined"
                                />
                              ))}
                            </Stack>
                          </Box>
                        ) : null}

                        {transitions.length ? (
                          <Box>
                            <Typography variant="subtitle2">Belief transitions</Typography>
                            <Stack spacing={0.5} sx={{ mt: 0.5 }}>
                              {transitions.slice(0, 3).map((transition, idx) => (
                                <Typography key={`${cluster.token}-transition-${idx}`} variant="body2">
                                  <strong>{transition.persona || "Persona"}</strong>{" "}
                                  {transition.stage ? `(${transition.stage})` : ""}:{" "}
                                  {transition.narrative || "Belief progression"} ·{" "}
                                  {fmtBasisPoints(transition.deltaBp)} lift
                                </Typography>
                              ))}
                            </Stack>
                          </Box>
                        ) : null}

                        {primaryPlays.length ? (
                          <Box>
                            <Typography variant="subtitle2">Primary plays</Typography>
                            <TableContainer component={Paper} variant="outlined" sx={{ mt: 0.5 }}>
                              <Table size="small">
                                <TableHead>
                                  <TableRow>
                                    <TableCell>Asset</TableCell>
                                    <TableCell>Channel</TableCell>
                                    <TableCell>Personas</TableCell>
                                    <TableCell align="right">Δ belief</TableCell>
                                  </TableRow>
                                </TableHead>
                                <TableBody>
                                  {primaryPlays.slice(0, 4).map((play, idx) => (
                                    <TableRow key={`${cluster.token}-play-${idx}`}>
                                      <TableCell>
                                        <Typography variant="body2">{play.asset || "Asset"}</Typography>
                                        {play.assetType && (
                                          <Typography variant="caption" color="text.secondary">
                                            {play.assetType}
                                          </Typography>
                                        )}
                                      </TableCell>
                                      <TableCell>
                                        <Typography variant="body2">{play.channel || "Channel"}</Typography>
                                        {play.channelType && (
                                          <Typography variant="caption" color="text.secondary">
                                            {play.channelType}
                                          </Typography>
                                        )}
                                      </TableCell>
                                      <TableCell>
                                        <Typography variant="body2" color="text.secondary">
                                          {(play.personaLabels || []).slice(0, 2).join(", ") ||
                                            "Persona focus"}
                                        </Typography>
                                      </TableCell>
                                      <TableCell align="right">{fmtBasisPoints(play.deltaBp)}</TableCell>
                                    </TableRow>
                                  ))}
                                </TableBody>
                              </Table>
                            </TableContainer>
                          </Box>
                        ) : null}

                        <Box>
                          <Typography variant="subtitle2">Fallback moves</Typography>
                          {fallbackPlan.length ? (
                            <Stack spacing={0.5} sx={{ mt: 0.5 }}>
                              {fallbackPlan.slice(0, 4).map((item, idx) => (
                                <Typography key={`${cluster.token}-fallback-${idx}`} variant="body2">
                                  <strong>{item.persona || "Persona"}</strong>{" "}
                                  {item.stage ? `(${item.stage})` : ""}: {item.reason || "Activate backup motion"}
                                  {item.accounts?.length ? (
                                    <Typography component="span" variant="caption" color="text.secondary">
                                      {" "}
                                      · {item.accounts.join(", ")}
                                    </Typography>
                                  ) : null}
                                </Typography>
                              ))}
                            </Stack>
                          ) : (
                            <Typography variant="body2" color="text.secondary" sx={{ mt: 0.5 }}>
                              No backup actions required yet—engagement coverage is solid.
                            </Typography>
                          )}
                        </Box>
                      </Stack>
                    </Paper>
                  );
                })}
              </Stack>
            </CardContent>
          </Card>
        ) : null}

        <Grid container spacing={2}>
          <Grid item xs={12} md={4}>
            <Card variant="outlined">
              <CardContent>
                <Typography variant="subtitle2" gutterBottom>
                  Who we’re betting on
                </Typography>
                <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
                  Primary personas and their belief jumps.
                </Typography>
                <Stack spacing={1}>
                  {personaSpotlights.length ? (
                    personaSpotlights.map((spotlight, idx) => (
                      <Paper
                        key={`persona-chip-${spotlight.persona}-${idx}`}
                        variant="outlined"
                        sx={{ p: 1 }}
                      >
                        <Stack direction="row" spacing={1} alignItems="center">
                          <Chip size="small" label={`Stage: ${spotlight.stage}`} />
                          <Typography variant="body2">{spotlight.persona}</Typography>
                        </Stack>
                        {spotlight.summary && (
                          <Typography variant="caption" color="text.secondary">
                            {spotlight.summary}
                          </Typography>
                        )}
                        {spotlight.delta !== null && (
                          <Typography variant="caption" color="text.secondary">
                            Δ {fmtBasisPoints(spotlight.delta)}
                          </Typography>
                        )}
                      </Paper>
                    ))
                  ) : (
                    <Typography variant="body2" color="text.secondary">
                      Personas will populate after planner refresh.
                    </Typography>
                  )}
                </Stack>
              </CardContent>
            </Card>
          </Grid>
          <Grid item xs={12} md={4}>
            <Card variant="outlined">
              <CardContent>
                <Typography variant="subtitle2" gutterBottom>
                  What we need them to believe
                </Typography>
                <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
                  Belief transitions with the biggest lift.
                </Typography>
                <Stack spacing={1}>
                  {beliefTransitions.length ? (
                    beliefTransitions.map((belief, idx) => (
                      <Paper key={`belief-${idx}`} variant="outlined" sx={{ p: 1 }}>
                        <Typography variant="body2">{belief}</Typography>
                        <Typography variant="caption" color="text.secondary">
                          Lift target: +{Math.round(expectedLiftPts / (idx + 2))} bps
                        </Typography>
                      </Paper>
                    ))
                  ) : (
                    <Typography variant="body2" color="text.secondary">
                      Belief thesis is syncing—check back soon.
                    </Typography>
                  )}
                </Stack>
              </CardContent>
            </Card>
          </Grid>
          <Grid item xs={12} md={4}>
            <Card variant="outlined">
              <CardContent>
                <Typography variant="subtitle2" gutterBottom>
                  How we’ll get there
                </Typography>
                <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
                  Plays spanning persona, stage, and channels.
                </Typography>
                <Stack spacing={1}>
                  {topPlays.length ? (
                    topPlays.map((play) => (
                      <Paper key={play.key} variant="outlined" sx={{ p: 1 }}>
                        <Typography variant="body2">{play.assetLabel}</Typography>
                        <Stack direction="row" spacing={0.5} sx={{ flexWrap: "wrap", mt: 0.5 }}>
                          <Chip
                            size="small"
                            label={play.personas[0] || "Persona"}
                            variant="outlined"
                          />
                          <Chip size="small" label={play.channelLabel} variant="outlined" />
                          <Chip
                            size="small"
                            label={`Δ ${fmtBasisPoints(play.deltaBp)}`}
                            color="primary"
                          />
                        </Stack>
                      </Paper>
                    ))
                  ) : (
                    <Typography variant="body2" color="text.secondary">
                      Arsenal is syncing for this portfolio.
                    </Typography>
                  )}
                </Stack>
              </CardContent>
            </Card>
          </Grid>
        </Grid>

        <Card variant="outlined">
          <CardContent>
            <Typography variant="h6" sx={{ mb: 1 }}>
              Quarterly Outcomes (Plan → Metrics)
            </Typography>
            {thesisData?.quarterOutcomes.length ? (
              <Grid container spacing={2}>
                {thesisData.quarterOutcomes.map((quarter, idx) => (
                  <Grid item xs={12} md={6} key={quarter.quarter}>
                    <Paper variant="outlined" sx={{ p: 2, height: "100%" }}>
                      <Stack spacing={1}>
                        <Stack direction="row" spacing={1} alignItems="center">
                          <Typography variant="subtitle1">
                            {quarter.quarter}: {quarter.summary}
                          </Typography>
                          <Chip size="small" color="info" label="Planned" />
                        </Stack>
                        <Typography variant="body2" color="text.secondary">
                          Target lift: +{fmtBasisPoints(quarter.deltaBp)}
                        </Typography>
                        <Typography variant="body2" color="text.secondary">
                          Conversion-ready personas: {quarter.conversionReadyCount ?? 0}
                        </Typography>
                        {quarter.segmentSummary && (
                          <Typography variant="body2" color="text.secondary">
                            Segments: {quarter.segmentSummary}
                          </Typography>
                        )}
                        {quarter.personaDescriptors?.length ? (
                          <Typography variant="body2" color="text.secondary">
                            Focus personas:{" "}
                            {quarter.personaDescriptors
                              .map((descriptor) => descriptor.descriptor)
                              .filter(Boolean)
                              .join(", ")}
                          </Typography>
                        ) : null}
                      </Stack>
                    </Paper>
                  </Grid>
                ))}
              </Grid>
            ) : (
              <Typography variant="body2" color="text.secondary">
                Quarterly OKRs will appear after next run.
              </Typography>
            )}
          </CardContent>
        </Card>

        <Card variant="outlined">
          <CardContent>
            <Typography variant="h6">Cadence policy</Typography>
            <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
              Portfolio-wide mix of touches by phase.
            </Typography>
            <Box
              sx={{
                display: "flex",
                borderRadius: 1,
                overflow: "hidden",
                border: "1px solid",
                borderColor: "divider",
              }}
            >
              {FOUR_PHASE_KEYS.map((phase) => {
                const share =
                  portfolioCadenceStats.totalTouches > 0
                    ? portfolioCadenceStats.touches[phase] /
                      portfolioCadenceStats.totalTouches
                    : 0;
                return (
                  <Box
                    key={`policy-${phase}`}
                    sx={{
                      flex: share || 0.1,
                      minWidth: 40,
                      backgroundColor: stageColorFor(phase),
                      color: "common.white",
                      px: 1,
                      py: 0.5,
                    }}
                  >
                    <Typography variant="caption" sx={{ fontWeight: 600 }}>
                      {titleize(phase)} {fmtPercent(share)}
                    </Typography>
                  </Box>
                );
              })}
            </Box>
            <Typography variant="body2" sx={{ mt: 1 }}>
              At current settings we expect ~
              {portfolioCadenceStats.totalTouches && personaTotal
                ? Math.round(
                    (portfolioCadenceStats.totalTouches / personaTotal) * 4
                  )
                : 0}{" "}
              touches per persona per month across the portfolio.
            </Typography>
          </CardContent>
        </Card>

        <FourQuestionsPortfolio
          plan={plan}
          thesis={thesisData}
          summary={portfolioSummaryReport}
          aggregatedArsenal={aggregatedPortfolioArsenal}
          personaEngagements={portfolioPersonaEngagements}
          cadenceStats={portfolioCadenceStats}
        />

        <Card variant="outlined">
          <CardContent>
            <Typography variant="subtitle1" sx={{ mb: 1 }}>
              Persona Journey Focus
            </Typography>
            {portfolioJourneySteps.length ? (
              <Stack spacing={1}>
                <Stack direction="row" spacing={1} alignItems="center" flexWrap="wrap">
                  {portfolioJourneySteps.slice(0, 3).map((step) => (
                    <Chip
                      key={`journey-${step.step}`}
                      label={`${step.step}. ${step.persona}`}
                      variant="outlined"
                    />
                  ))}
                </Stack>
                <Button size="small">Show all paths</Button>
              </Stack>
            ) : (
              <Typography variant="body2" color="text.secondary">
                Canonical journey is still loading for this portfolio.
              </Typography>
            )}
          </CardContent>
        </Card>
      </Stack>
    );
  }

const renderArsenalContent = () => {
  if (!plan) return null;
  if (selectedAccountPlan) {
    return (
      <Stack spacing={2}>
        <Card variant="outlined">
          <CardContent>
            <Typography variant="h6" sx={{ mb: 1 }}>
              Account Arsenal
            </Typography>
            <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
              Plays, assets, and cadence specific to {selectedAccountPlan.account_name}.
            </Typography>
            {accountAssetCadence.length ? (
              <AssetCadenceTable
                rows={accountAssetCadence}
                personaLabelLookup={personaLabelLookup}
              />
            ) : (
              <Typography variant="body2" color="text.secondary">
                Run the planner for this account to populate arsenal details.
              </Typography>
            )}
          </CardContent>
        </Card>
      </Stack>
    );
  }
  const rows = filteredArsenalEntries;

  return (
    <Stack spacing={2}>
      <Card variant="outlined">
        <CardContent>
          <Typography variant="h6" sx={{ mb: 1 }}>
            Arsenal Reference — Plays & Evidence
          </Typography>
          <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
            Filter by persona or lift to find the right playbook quickly.
          </Typography>
          <Stack direction="row" spacing={1} sx={{ flexWrap: "wrap" }}>
            <Button
              size="small"
              variant={arsenalPreset === null ? "contained" : "outlined"}
              onClick={() => setArsenalPreset(null)}
            >
              All plays
            </Button>
            <Button
              size="small"
              variant={arsenalPreset === "legalExec" ? "contained" : "outlined"}
              onClick={() => setArsenalPreset("legalExec")}
            >
              Executive legal
            </Button>
            <Button
              size="small"
              variant={arsenalPreset === "analyticsOps" ? "contained" : "outlined"}
              onClick={() => setArsenalPreset("analyticsOps")}
            >
              Analytics operators
            </Button>
            <Button
              size="small"
              variant={arsenalPreset === "highLift" ? "contained" : "outlined"}
              onClick={() => setArsenalPreset("highLift")}
            >
              High lift &gt;300 bps
            </Button>
            <Button
              size="small"
              variant={arsenalPreset === "shortDuration" ? "contained" : "outlined"}
              onClick={() => setArsenalPreset("shortDuration")}
            >
              Short duration (&lt;2 weeks)
            </Button>
          </Stack>
          <Stack direction="row" spacing={1} sx={{ mt: 2, flexWrap: "wrap" }}>
            <Button variant="contained" size="small">
              Add new play
            </Button>
            <Button variant="outlined" size="small">
              Mark asset unavailable
            </Button>
          </Stack>
        </CardContent>
      </Card>

      <Card variant="outlined">
        <CardContent>
          <Typography variant="subtitle1" sx={{ mb: 1 }}>
            Library view
          </Typography>
          {rows.length ? (
            <TableContainer component={Paper} variant="outlined">
              <Table size="small">
                <TableHead>
                  <TableRow>
                    <TableCell>Asset</TableCell>
                    <TableCell>Channel</TableCell>
                    <TableCell>Segments</TableCell>
                    <TableCell>Personas</TableCell>
                    <TableCell align="right">Δ belief</TableCell>
                    <TableCell align="right">Confidence</TableCell>
                    <TableCell>Actions</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {rows.map((entry, idx) => (
                    <React.Fragment key={entry.key}>
                      <TableRow hover>
                        <TableCell>
                          <Typography variant="body2">{entry.assetLabel}</Typography>
                          <Typography variant="caption" color="text.secondary">
                            {entry.assetFormat || entry.sampleRow.assetMeta?.type || "Asset"}
                          </Typography>
                        </TableCell>
                        <TableCell>
                          <Typography variant="body2">{entry.channelLabel}</Typography>
                        </TableCell>
                        <TableCell>
                          <Stack direction="row" spacing={0.5} sx={{ flexWrap: "wrap" }}>
                            {entry.segments.slice(0, 2).map((segment) => (
                              <Chip key={`${entry.key}-${segment}`} label={segment} size="small" />
                            ))}
                            {entry.segments.length > 2 && (
                              <Typography variant="caption" color="text.secondary">
                                +{entry.segments.length - 2} more
                              </Typography>
                            )}
                          </Stack>
                        </TableCell>
                        <TableCell>
                          <Stack direction="row" spacing={0.5} sx={{ flexWrap: "wrap" }}>
                            {entry.personas.slice(0, 2).map((persona) => (
                              <Chip key={`${entry.key}-${persona}`} label={persona} size="small" variant="outlined" />
                            ))}
                            {entry.personas.length > 2 && (
                              <Typography variant="caption" color="text.secondary">
                                +{entry.personas.length - 2}
                              </Typography>
                            )}
                          </Stack>
                        </TableCell>
                        <TableCell align="right">{fmtBasisPoints(entry.deltaBp)}</TableCell>
                        <TableCell align="right">
                          {entry.confidence !== null ? fmtPercent(entry.confidence) : "—"}
                        </TableCell>
                        <TableCell>
                          <Stack direction="row" spacing={1}>
                            <Button
                              size="small"
                              onClick={() =>
                                setOpenArsenalReasoning((prev) =>
                                  prev === entry.key ? null : entry.key
                                )
                              }
                            >
                              {openArsenalReasoning === entry.key ? "Hide reasoning" : "View reasoning"}
                            </Button>
                            {idx < 3 && (
                              <Chip size="small" color="primary" label="Top pick" />
                            )}
                          </Stack>
                        </TableCell>
                      </TableRow>
                      <TableRow>
                        <TableCell colSpan={7} sx={{ p: 0 }}>
                          <Collapse in={openArsenalReasoning === entry.key} timeout="auto" unmountOnExit>
                            <Box sx={{ p: 2, backgroundColor: "grey.50" }}>
                              <Typography variant="subtitle2" sx={{ mb: 1 }}>
                                Why this play works
                              </Typography>
                              {renderRationaleCell(entry.sampleRow)}
                            </Box>
                          </Collapse>
                        </TableCell>
                      </TableRow>
                    </React.Fragment>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          ) : (
            <Typography variant="body2" color="text.secondary">
              No arsenal rows match this filter yet.
            </Typography>
          )}
        </CardContent>
      </Card>
    </Stack>
  );
};

    return (
      <Stack spacing={2}>
        <Card variant="outlined">
          <CardContent>
            <Typography variant="subtitle1" sx={{ mb: 1 }}>
              People Coverage
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Coverage across required personas: {coverageValue} ({matchedPeopleCount}/
              {requiredPersonaCount})
            </Typography>
            {unmatchedPersonaReqs.length ? (
              <Alert severity="warning" sx={{ mt: 2 }}>
                <Typography variant="body2" sx={{ mb: 1 }}>
                  The following personas still need real people assigned:
                </Typography>
                <Stack direction="row" spacing={1} sx={{ flexWrap: "wrap" }}>
                  {unmatchedPersonaReqs.map((req) => (
                    <Chip
                      key={req.persona_id}
                      label={`${req.persona_label || req.persona_id} · ${req.stage_label}`}
                      variant="outlined"
                      size="small"
                    />
                  ))}
                </Stack>
              </Alert>
            ) : (
              <Alert severity="success" sx={{ mt: 2 }}>
                Every required persona in this account has at least one person mapped.
              </Alert>
            )}
          </CardContent>
        </Card>

        {zmotWatchlist.length ? (
          <Card variant="outlined">
            <CardContent>
              <Typography variant="subtitle1" sx={{ mb: 1 }}>
                ZMOT Watchlist
              </Typography>
              <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                Track upcoming Zero-Moment-of-Truth events that raise perceptibility for
                key personas before belief transitions.
              </Typography>
              <Stack spacing={2}>
                {zmotWatchlist.map((watch) => {
                  const personaName = watch.persona_label || watch.persona_id;
                  const hasObserved = watch.events.some((event) => event.already_observed);
                  return (
                    <Box key={watch.persona_id}>
                      <Stack direction="row" spacing={1} alignItems="center">
                        <Typography variant="subtitle2">{personaName}</Typography>
                        {hasObserved ? (
                          <Chip size="small" color="success" label="Observed signals in journey" />
                        ) : null}
                      </Stack>
                      <Stack spacing={1} sx={{ mt: 1 }}>
                        {watch.events.map((event, idx) => {
                          const eventLabel = event.zmot_label || event.zmot_event_id || "ZMOT event";
                          const key = `${watch.persona_id}-${event.zmot_event_id || idx}`;
                          const boostValue =
                            event.boost ?? event.trigger_boost ?? null;
                          return (
                            <Paper
                              key={key}
                              variant="outlined"
                              sx={{
                                p: 1.25,
                                borderStyle: event.already_observed ? "dashed" : "solid",
                                borderColor: event.already_observed ? "success.light" : "divider",
                              }}
                            >
                              <Stack
                                direction={{ xs: "column", sm: "row" }}
                                spacing={1}
                                alignItems={{ xs: "flex-start", sm: "center" }}
                                justifyContent="space-between"
                              >
                                <Stack direction="row" spacing={1} alignItems="center" flexWrap="wrap">
                                  <Chip size="small" color="info" label={eventLabel} />
                                  {event.pain_trigger_label && (
                                    <Typography variant="caption" color="text.secondary">
                                      Trigger: {event.pain_trigger_label}
                                    </Typography>
                                  )}
                                  {event.pain_label && (
                                    <Typography variant="caption" color="text.secondary">
                                      Pain: {event.pain_label}
                                    </Typography>
                                  )}
                                </Stack>
                                {boostValue !== null && boostValue !== undefined ? (
                                  <Typography variant="caption" color="text.secondary">
                                    Boost weight {fmtNumber(boostValue, 3)}
                                  </Typography>
                                ) : null}
                              </Stack>
                              {event.observable_moments?.length ? (
                                <Typography
                                  variant="caption"
                                  color="text.secondary"
                                  sx={{ display: "block", mt: 0.75 }}
                                >
                                  Observable moments: {formatObservables(event.observable_moments)}
                                </Typography>
                              ) : null}
                              {event.keywords?.length ? (
                                <Typography
                                  variant="caption"
                                  color="text.secondary"
                                  sx={{ display: "block", mt: 0.5 }}
                                >
                                  Keywords: {formatKeywords(event.keywords)}
                                </Typography>
                              ) : null}
                            </Paper>
                          );
                        })}
                      </Stack>
                    </Box>
                  );
                })}
              </Stack>
            </CardContent>
          </Card>
        ) : null}

        {zmotEventPortfolio.length ? (
          <Card variant="outlined">
            <CardContent>
              <Typography variant="subtitle1" sx={{ mb: 1 }}>
                Priority ZMOT Events to Monitor
              </Typography>
              <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                Feed these signals into your monitoring cadences to pre-empt pains and accelerate
                activation.
              </Typography>
              <TableContainer component={Paper} variant="outlined">
                <Table size="small">
                  <TableHead>
                    <TableRow>
                      <TableCell>ZMOT Event</TableCell>
                      <TableCell>Pain Trigger</TableCell>
                      <TableCell>Personas</TableCell>
                      <TableCell align="right">Boost</TableCell>
                      <TableCell>Observable Moments</TableCell>
                      <TableCell>Keywords</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {zmotEventPortfolio.slice(0, 10).map((event, idx) => {
                      const personas = (event.personas || []).map((p) =>
                        `${p.persona_label || p.persona_id}${p.already_observed ? " ✓" : ""}`
                      );
                      return (
                        <TableRow key={event.zmot_event_id || idx}>
                          <TableCell>{event.zmot_label || event.zmot_event_id || "ZMOT event"}</TableCell>
                          <TableCell>{event.pain_trigger_label || "—"}</TableCell>
                          <TableCell>
                            {personas.length ? personas.join(", ") : "—"}
                          </TableCell>
                          <TableCell align="right">
                            {event.boost || event.trigger_boost
                              ? fmtNumber(event.boost ?? event.trigger_boost ?? 0, 3)
                              : "—"}
                          </TableCell>
                          <TableCell>{formatObservables(event.observable_moments)}</TableCell>
                          <TableCell>{formatKeywords(event.keywords)}</TableCell>
                        </TableRow>
                      );
                    })}
                  </TableBody>
                </Table>
              </TableContainer>
            </CardContent>
          </Card>
        ) : null}

        <Card variant="outlined">
          <CardContent>
            <Typography variant="subtitle1" sx={{ mb: 1 }}>
              Conversion Sequence Blueprint
            </Typography>
            {accountConversionSequence.length ? (
              <Stack spacing={2}>
                {accountConversionSequence.map((entry) =>
                  renderConversionSequenceEntry(entry)
                )}
              </Stack>
            ) : (
              <Typography variant="body2" color="text.secondary">
                Conversion sequence will populate once engagements are generated.
              </Typography>
            )}
          </CardContent>
        </Card>

        <Card variant="outlined">
          <CardContent>
            <Typography variant="subtitle1" sx={{ mb: 1 }}>
              Campaign Execution
            </Typography>
            {accountCampaigns.length ? (
              <Stack spacing={2}>
                {accountCampaigns.map((campaign) => renderCampaignCard(campaign))}
              </Stack>
            ) : (
              <Typography variant="body2" color="text.secondary">
                No campaigns generated yet—run the planner after belief thesis is ready.
              </Typography>
            )}
          </CardContent>
        </Card>

        {accountAssetCadence.length ? (
          <Card variant="outlined">
            <CardContent>
              <Typography variant="subtitle1" sx={{ mb: 1 }}>
                Asset Cadence Library
              </Typography>
              <AssetCadenceTable
                rows={accountAssetCadence}
                personaLabelLookup={personaLabelLookup}
              />
            </CardContent>
          </Card>
        ) : null}

        <Card variant="outlined">
          <CardContent>
            <Typography variant="subtitle1" sx={{ mb: 1 }}>
              Persona Paths
            </Typography>
            <Stack spacing={1.5}>
              {selectedAccountPlan.prediction.persona_paths.map((path) => (
                <Box
                  key={path.id}
                  sx={{
                    border: "1px solid",
                    borderColor: path.is_primary ? "primary.main" : "divider",
                    borderRadius: 1,
                    p: 1,
                  }}
                >
                  <Stack direction="row" spacing={1} alignItems="center">
                    <Typography variant="subtitle2">
                      {path.is_primary ? "Primary path" : "Alternate path"}
                    </Typography>
                    <Chip
                      size="small"
                      label={`P=${fmtPercent(path.probability)}`}
                      color={path.is_primary ? "primary" : "default"}
                    />
                    {path.score !== undefined && (
                      <Chip
                        size="small"
                        label={`Score ${fmtNumber(path.score)}`}
                        variant="outlined"
                      />
                    )}
                  </Stack>
                  <Stack direction="row" spacing={2} sx={{ mt: 1, flexWrap: "wrap" }}>
                    {path.personas.map((persona) => {
                      const committeeLikelihood =
                        persona.expected_in_deal_prob ??
                        (persona.expected_in_deal_pct !== undefined &&
                        persona.expected_in_deal_pct !== null
                          ? persona.expected_in_deal_pct / 100
                          : null);
                      const personaPhase = dominantPhaseFromProbs(
                        persona.phase_probs,
                        persona.dominant_phase
                      );
                      return (
                        <Box key={persona.id} sx={{ minWidth: 180 }}>
                          <Tooltip title={personaTooltip(persona)}>
                            <Chip
                              label={`${persona.order + 1}. ${persona.label}`}
                              size="small"
                              variant={path.is_primary ? "filled" : "outlined"}
                              color={path.is_primary ? "primary" : "default"}
                            />
                          </Tooltip>
                          <Typography variant="caption" color="text.secondary" sx={{ display: "block", mt: 0.5 }}>
                            Belief {fmtPercent(persona.belief_level ?? null)} · Next {fmtPercent(persona.expected_next_prob ?? null)}
                          </Typography>
                          {personaPhase ? (
                            <Typography variant="caption" color="text.secondary" sx={{ display: "block" }}>
                              Belief phase {personaPhase}
                            </Typography>
                          ) : null}
                          {committeeLikelihood !== null ? (
                            <Typography variant="caption" color="text.secondary" sx={{ display: "block" }}>
                              Committee likelihood {fmtPercent(committeeLikelihood)}
                            </Typography>
                          ) : null}
                          {persona.fatigue !== null && persona.fatigue !== undefined ? (
                            <Typography variant="caption" color="text.secondary">
                              Fatigue {fmtPercent(persona.fatigue)}{" "}
                              {persona.fatigue_reason ? `(${persona.fatigue_reason})` : ""}
                            </Typography>
                          ) : null}
                          {persona.matched_people && persona.matched_people.length ? (
                            <Stack direction="row" spacing={0.5} sx={{ flexWrap: "wrap", mt: 0.75 }}>
                              {[...persona.matched_people]
                                .sort((a, b) => {
                                  const scoreA =
                                    a.committee_probability ??
                                    a.person_involvement_score ??
                                    a.person_belief_level ??
                                    0;
                                  const scoreB =
                                    b.committee_probability ??
                                    b.person_involvement_score ??
                                    b.person_belief_level ??
                                    0;
                                  return scoreB - scoreA;
                                })
                                .map((match) => (
                                  <Tooltip key={match.match_id} title={personMatchTooltip(match)}>
                                    <Chip label={personMatchLabel(match)} size="small" variant="outlined" />
                                  </Tooltip>
                                ))}
                            </Stack>
                          ) : (
                            <Typography
                              variant="caption"
                              color="error"
                              sx={{ display: "block", mt: 0.75 }}
                            >
                              No people assigned
                            </Typography>
                          )}
                          {persona.top_people && persona.top_people.length ? (
                            <Typography variant="caption" color="text.secondary" sx={{ display: "block", mt: 0.5 }}>
                              Top targets:{" "}
                              {persona.top_people
                                .map((person) => {
                                  const base = `${person.display_name || "Person"} (${fmtPercent(
                                    person.committee_probability ??
                                      person.person_involvement_score ??
                                      person.person_belief_level ??
                                      null
                                  )})`;
                                  const phase = dominantPhaseFromProbs(
                                    person.belief_phase_probs,
                                    person.dominant_phase
                                  );
                                  return phase ? `${base} · ${phase}` : base;
                                })
                                .join(", ")}
                            </Typography>
                          ) : null}
                        </Box>
                      );
                    })}
                  </Stack>
                </Box>
              ))}
            </Stack>
          </CardContent>
        </Card>

        <Card variant="outlined">
          <CardContent>
            <Typography variant="subtitle1" sx={{ mb: 1 }}>
              Persona Scatter (Proximity × Perceptibility)
            </Typography>
            <TableContainer component={Paper} variant="outlined">
              <Table size="small">
                <TableHead>
                  <TableRow>
                    <TableCell>Persona</TableCell>
                    <TableCell>Belief Phase</TableCell>
                    <TableCell>Matched People</TableCell>
                    <TableCell align="right">Perceptibility</TableCell>
                    <TableCell align="right">Proximity</TableCell>
                    <TableCell align="right">Involvement</TableCell>
                    <TableCell align="right">Path Prob.</TableCell>
                    <TableCell align="right">Belief</TableCell>
                    <TableCell align="right">Next Prob.</TableCell>
                    <TableCell align="right">Deal Likelihood</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {(selectedAccountPlan.scatter?.personas || []).map((persona) => (
                    <TableRow key={`${persona.id}-${persona.order}`}>
                      <TableCell>{persona.label}</TableCell>
                      <TableCell>
                        {dominantPhaseFromProbs(persona.phase_probs, persona.dominant_phase) || "—"}
                      </TableCell>
                      <TableCell>
                        {persona.matched_people && persona.matched_people.length ? (
                          <Stack direction="row" spacing={0.5} sx={{ flexWrap: "wrap" }}>
                            {persona.matched_people.map((match) => (
                              <Tooltip key={match.match_id} title={personMatchTooltip(match)}>
                                <Chip label={personMatchLabel(match)} size="small" variant="outlined" />
                              </Tooltip>
                            ))}
                          </Stack>
                        ) : (
                          <Typography variant="caption" color="error">
                            No people assigned
                          </Typography>
                        )}
                      </TableCell>
                      <TableCell align="right">
                        {fmtNumber(persona.perceptibility)}
                      </TableCell>
                      <TableCell align="right">
                        {fmtNumber(persona.proximity)}
                      </TableCell>
                      <TableCell align="right">
                        {fmtNumber(persona.involvement)}
                      </TableCell>
                      <TableCell align="right">
                        {fmtPercent(persona.path_probability)}
                      </TableCell>
                      <TableCell align="right">
                        {fmtPercent(persona.belief_level ?? null)}
                      </TableCell>
                      <TableCell align="right">
                        {fmtPercent(persona.expected_next_prob ?? null)}
                      </TableCell>
                      <TableCell align="right">
                        {fmtPercent(
                          persona.expected_in_deal_prob ??
                            (persona.expected_in_deal_pct !== undefined &&
                            persona.expected_in_deal_pct !== null
                              ? persona.expected_in_deal_pct / 100
                              : null)
                        )}
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          </CardContent>
        </Card>

        <Card variant="outlined">
          <CardContent>
            <Typography variant="subtitle1" sx={{ mb: 1 }}>
              Belief Transitions
            </Typography>
            <Stack spacing={1.5}>
              {(selectedAccountPlan.transitions || []).map((transition, idx) => (
                <Box
                  key={`${transition.persona.id}-${idx}`}
                  sx={{
                    border: "1px solid",
                    borderColor: "divider",
                    borderRadius: 1,
                    p: 1,
                  }}
                >
                  <Typography variant="subtitle2">
                    Stage {transition.stage_index + 1}: {transition.persona.label}
                  </Typography>
                <Typography variant="body2" color="text.secondary">
                  Problem: {transition.belief_transition?.problem?.label || "—"} · Pain:{" "}
                  {transition.belief_transition?.pain?.label || "—"} · Resolution:{" "}
                  {transition.belief_transition?.resolution?.label || "—"}
                </Typography>
                {transition.persona?.matched_people && transition.persona.matched_people.length ? (
                  <Stack direction="row" spacing={0.5} sx={{ flexWrap: "wrap", mt: 0.75 }}>
                    {transition.persona.matched_people.map((match) => (
                      <Tooltip key={match.match_id} title={personMatchTooltip(match)}>
                        <Chip label={personMatchLabel(match)} size="small" variant="outlined" />
                      </Tooltip>
                    ))}
                  </Stack>
                ) : (
                  <Typography variant="caption" color="error" sx={{ display: "block", mt: 0.75 }}>
                    No people assigned
                  </Typography>
                )}
                </Box>
              ))}
            </Stack>
          </CardContent>
        </Card>

        <Card variant="outlined">
          <CardContent>
            <Typography variant="subtitle1" sx={{ mb: 1 }}>
              Explore vs Exploit Strategy
            </Typography>
            {selectedAccountPlan.randomization?.path_allocation?.length ? (
              <TableContainer component={Paper} variant="outlined">
                <Table size="small">
                  <TableHead>
                    <TableRow>
                      <TableCell>Path</TableCell>
                      <TableCell align="right">Probability</TableCell>
                      <TableCell align="right">Exploration</TableCell>
                      <TableCell>Personas</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {selectedAccountPlan.randomization.path_allocation.map((row) => (
                      <TableRow key={row.path_id}>
                        <TableCell>{row.path_id}</TableCell>
                        <TableCell align="right">
                          {fmtPercent(row.probability)}
                        </TableCell>
                        <TableCell align="right">
                          {fmtPercent(row.exploration_weight)}
                        </TableCell>
                        <TableCell>{row.persona_ids.join(" → ") || "—"}</TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </TableContainer>
            ) : (
              <Typography variant="body2" color="text.secondary">
                No randomization policy available yet.
              </Typography>
            )}
            <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
              Early-cycle exploration:{" "}
              {fmtPercent(
                selectedAccountPlan.randomization?.belief_spread?.early_cycle_avg
              )}{" "}
              · Late-cycle:{" "}
              {fmtPercent(
                selectedAccountPlan.randomization?.belief_spread?.late_cycle_avg
              )}
            </Typography>
            <Typography variant="caption" color="text.secondary">
              {selectedAccountPlan.randomization?.notes || "—"}
            </Typography>
          </CardContent>
        </Card>
      </Stack>
    );
  };

  if (loading) {
    return (
      <Box
        sx={{
          p: 6,
          display: "flex",
          justifyContent: "center",
          alignItems: "center",
          minHeight: 240,
        }}
      >
        <CircularProgress size={32} />
      </Box>
    );
  }

  if (error) {
    return (
      <Box sx={{ p: 3 }}>
        <Alert severity="error">{error}</Alert>
      </Box>
    );
  }

  if (!plan) {
    return (
      <Box sx={{ p: 3 }}>
        <Alert severity="warning">
          Unable to load a marketing plan for the selected product.
        </Alert>
      </Box>
    );
  }

  return (
    <Box sx={{ p: 3 }}>
      <Stack direction={{ xs: "column", md: "row" }} spacing={2} alignItems={{ xs: "flex-start", md: "center" }} sx={{ mb: 2 }}>
        <Typography variant="h5">Marketing Planner</Typography>

        <Box sx={{ display: "flex", gap: 1, alignItems: "center", ml: { xs: 0, md: "auto" }, flexWrap: "wrap" }}>
          {products.length > 0 && (
            <FormControl size="small" sx={{ minWidth: 200 }}>
              <InputLabel id="product-select-label">Product</InputLabel>
              <Select
                labelId="product-select-label"
                label="Product"
                value={selectedProductId}
                onChange={(e) => setSelectedProductId(String(e.target.value))}
              >
                {products.map((product) => (
                  <MenuItem key={product.id} value={product.id}>
                    {product.name || product.id}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
          )}

          {plan?.accounts?.length ? (
            <FormControl size="small" sx={{ minWidth: 220 }}>
              <InputLabel id="view-select-label">View</InputLabel>
              <Select
                labelId="view-select-label"
                label="View"
                value={viewAccountId}
                onChange={(e) => setViewAccountId(String(e.target.value))}
              >
                <MenuItem value="portfolio">Portfolio (All Accounts)</MenuItem>
                {plan.accounts.map((account) => (
                  <MenuItem key={account.account_id} value={account.account_id}>
                    {account.account_name}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
          ) : null}

          <Button
            variant="outlined"
            size="small"
            component={Link}
            to="/account-plan"
          >
            Account Plans →
          </Button>
        </Box>
      </Stack>

      {!!statusMsg && (
        <Alert severity="info" sx={{ mb: 2 }}>
          {statusMsg}
        </Alert>
      )}

      <Stack spacing={2} sx={{ mb: 3 }}>
        <Card variant="outlined">
          <CardContent>
            <Typography variant="subtitle1" sx={{ mb: 1 }}>
              {summaryTitle}
            </Typography>
            <Stack direction="row" spacing={3} sx={{ flexWrap: "wrap" }}>
              {selectedAccountPlan ? (
                <>
                  <SummaryMetric label="Best Path Probability" value={bestPathValue} />
                  <SummaryMetric label="Prediction Accuracy" value={accuracyValue} />
                  <SummaryMetric label="Expected Lift" value={liftValue} />
                  <SummaryMetric label="People Coverage" value={coverageValue} />
                </>
              ) : (
                <>
                  <SummaryMetric
                    label="Accounts"
                    value={String(portfolioSummaryReport?.totalAccounts ?? accounts.length)}
                  />
                  <SummaryMetric
                    label="Enrichment %"
                    value={fmtPercent(portfolioEnrichmentRatio)}
                  />
                  <SummaryMetric label="Avg Best Path Probability" value={bestPathValue} />
                  <SummaryMetric label="Avg Accuracy" value={fmtPercent(portfolioAvgAccuracy)} />
                  <SummaryMetric
                    label="Total Expected Lift"
                    value={fmtBasisPoints(portfolioSummaryReport?.totalDeltaBp ?? 0)}
                  />
                  <SummaryMetric
                    label="Avg Coverage"
                    value={fmtPercent(portfolioSummaryReport?.coveragePct ?? null)}
                  />
                  <SummaryMetric
                    label="Avg Depth"
                    value={fmtPercent(portfolioSummaryReport?.maturityScore ?? null)}
                  />
                  <SummaryMetric
                    label="Time to Conversion"
                    value={toWeeks(
                      portfolioSummaryReport?.expectedTimeToConversionDays ?? undefined
                    )}
                  />
                </>
              )}
            </Stack>

            {!selectedAccountPlan &&
              (plan.summary?.unmatched_persona_count || 0) > 0 && (
                <Alert severity="warning" sx={{ mt: 2 }}>
                  {plan.summary?.unmatched_persona_count || 0} persona
                  {(plan.summary?.unmatched_persona_count || 0) === 1 ? "" : "s"} across the
                  portfolio still need real people assigned.
                </Alert>
              )}

            {!selectedAccountPlan && portfolioSummaryReport ? (
              <Stack
                direction={{ xs: "column", md: "row" }}
                spacing={2}
                sx={{ mt: 2 }}
                alignItems="stretch"
              >
                <Box sx={{ flex: 1 }}>
                  <Typography variant="subtitle2">High-risk accounts</Typography>
                  {portfolioSummaryReport.highRiskAccounts.length ? (
                    <Stack spacing={0.75} sx={{ mt: 1 }}>
                      {portfolioSummaryReport.highRiskAccounts.map((acct) => (
                        <Paper
                          key={`risk-${acct.id}`}
                          variant="outlined"
                          sx={{ p: 1.25, borderColor: "error.light" }}
                        >
                          <Typography variant="body2">{acct.name}</Typography>
                          <Typography variant="caption" color="text.secondary">
                            {acct.reason}
                          </Typography>
                        </Paper>
                      ))}
                    </Stack>
                  ) : (
                    <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                      No blocking risks detected.
                    </Typography>
                  )}
                </Box>
                <Box sx={{ flex: 1 }}>
                  <Typography variant="subtitle2">High-opportunity accounts</Typography>
                  {portfolioSummaryReport.highOpportunityAccounts.length ? (
                    <Stack spacing={0.75} sx={{ mt: 1 }}>
                      {portfolioSummaryReport.highOpportunityAccounts.map((acct) => (
                        <Paper
                          key={`opp-${acct.id}`}
                          variant="outlined"
                          sx={{ p: 1.25, borderColor: "success.light" }}
                        >
                          <Typography variant="body2">{acct.name}</Typography>
                          <Typography variant="caption" color="text.secondary">
                            {acct.reason}
                          </Typography>
                        </Paper>
                      ))}
                    </Stack>
                  ) : (
                    <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                      Portfolio needs additional activation before identifying wins.
                    </Typography>
                  )}
                </Box>
              </Stack>
            ) : null}
          </CardContent>
        </Card>
        <Card variant="outlined">
          <CardContent>
            <Typography variant="subtitle1" sx={{ mb: 1 }}>
              {selectedAccountPlan ? "Campaign Mix" : "Portfolio Mix"}
            </Typography>
            {selectedAccountPlan ? (
              accountMix ? (
                <>
                  <Typography variant="body2" color="text.secondary">
                    Broad engagements: {accountMix.broad} · Focused engagements: {accountMix.focused} ·
                    Broad ratio: {fmtPercent(accountMix.ratio)}
                  </Typography>
                </>
              ) : (
                <Typography variant="body2" color="text.secondary">
                  No executions generated yet.
                </Typography>
              )
            ) : (
              <Typography variant="body2" color="text.secondary">
                Broad engagements: {portfolioMix?.broad ?? 0} · Focused engagements:{" "}
                {portfolioMix?.focused ?? 0} · Broad ratio: {fmtPercent(portfolioMix?.broad_ratio ?? null)}
              </Typography>
            )}
            <Divider sx={{ my: 2 }} />
            <Typography variant="subtitle2" sx={{ mb: 1 }}>
              Explore / Exploit Snapshot
            </Typography>
            <Typography variant="body2" color="text.secondary">
              {selectedAccountPlan
                ? `Paths tracked: ${selectedAccountPlan.randomization?.path_allocation?.length ?? 0}`
                : `Paths sampled: ${portfolioRandomization?.path_count_sampled ?? 0}`}
            </Typography>
          </CardContent>
        </Card>
      </Stack>

      <Box sx={{ borderBottom: 1, borderColor: "divider", mb: 2 }}>
        <Tabs
          value={plannerPanel}
          onChange={(_, value) => setPlannerPanel(value)}
          aria-label="marketing planner sections"
        >
          <Tab label="Overview" value="overview" />
          <Tab label="Execution Timeline" value="timeline" />
          <Tab label="Arsenal Reference" value="arsenal" />
        </Tabs>
      </Box>

      {plannerPanel === "overview"
        ? renderOverviewContent()
        : plannerPanel === "timeline"
        ? renderExecutionContent()
        : renderArsenalContent()}
      {renderAccountsPopover()}
    </Box>
  );
};

export default MarketingPlanner;
