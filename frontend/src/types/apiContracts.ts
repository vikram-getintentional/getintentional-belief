export type BeliefStage =
  | "Unaware"
  | "ZMOT"
  | "Problem"
  | "InternalBarrier"
  | "Evaluation"
  | "Approval"
  | "Implementation"
  | "PromisedLand";

export type PlanningMode = "graph_hypothesis" | "observed_signal" | "learned_strategy";

export interface MetaWithBeliefScale {
  generatedAt: string;
  beliefScale: BeliefStage[];
  planningMode?: PlanningMode;
  version?: string;
}

export interface PortfolioKeyStats {
  totalTargetAccounts: number;
  totalPersonasToEngage: number;
  expectedWinsPct: number;
  averageAccountBelief: string;
  timeToWinMonths: number;
  wolfIdentifiedPct: number;
  avgEnablementReadiness: number;
}

export interface DecisionDriver {
  personaId: string;
  personaName: string;
  winGateSharePct: number;
  medianDaysBeforeClose: number;
  typicalBlockingBeliefs: string[];
}

export interface SubsidyDownstreamPersona {
  personaId: string;
  personaName: string;
  changeBurdenDescription: string;
  failureRisk: "LOW" | "MEDIUM" | "HIGH";
}

export interface SubsidyMapEntry {
  wolfPersonaId: string;
  wolfPersonaName: string;
  avgSubsidyPersonasCount: number;
  downstreamSubsidies: SubsidyDownstreamPersona[];
}

export interface BeliefProgression {
  canonicalPath: BeliefStage[];
  medianDaysBetweenStages: Record<string, number>;
  dropOffPctByStage: Record<string, number>;
}

export interface PainTheme {
  id: string;
  label: string;
  prevalenceInWinsPct: number;
  dominantStage: BeliefStage | string;
}

export interface PortfolioCampaignTimeframe {
  startDate: string;
  endDate: string;
}

export interface BeliefShift {
  id: string;
  targetPersonaId: string;
  targetPersonaName: string;
  fromBelief: BeliefStage | string;
  toBelief: BeliefStage | string;
  corePainId: string;
  corePainLabel: string;
  wolfPersonaId: string;
  unblocksPersonaId: string | null;
  expectedLiftPct: number;
  confidence: number;
}

export interface ArsenalRow {
  asset: string;
  channels: string[];
  fitment: number;
  engagement: number;
  expectedLift: number;
  targetPersonaId: string;
  beliefShiftId: string;
  orgStage: string;
  avgTimeToImpactDays: number;
}

export interface PortfolioCampaign {
  id: string;
  description: string;
  timeframe: PortfolioCampaignTimeframe;
  personas: string[];
  beliefShifts: BeliefShift[];
  arsenalTable: ArsenalRow[];
}

export interface ArsenalAssetCategoryMeta {
  key: string;
  name: string;
  label?: string;
  activity_stages: string[];
  asset_ids?: string[];
}

export interface ArsenalChannelCategoryMeta {
  key: string;
  name: string;
  label?: string;
  channel_activity_stages: string[];
  delivery_mode?: string | null;
  channel_ids?: string[];
  reach_score_estimate?: number | null;
}

export interface PortfolioTheme {
  id: string;
  name: string;
  explanation: string;
  objective: string;
  targetAccounts: string[];
  campaigns: PortfolioCampaign[];
}

export interface RecommendedAction {
  label: string;
  route: string;
}

export interface QualityReadinessDimension {
  id: string;
  label: string;
  score: number;
  status: "ready" | "partial" | "missing" | string;
  reason: string;
  recommended_actions: RecommendedAction[];
}

export interface PlanReadiness {
  score: number;
  dimensions: QualityReadinessDimension[];
}

export interface PredictiveConfidenceComponent {
  id: string;
  score: number;
  note: string;
}

export interface PlanPredictiveConfidence {
  stars: number;
  score: number;
  components: PredictiveConfidenceComponent[];
  explanation: string;
}

export interface PlanQuality {
  readiness: PlanReadiness;
  predictiveConfidence: PlanPredictiveConfidence;
}

export interface ComprehensiveExecutionPlan {
  meta: {
    version: string;
    generatedAt: string;
    beliefScale: BeliefStage[];
  };
  portfolio: {
    id: string;
    name: string;
    keyStats: PortfolioKeyStats;
    decisionDrivers: DecisionDriver[];
    subsidyMap: SubsidyMapEntry[];
    beliefProgression: BeliefProgression;
    painThemes: PainTheme[];
  };
  themes: PortfolioTheme[];
  quality: PlanQuality;
}

export interface AccountPlanPersona {
  personaId: string;
  canonicalPersonaId: string;
  name: string;
  beliefStage: BeliefStage | string;
  influenceScore: number;
  gateType?: string;
}

export interface DecisionMap {
  primaryGate: AccountPlanPersona & { gateType: string };
  secondaryGates: AccountPlanPersona[];
  champion: AccountPlanPersona;
  supportingPersons: AccountPlanPersona[];
}

export interface EnablementQueueItem {
  personaId: string;
  canonicalPersonaId: string;
  personaName: string;
  fear: string;
  neededEvidence: string[];
  riskIfIgnored: "LOW" | "MEDIUM" | "HIGH";
  status: "GREEN" | "AMBER" | "RED" | string;
}

export interface AccountPlanNextBestAction {
  id: string;
  priority: number;
  targetPersonaId: string;
  canonicalPersonaId: string;
  targetPersonaName: string;
  fromBelief: BeliefStage | string;
  toBelief: BeliefStage | string;
  corePainId: string;
  corePainLabel: string;
  assetType: string;
  assetId: string | null;
  channel: string;
  whyNow: string;
  whatItEnables: string;
  expectedLiftPct: number;
  confidence: number;
  recommendedWindowDays: number;
}

export interface OutcomeSimulatorState {
  predictedWinPct: number;
  expectedCloseWindowDays: [number, number];
}

export interface OutcomeSimulator {
  current: OutcomeSimulatorState;
  ifRecommendedActionsExecuted: OutcomeSimulatorState;
  ifIgnored: {
    predictedWinPct: number;
    stallProbabilityPct: number;
    predictedStallInDays: number;
  };
}

export interface WinOutlookCoverage {
  similar_deals: number;
  wins: number;
  losses: number;
}

export interface WinOutlookDriver {
  feature: string;
  direction: "+" | "-";
  weight: number;
}

export interface WinOutlookDiagnostics {
  graph_walk_reachability: number;
  note?: string;
}

export interface WinOutlookRightToWin {
  p: number | null;
  ci: { low: number | null; high: number | null };
  coverage: WinOutlookCoverage;
  top_drivers: WinOutlookDriver[];
  status: string;
}

export interface WinOutlookEvidence {
  engagement_count: number;
  summary: string;
}

export interface WinOutlookBaselineWin {
  p: number | null;
  ci: { low: number | null; high: number | null };
  evidence: WinOutlookEvidence;
  status: string;
}

export interface WinOutlookForecastAssumption {
  action_id: string;
  label: string;
  p_engage: number;
  expected_delta_log_odds: number;
  history_note?: string;
  history_engagements?: number | null;
}

export interface WinOutlookPredictedWin {
  p: number | null;
  ci: { low: number | null; high: number | null };
  range: { low: number | null; high: number | null };
  assumptions: WinOutlookForecastAssumption[];
  lift_over_current: number;
  status: string;
  evidence: WinOutlookEvidence;
}

export interface WinOutlook {
  diagnostics: WinOutlookDiagnostics;
  right_to_win: WinOutlookRightToWin;
  baseline_win: WinOutlookBaselineWin;
  predicted_win: WinOutlookPredictedWin;
}

export interface EngagementSignals {
  observed: number;
  projected: number;
  lastObservedAt?: string | null;
}

export interface DataAvailability {
  graph: boolean;
  historicDeals: number;
  observedEngagements: number;
  arsenalAssets: number;
  peopleMapped: number;
}

export interface AccountPlanContract {
  meta: MetaWithBeliefScale;
  account: {
    id: string;
    name: string;
    segment: string;
    industry: string;
    currentBeliefStage: BeliefStage;
    dealValueEstimate: number;
    predictedWinPct: number;
    championStrength: number;
    enablementReadiness: number;
    momentum: string;
    predictedStallInDays: number;
  };
  decisionMap: DecisionMap;
  enablementQueue: {
    overallReadiness: number;
    items: EnablementQueueItem[];
  };
  nextBestActions: AccountPlanNextBestAction[];
  outcomeSimulator: OutcomeSimulator;
  quality: PlanQuality;
  winOutlook?: WinOutlook;
  engagementSignals?: EngagementSignals;
  dataAvailability?: DataAvailability;
}

export interface InsightEvidenceMetrics {
  metric: string;
  value: number;
}

export interface InsightEvidence {
  sampleSize: number;
  timeWindowDays: number;
  beforeMetric?: InsightEvidenceMetrics;
  afterMetric?: InsightEvidenceMetrics;
  affectedPersonaIds?: string[];
  correlation?: number;
}

export interface InsightScope {
  icpIds?: string[];
  segments?: string[];
  products?: string[];
}

export type InsightType = "DECISION_DRIVER_SHIFT" | "SUBSIDY_FAILURE" | "SUBSIDY_SUCCESS";

export type InsightSeverity = "LOW" | "MEDIUM" | "HIGH" | "CRITICAL";

export interface InsightCard {
  id: string;
  type: InsightType;
  severity: InsightSeverity;
  title: string;
  description: string;
  scope: InsightScope;
  evidence: InsightEvidence;
  implication: string;
  recommendedAction: string;
  createdAt: string;
  tags: string[];
}

export interface InsightsInbox {
  meta: {
    generatedAt: string;
    planningMode?: PlanningMode;
  };
  insights: InsightCard[];
}

export type CoalitionRole = "WOLF" | "CHAMPION" | "BLOCKER" | "DRIVER" | "USER";

export interface PersonaPainRef {
  id: string;
  label: string;
}

export interface PersonaAtlasEntry {
  id: string;
  name: string;
  departments: string[];
  exampleTitles: string[];
  seniority: string;
  coalitionRole: CoalitionRole;
  entryBeliefStage: BeliefStage | string;
  commonPains: PersonaPainRef[];
  occurrenceInWinsPct: number;
  avgTimeToActivationDays: number;
  typicalPositionInCoalition: number;
  exampleAccounts: string[];
  metaSignals: {
    oftenSubsidizedByPersonas?: string[];
    subsidizesWolves?: string[];
    blockingRiskIfNeglected?: string;
  };
}

export interface PersonasAtlas {
  meta: {
    generatedAt: string;
  };
  personas: PersonaAtlasEntry[];
}

export interface ICPEntry {
  id: string;
  label: string;
  industry: string;
  employeeRange: {
    min: number;
    max: number;
  };
  geo: string[];
  techStackSignals: string[];
  typicalWolfPersonaId: string;
  typicalWolfPersonaName: string;
  medianWinRatePct: number;
  medianSalesCycleDays: number;
  avgDealSize: number;
  avgSubsidyBurdenScore: number;
  dominantPains: string[];
  emergingTrends: string[];
  status: "CORE" | "EMERGING" | "DECLINING";
}

export interface ICPOverview {
  meta: {
    generatedAt: string;
  };
  icps: ICPEntry[];
}

export type ThesisAttributeType = "numeric" | "categorical" | "boolean";

export interface ThesisAttributeBin {
  id: string;
  label?: string | null;
  min?: number | null;
  max?: number | null;
}

export interface ThesisSegmentFeature {
  id: string;
  product_id: string;
  name: string;
  key: string;
  type: ThesisAttributeType;
  bins?: ThesisAttributeBin[];
  categories?: string[];
  created_at: string;
}

export type SegmentRuleConditionOperator = "IN" | "ANY";

export interface SegmentRuleCondition {
  feature_key: string;
  op: SegmentRuleConditionOperator;
  values?: string[];
}

export interface SegmentRules {
  conditions: SegmentRuleCondition[];
}

export interface ThesisSegment {
  id: string;
  product_id: string;
  name: string;
  description?: string | null;
  rules: SegmentRules;
  prior_weight: number;
  created_at: string;
}

export type DriftLevel = "stable" | "moderate" | "strong";

export interface DistributionSummary {
  prior: Record<string, number>;
  posterior: Record<string, number>;
  drift: number;
  drift_level: DriftLevel;
}

export interface AttributeDistributionSummary extends DistributionSummary {
  feature_key: string;
}

export interface PainDistributionSummary extends DistributionSummary {
  parent_pain_id?: string | null;
}

export interface ZmotDistributionSummary extends DistributionSummary {
  parent_pain_id: string;
}

export interface AspirationDistributionSummary extends DistributionSummary {
  parent_pain_id: string;
}

export interface BeliefDistributionSummary extends DistributionSummary {
  parent_pain_id: string;
  parent_aspiration_id: string;
}

export interface ThesisDistributions {
  attributes: AttributeDistributionSummary[];
  pains: PainDistributionSummary[];
  zmots: ZmotDistributionSummary[];
  aspirations: AspirationDistributionSummary[];
  beliefs: BeliefDistributionSummary[];
}

export interface SegmentThesisSummary {
  segment: ThesisSegment;
  distributions: ThesisDistributions;
}

export interface ThesisBuilderCall {
  call_id: string;
  account_id?: string | null;
  external_call_id?: string | null;
  attribute_bins: Record<string, string>;
  segment_id?: string | null;
  created_at: string;
}
