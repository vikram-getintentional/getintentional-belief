export type BeliefStage =
  | "Unaware"
  | "ZMOT"
  | "Problem"
  | "InternalBarrier"
  | "Evaluation"
  | "Approval"
  | "Implementation"
  | "PromisedLand";

export interface MetaWithBeliefScale {
  generatedAt: string;
  beliefScale: BeliefStage[];
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
