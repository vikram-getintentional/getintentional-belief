// filepath: /Users/vikrambhaskaran/getintentional/frontend/src/types/index.ts
export type PersonaConcern = {
  persona: string;
  stage: string;
  ROI: number;
  lift: number;
  delta_care: number;
  description: any;
  plays: any[];
};

export type PersonaCardRCSPayload = {
  persona_title: any;
  persona_departments: any;
  persona_seniority: any;
  persona_id: string;
  persona: { title: string; department: string; seniority: string };
  importance: number;        // involvement
  activation: number;
  care: number;
  marginal_lift: number;
  priority_score: number;    // importance * activation
  jobs: { description: string; relevance: number }[];
  pains: { description: string; relevance: number }[];
};

export type IcpCombo = {
  chips: { family: string; node_id: string; label: string }[];
  win_rate: number;
  lift_abs: number;
  lift_rel: number;
  zmots: Array<{
    zmot_event_id: string;
    zmot_label: string;
    score: number;
    from_chips: { node_id: string; label: string; edge_score: number }[];
    observable_moments: { id: string; label: string; fitness: number }[];
    trigger_keywords: { id: string; label: string; fitness: number }[];
  }>;
};

export type IcpUpliftsPayload = {
  graph_fingerprint?: string;
  baseline: { win_rate: number };
  combos: IcpCombo[];
};
