// DROP-IN REPLACEMENT
export type SeqStep = {
  step: number;
  persona: string;
  concern_id: string;
  stage?: string;
  concern_label?: string;
  delta_lift?: number;
  win_likelihood?: number;
};

export type NormalizedSequence = {
  title?: string;
  sequence: { persona: string; concern_id: string; stage?: string; concern_label?: string }[];
  steps: SeqStep[];
  base_win?: number;
  final_win?: number;
  total_lift?: number;
};

const num = (v: any, d = 0) => (typeof v === "number" ? v : d);

/** Preferred-source resolution: prefer frozen_strategy first, then rcs, then tactical_update / output, then raw */
export function pickPreferredSource(rcs: any): any {
  if (!rcs) return {};
  if (rcs.frozen_strategy && Object.keys(rcs.frozen_strategy).length) return rcs.frozen_strategy;
  if (rcs.rcs && Object.keys(rcs.rcs).length) return rcs.rcs;
  if (rcs.tactical_update && Object.keys(rcs.tactical_update).length) return rcs.tactical_update;
  if (rcs.output) {
    if (rcs.output.frozen_strategy && Object.keys(rcs.output.frozen_strategy).length) return rcs.output.frozen_strategy;
    if (rcs.output.tactical_update && Object.keys(rcs.output.tactical_update).length) return rcs.output.tactical_update;
    if (Object.keys(rcs.output).length) return rcs.output;
  }
  return rcs;
}

/** Pick the block that actually has sequences, regardless of backend vintage */
function pickSequenceSource(rcs: any): any[] {
  const src = pickPreferredSource(rcs) || {};
  if (Array.isArray(src.concern_sequences) && src.concern_sequences.length) return src.concern_sequences;
  if (Array.isArray(src.sequences_and_campaigns) && src.sequences_and_campaigns.length) return src.sequences_and_campaigns;
  if (Array.isArray(src.next_sequences) && src.next_sequences.length) return src.next_sequences;
  if (Array.isArray(src.themes) && src.themes.length) return src.themes;
  // also tolerate top-level sequences if nothing preferred contained them
  if (Array.isArray(rcs?.concern_sequences) && rcs.concern_sequences.length) return rcs.concern_sequences;
  if (Array.isArray(rcs?.sequences_and_campaigns) && rcs.sequences_and_campaigns.length) return rcs.sequences_and_campaigns;
  return [];
}

/** Normalize sequences/campaigns for <RCSCampaignSequences /> */
export function normalizeRcsSequences(rcs: any): NormalizedSequence[] {
  const src = pickSequenceSource(rcs);
  if (!src.length) return [];

  const preferred = pickPreferredSource(rcs) || {};
  const baseline =
    num(preferred?.graph_win_likelihood) ||
    num(preferred?.baseline?.win_likelihood) ||
    num(rcs?.graph_win_likelihood) ||
    num(rcs?.baseline?.win_likelihood) ||
    0;

  return src.map((it: any) => {
    const title = it.title || it.theme || undefined;
    const seqArray: any[] = Array.isArray(it.sequence) ? it.sequence : [];
    const steps: SeqStep[] = [];

    let cum = baseline;
    let totalLift = 0;

    seqArray.forEach((s, i) => {
      const delta = num(s.lift_proxy);
      totalLift += delta;
      cum = Math.max(cum, baseline + totalLift);
      steps.push({
        step: i + 1,
        persona: s.persona || it.persona || s.pid,
        concern_id: s.concern_id || s.cid,
        stage: s.stage,
        concern_label: s.label || s.concern_label,
        delta_lift: delta,
        win_likelihood: cum,
      });
    });

    return {
      title,
      sequence: seqArray.map(s => ({
        persona: s.persona || it.persona || s.pid,
        concern_id: s.concern_id || s.cid,
        stage: s.stage,
        concern_label: s.label || s.concern_label,
      })),
      steps,
      base_win: baseline,
      final_win: num(it.final_win, baseline + totalLift),
      total_lift: totalLift,
    };
  });
}

/** Strategy meta for headers / summary blocks */
export function normalizeRcsStrategy(rcs: any) {
  const live = rcs || {};
  const preferred = pickPreferredSource(rcs) || {};
  // ensure frozen is the explicit frozen_strategy if present (preferred already prefers frozen)
  const frozen = preferred?.frozen_persona_pool ? preferred : live.frozen_strategy || live?.output?.frozen_strategy || preferred;

  const baseline =
    num(preferred?.baseline?.win_likelihood) ||
    num(live?.baseline?.win_likelihood) ||
    num(frozen?.baseline?.win_likelihood) ||
    0;

  // Personas: prefer frozen_persona_pool from frozen_strategy, then from preferred, then fallback to empty
  const personasSrc =
    preferred?.frozen_persona_pool ||
    frozen?.frozen_persona_pool ||
    preferred?.all_personas ||
    live?.all_personas ||
    [];

  const normalizePersonaEntry = (p: any) => {
    if (typeof p === "string") {
      return { id: p, label: String(p), involvement: 0, activation: 0 };
    }
    const id = String(p?.id ?? p?.persona ?? p?.key ?? p?.pid ?? "");
    // prefer explicit persona_label / label / name, otherwise fall back to id
    let label: any = p?.persona_label ?? p?.label ?? p?.name ?? p?.display_name ?? id;
    if (typeof label !== "string") {
      try {
        label = JSON.stringify(label);
      } catch {
        label = String(id);
      }
    }
    const involvement = Number(p?.involvement ?? p?.I ?? p?.involvement_score ?? 0) || 0;
    const activation = Number(p?.activation ?? p?.A ?? p?.activation_score ?? 0) || 0;
    return { id, label, involvement, activation };
  };

  const personas = Array.isArray(personasSrc) ? personasSrc.map(normalizePersonaEntry) : [];
  // Coalitions: prefer frozen then preferred then live
  const coalitions =
    preferred?.concern_coalitions ||
    preferred?.coalitions ||
    frozen?.concern_coalitions ||
    frozen?.coalitions ||
    live?.concern_coalitions ||
    live?.coalitions ||
    [];

  // Themes: frozen themes preferred, then sequences on preferred/live
  const themes =
    (preferred?.themes || frozen?.themes) ||
    (preferred?.concern_sequences || preferred?.sequences_and_campaigns) ||
    (live?.concern_sequences || live?.sequences_and_campaigns) ||
    [];

  return {
    baseline_win: baseline,
    personas,
    coalitions,
    themes,
  };
}