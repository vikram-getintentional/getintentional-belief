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

  // If backend wrapped payload under `report`, prefer that inner object
  const rpt = rcs.report && Object.keys(rcs.report).length ? rcs.report : rcs;

  if (rpt.frozen_strategy && Object.keys(rpt.frozen_strategy).length) return rpt.frozen_strategy;
  if (rpt.rcs && Object.keys(rpt.rcs).length) return rpt.rcs;
  if (rpt.tactical_update && Object.keys(rpt.tactical_update).length) return rpt.tactical_update;

  // support older "output" wrapper as well (maybe under top-level or report)
  const out = rpt.output || rcs.output;
  if (out) {
    if (out.frozen_strategy && Object.keys(out.frozen_strategy).length) return out.frozen_strategy;
    if (out.tactical_update && Object.keys(out.tactical_update).length) return out.tactical_update;
    if (Object.keys(out).length) return out;
  }

  return rpt;
}

/** Pick the block that actually has sequences, regardless of backend vintage */
function pickSequenceSource(rcs: any): any[] {
  const src = pickPreferredSource(rcs) || {};
  if (Array.isArray(src.concern_sequences) && src.concern_sequences.length) return src.concern_sequences;
  if (Array.isArray(src.sequences_and_campaigns) && src.sequences_and_campaigns.length) return src.sequences_and_campaigns;
  if (Array.isArray(src.next_sequences) && src.next_sequences.length) return src.next_sequences;
  if (Array.isArray(src.themes) && src.themes.length) return src.themes;
  // also tolerate top-level sequences if nothing preferred contained them
  const raw = rcs?.report ?? rcs;
  if (Array.isArray(raw?.concern_sequences) && raw.concern_sequences.length) return raw.concern_sequences;
  if (Array.isArray(raw?.sequences_and_campaigns) && raw.sequences_and_campaigns.length) return raw.sequences_and_campaigns;
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
    preferred?.persona_pool ||
    frozen?.persona_pool ||
    preferred?.personas ||
    frozen?.personas ||
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