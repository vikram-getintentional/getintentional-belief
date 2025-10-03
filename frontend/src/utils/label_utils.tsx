export function buildAssetsByPersona(rcs: any) {
  const out: Record<string, Record<string, any[]>> = {};
  (rcs?.concerns_by_persona || []).forEach((row: any) => {
    const pid = row.persona;
    const assets = row.assets || {};
    if (!out[pid]) out[pid] = {};
    Object.keys(assets).forEach((cid) => {
      out[pid][cid] = assets[cid];
    });
  });
  return out;
}

export function buildLabelMaps(rcs: any) {
  const personaLabelById: Record<string,string> = {};
  const concernLabelById: Record<string,string> = {};

  (rcs?.concerns_flat || []).forEach((c: any) => {
    personaLabelById[c.persona_id] = c.persona_label || c.persona_id;
    concernLabelById[c.concern_id] = c.concern_label || c.concern_id;
  });

  // fallback: try all_personas
  (rcs?.all_personas || []).forEach((p: any) => {
    personaLabelById[p.id] = p.label || p.id;
  });

  return { personaLabelById, concernLabelById };
}