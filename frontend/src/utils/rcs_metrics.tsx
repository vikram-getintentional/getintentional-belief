// ============================
// utils/rcsMetrics.ts
// ============================

export type PersonaMetric = {
  id?: string;
  label: string;
  involvement?: number; // raw X
  activation?: number;  // raw Y
};

export type ZPersonaMetric = PersonaMetric & {
  x: number;  // raw involvement
  y: number;  // raw activation
  zx: number; // z involvement
  zy: number; // z activation
};

export type Role =
  | "Potential Champion"
  | "Blocker"
  | "Operator"
  | "Passive Influencer";

const safeMean = (arr: number[]) => (arr.length ? arr.reduce((a,b)=>a+b,0)/arr.length : 0);
const safeStd  = (arr: number[], m: number) =>
  arr.length ? Math.sqrt(arr.reduce((a,b)=>a+(b-m)*(b-m),0)/arr.length) || 1 : 1;

/**
 * Compute z-scores for involvement/activation and return enriched records.
 * Keeps original raw x/y for tooltips and other consumers.
 */
export function zNormalizePersonas(personas: PersonaMetric[]): ZPersonaMetric[] {
  const xs = personas.map(p => p.involvement ?? 0);
  const ys = personas.map(p => p.activation ?? 0);

  const mx = safeMean(xs), sx = safeStd(xs, mx) || 1;
  const my = safeMean(ys), sy = safeStd(ys, my) || 1;

  return personas.map(p => {
    const x = p.involvement ?? 0;
    const y = p.activation ?? 0;
    return {
      ...p,
      x, y,
      zx: (x - mx) / sx,
      zy: (y - my) / sy,
    };
  });
}

/**
 * Role classification (choose 'raw' or 'z' space).
 * - raw: thresholds in [0..1] by default (0.5/0.5)
 * - z: thresholds around 0 (0/0)
 */
export function classifyRole(
  p: ZPersonaMetric,
  opts?: {
    mode?: "raw" | "z";
    invCut?: number; // raw cut
    actCut?: number; // raw cut
    zxCut?: number;  // z cut
    zyCut?: number;  // z cut
  }
): Role {
  const {
    mode = "z",
    invCut = 0.5,
    actCut = 0.5,
    zxCut = 0,
    zyCut = 0,
  } = opts || {};

  const invHigh = mode === "z" ? p.zx >= zxCut : (p.x >= invCut);
  const actHigh = mode === "z" ? p.zy >= zyCut : (p.y >= actCut);

  if (invHigh && actHigh) return "Potential Champion";
  if (invHigh && !actHigh) return "Blocker";
  if (!invHigh && actHigh) return "Operator";
  return "Passive Influencer";
}

/** Bucket personas by role using the same classifier. */
export function bucketByRole(
  personas: PersonaMetric[],
  opts?: Parameters<typeof classifyRole>[1]
): Record<Role, ZPersonaMetric[]> {
  const zs = zNormalizePersonas(personas);
  const out: Record<Role, ZPersonaMetric[]> = {
    "Potential Champion": [],
    "Blocker": [],
    "Operator": [],
    "Passive Influencer": [],
  };
  zs.forEach(p => out[classifyRole(p, opts)].push(p));
  return out;
}

/** Helper to compute nice domains */
export function domainOf(
  vals: number[],
  pad = 0.05
): [number, number] {
  if (!vals.length) return [0, 1];
  const mn = Math.min(...vals), mx = Math.max(...vals);
  if (mn === mx) return [mn - 1, mx + 1];
  const span = mx - mn;
  return [mn - span*pad, mx + span*pad];
}
