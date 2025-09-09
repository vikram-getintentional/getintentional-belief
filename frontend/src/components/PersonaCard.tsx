import React, { useEffect, useState } from "react";
import { getCardClass } from "./interfaceElements/cardUtils";
import Pill from "./interfaceElements/pillbox";

export type PersonaSuggestion = {
  persona_title: string;
  persona_departments: string[];
  persona_seniority: string[];
  persona_ids: string[];
  max_relevance: number;
  jobs: string[];   // Each is a JSON string: {description, relevance}
  pains: string[];  // Each is a JSON string: {description, relevance}
};

type Props = {
  persona: PersonaSuggestion;
  isSelected: boolean;
  onToggle: () => void;
  productId?: string;
  personaNodesById?: Record<string, any>;
};

const relevanceLabel = (score: number): string => {
  if (score >= 0.8) return "★★★★★";
  if (score >= 0.6) return "★★☆☆☆";
  if (score >= 0.3) return "★☆☆☆☆";
  return "☆☆☆☆☆";
};

const PersonaCard = ({ persona, isSelected, onToggle, productId, personaNodesById }: Props) => {
  const [showAllJobs, setShowAllJobs] = useState(false);
  const [showAllPains, setShowAllPains] = useState(false);
  const [confirm, setConfirm] = useState<{
    open: boolean;
    targetId?: string;
    preview?: { counts: Record<string, number>; nodes: { id: string; node_type: string; label: string }[] } | null;
    loading?: boolean;
    error?: string | null;
  }>({ open: false, preview: null, loading: false, error: null });
  const token = (typeof window !== 'undefined') ? localStorage.getItem('token') : null;
  const [editState, setEditState] = useState<Record<string, { title: string; department: string; linkedin_url: string; relevance_hint: number }>>({});

  if (!persona) return null;

  // Helper to parse job/pain strings
  const parseItem = (str: string) => {
    try {
      return JSON.parse(str);
    } catch {
      return { description: str, relevance: 0 };
    }
  };

  const jobsToShow = showAllJobs ? persona.jobs : persona.jobs.slice(0, 3);
  const painsToShow = showAllPains ? persona.pains : persona.pains.slice(0, 3);

  useEffect(() => {
    const next: Record<string, { title: string; department: string; linkedin_url: string; relevance_hint: number }> = {};
    (persona.persona_ids || []).forEach((pid) => {
      const node = personaNodesById?.[pid] || {};
      next[pid] = {
        title: node.title || persona.persona_title || "",
        department: node.department || (persona.persona_departments?.[0] || ""),
        linkedin_url: node.linkedin_url || "",
        relevance_hint: typeof node.relevance_hint === 'number' ? node.relevance_hint : 0,
      };
    });
    setEditState(next);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [JSON.stringify(persona.persona_ids), personaNodesById, persona.persona_title, JSON.stringify(persona.persona_departments)]);

  const updatePersona = async (id: string, payload: { title?: string; department?: string; linkedin_url?: string; relevance_hint?: number }) => {
    if (!productId) return;
    try {
      const res = await fetch(`http://localhost:8000/graph/persona/${encodeURIComponent(id)}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` },
        body: JSON.stringify({ product_id: productId, ...payload })
      });
      if (!res.ok) throw new Error(await res.text());
    } catch (e) {
      console.error('Failed to update persona', e);
    }
  };

  const previewDelete = async (id: string) => {
    if (!productId) return;
    setConfirm({ open: true, targetId: id, preview: null, loading: true, error: null });
    try {
      const url = new URL(`http://localhost:8000/graph/persona/${encodeURIComponent(id)}/orphan-preview`);
      url.searchParams.set('product_id', productId);
      const res = await fetch(url.toString(), { headers: { Authorization: `Bearer ${token}` } });
      const data = await res.json();
      if (!res.ok) throw new Error(data?.detail || 'Preview failed');
      setConfirm({ open: true, targetId: id, preview: { counts: data.counts || {}, nodes: data.nodes || [] }, loading: false, error: null });
    } catch (e) {
      console.error('Preview failed', e);
      setConfirm(c => ({ ...c, loading: false, error: 'Could not load orphan preview. You can still proceed.' }));
    }
  };

  const performDelete = async () => {
    if (!productId || !confirm.targetId) { setConfirm({ open: false, preview: null, loading: false, error: null }); return; }
    try {
      const url = new URL(`http://localhost:8000/graph/persona/${encodeURIComponent(confirm.targetId)}`);
      url.searchParams.set('product_id', productId);
      if (confirm.preview && (confirm.preview.nodes || []).length > 0) url.searchParams.set('force', 'true');
      const res = await fetch(url.toString(), { method: 'DELETE', headers: { Authorization: `Bearer ${token}` } });
      if (res.status === 409) {
        const data = await res.json().catch(() => ({}));
        const detail = data?.detail || data;
        setConfirm(c => ({ ...c, preview: { counts: detail.counts || {}, nodes: detail.nodes || [] }, loading: false }));
        return;
      }
      if (!res.ok) throw new Error(await res.text());
      setConfirm({ open: false, preview: null, loading: false, error: null });
      // No direct refresh here; parent will refetch on navigation or next view action.
    } catch (e) {
      console.error('Delete failed', e);
    }
  };

  return (
    <div className={getCardClass(persona.max_relevance)}>
      <div className="mb-2">
        <h4 className="font-semibold text-indigo-700">{persona.persona_title}</h4>
        <p className="text-sm text-gray-500">
          {persona.persona_seniority.join(", ")} • {persona.persona_departments.join(", ")}
        </p>
        <p className="text-xs text-indigo-600 font-medium mt-1">
          Relevance Label: {relevanceLabel(persona.max_relevance)}
        </p>
      </div>

      {/* Editable anchors for each concrete persona node */}
      {productId && persona.persona_ids && persona.persona_ids.length > 0 && (
        <div className="mt-3 space-y-2">
          <div className="text-sm font-medium text-slate-700">Anchors & Actions</div>
          <ul className="divide-y divide-slate-200 rounded-md border border-slate-200 bg-slate-50">
            {persona.persona_ids.map((pid) => (
              <li key={pid} className="flex flex-col gap-3 px-3 py-2">
                <div className="flex items-center justify-between">
                  <div className="min-w-0">
                    <div className="truncate text-xs text-slate-500">{pid}</div>
                  </div>
                </div>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
                  <div>
                    <label className="mb-1 block text-xs font-medium text-slate-700">Title</label>
                    <input className="w-full rounded-md border border-slate-300 px-2 py-1 text-sm focus:border-indigo-500 focus:outline-none focus:ring-1 focus:ring-indigo-500" value={editState[pid]?.title || ''} onChange={e => setEditState(s => ({ ...s, [pid]: { ...(s[pid]||{title:'',department:'',linkedin_url:'',relevance_hint:0}), title: e.target.value } }))} />
                  </div>
                  <div>
                    <label className="mb-1 block text-xs font-medium text-slate-700">Department</label>
                    <input className="w-full rounded-md border border-slate-300 px-2 py-1 text-sm focus:border-indigo-500 focus:outline-none focus:ring-1 focus:ring-indigo-500" value={editState[pid]?.department || ''} onChange={e => setEditState(s => ({ ...s, [pid]: { ...(s[pid]||{title:'',department:'',linkedin_url:'',relevance_hint:0}), department: e.target.value } }))} />
                  </div>
                </div>
                <div>
                  <label className="mb-1 block text-xs font-medium text-slate-700">LinkedIn URL</label>
                  <input className="w-full rounded-md border border-slate-300 px-2 py-1 text-sm focus:border-indigo-500 focus:outline-none focus:ring-1 focus:ring-indigo-500" placeholder="https://www.linkedin.com/in/..." value={editState[pid]?.linkedin_url || ''} onChange={e => setEditState(s => ({ ...s, [pid]: { ...(s[pid]||{title:'',department:'',linkedin_url:'',relevance_hint:0}), linkedin_url: e.target.value } }))} />
                </div>
                <div>
                  <label className="mb-1 block text-xs font-medium text-slate-700">Relevance</label>
                  <input type="range" min={0} max={100} value={editState[pid]?.relevance_hint ?? 0} onChange={e => setEditState(s => ({ ...s, [pid]: { ...(s[pid]||{title:'',department:'',linkedin_url:'',relevance_hint:0}), relevance_hint: Number(e.target.value) } }))} className="w-full accent-indigo-600" />
                  <div className="mt-1 h-1 w-full rounded bg-slate-200">
                    <div className="h-1 rounded bg-gradient-to-r from-sky-400 via-indigo-500 to-fuchsia-500" style={{ width: `${editState[pid]?.relevance_hint ?? 0}%` }} />
                  </div>
                </div>
                <div className="mt-2 flex items-center justify-end gap-2">
                  <button
                    type="button"
                    onClick={() => previewDelete(pid)}
                    className="inline-flex items-center gap-1.5 rounded-md border border-red-600 bg-white px-4 py-2 text-sm font-medium text-red-700 shadow-sm transition hover:bg-red-50 focus:outline-none focus:ring-2 focus:ring-red-500 focus:ring-offset-2"
                  >
                    Delete
                  </button>
                  <button
                    type="button"
                    onClick={() => updatePersona(pid, editState[pid] || {})}
                    className="inline-flex items-center gap-1.5 rounded-md bg-indigo-600 px-4 py-2 text-sm font-medium text-white shadow-sm transition hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2 disabled:opacity-60 disabled:cursor-not-allowed"
                  >
                    Save
                  </button>
                </div>
              </li>
            ))}
          </ul>
        </div>
      )}

      {persona.jobs.length > 0 && (
        <div className="space-y-2 mt-3">
          <strong>Jobs:</strong>
          <div className="flex flex-wrap mt-1">
            {jobsToShow.map((jobStr, jdx) => {
              const jobObj = parseItem(jobStr);
              return (
                <Pill key={jdx} label={jobObj.description} score={jobObj.relevance} />
              );
            })}
            {!showAllJobs && persona.jobs.length > 3 && (
              <button
                className="text-xs text-indigo-600 underline ml-2"
                onClick={e => {
                  e.stopPropagation();
                  setShowAllJobs(true);
                }}
              >
                <span className="text-lg font-bold">▼</span>
              </button>
            )}
            {showAllJobs && persona.jobs.length > 3 && (
              <button
                className="text-xs text-indigo-600 underline ml-2"
                onClick={e => {
                  e.stopPropagation();
                  setShowAllJobs(false);
                }}
              >
                <span className="text-lg font-bold">▲</span>
              </button>
            )}
          </div>
        </div>
      )}

      {persona.pains.length > 0 && (
        <div className="mt-3">
          <strong>Pains:</strong>
          <div className="flex flex-wrap mt-1">
            {painsToShow.map((painStr, jdx) => {
              const painObj = parseItem(painStr);
              return (
                <Pill key={jdx} label={painObj.description} score={painObj.relevance} />
              );
            })}
            {!showAllPains && persona.pains.length > 3 && (
              <button
                className="text-xs text-indigo-600 underline ml-2"
                onClick={e => {
                  e.stopPropagation();
                  setShowAllPains(true);
                }}
              >
                <span className="text-lg font-bold">▼</span>
              </button>
            )}
            {showAllPains && persona.pains.length > 3 && (
              <button
                className="text-xs text-indigo-600 underline ml-2"
                onClick={e => {
                  e.stopPropagation();
                  setShowAllPains(false);
                }}
              >
                <span className="text-lg font-bold">▲</span>
              </button>
            )}
          </div>
        </div>
      )}
      {/* Modal for delete confirmation */}
      {confirm.open && (
        <div className="fixed inset-0 z-50 flex items-center justify-center">
          <div className="absolute inset-0 bg-slate-900/50 backdrop-blur-sm" onClick={() => setConfirm({ open: false, preview: null, loading: false, error: null })} />
          <div className="relative mx-4 w-full max-w-2xl overflow-hidden rounded-xl border border-slate-200 bg-white shadow-2xl">
            <div className="border-b border-slate-200 bg-slate-50 px-5 py-3">
              <h4 className="text-lg font-semibold text-slate-800">Delete Persona</h4>
              <p className="text-sm text-slate-600">Review impacted nodes before confirming.</p>
            </div>
            <div className="max-h-[60vh] overflow-auto px-5 py-4">
              {confirm.loading && <div className="text-slate-500">Loading impact…</div>}
              {confirm.error && <div className="mb-3 rounded border border-amber-200 bg-amber-50 px-3 py-2 text-amber-800">{confirm.error}</div>}
              {!confirm.loading && (
                <div className="space-y-3">
                  {Object.values(confirm.preview?.counts || {}).reduce((a: any, b: any) => (a as number) + (b as number), 0) > 0 ? (
                    <div className="rounded-md border border-amber-200 bg-amber-50 p-3 text-amber-900">
                      <div className="mb-1 font-medium">This delete will orphan nodes:</div>
                      <div className="flex flex-wrap gap-2">
                        {Object.entries(confirm.preview?.counts || {}).map(([k, v]) => (
                          <span key={k} className="inline-flex items-center rounded-full bg-slate-100 px-2 py-0.5 text-xs font-medium text-slate-700">{k}: {v as number}</span>
                        ))}
                      </div>
                    </div>
                  ) : (
                    <div className="rounded-md border border-emerald-200 bg-emerald-50 p-3 text-emerald-800">No downstream nodes will be orphaned.</div>
                  )}
                  {(confirm.preview?.nodes || []).length > 0 && (
                    <div>
                      <div className="mb-2 text-sm font-medium text-slate-700">Impacted nodes</div>
                      <ul className="divide-y divide-slate-200 rounded-md border border-slate-200">
                        {confirm.preview?.nodes?.map(n => (
                          <li key={n.id} className="flex items-center justify-between gap-3 px-3 py-2">
                            <div className="min-w-0">
                              <div className="truncate text-sm font-medium text-slate-800">{n.label}</div>
                              <div className="truncate text-xs text-slate-500">{n.id}</div>
                            </div>
                            <span className="inline-flex items-center rounded-full bg-slate-100 px-2 py-0.5 text-xs font-medium text-slate-700">{n.node_type}</span>
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}
                </div>
              )}
            </div>
            <div className="flex items-center justify-end gap-2 border-t border-slate-200 bg-slate-50 px-5 py-3">
              <button onClick={() => setConfirm({ open: false, preview: null, loading: false, error: null })} className="rounded-md border border-slate-300 bg-white px-3 py-2 text-sm font-medium text-slate-700 shadow-sm hover:bg-slate-50">Cancel</button>
              <button onClick={performDelete} className="rounded-md bg-red-700 px-3 py-2 text-sm font-medium text-white shadow-sm hover:bg-red-800">Delete anyway</button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default PersonaCard;
