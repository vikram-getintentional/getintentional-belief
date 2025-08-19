import React from "react";

type Archetype = {
  archetype_id: string;
  label?: string;
  industry?: string;
  revenue_range?: string;
  employee_range?: string;
  funding_stage?: string;
  geography?: string;
};

type SuggestedEvent = {
  event_id: string;
  trigger_event: string;
  fit: number;
};

type RCSScaffold = {
  rcs_id: string;
  stage: "archetype" | "event" | "engagement";
  archetype: Archetype;
  insights: {
    pains: { pain_id: string; text: string }[];
    jobs: { job_id: string; text: string }[];
    personas: { title: string; department: string; seniority: string }[];
    prevalent_triggers: {
      pain_trigger_id: string;
      attribute: string;
      prevalence: number;
    }[];
  };
  event: {
    suggested_events: SuggestedEvent[];
    suggested_engagements: { engagement_id: string; name: string; fit: number }[];
  };
  engagement: any;
  plan: { sequence: any[] };
  metrics: { leading: any[]; lagging: any[] };
  notes: string[];
};

function clsx(...xs: (string | false | null | undefined)[]) {
  return xs.filter(Boolean).join(" ");
}

export default function RCSArchetypePicker({
  selectedProductId,
  token,
}: {
  selectedProductId: string;
  token: string;
}) {
  const [archetypes, setArchetypes] = React.useState<Archetype[]>([]);
  const [selectedArch, setSelectedArch] = React.useState<string>("");
  const [rcs, setRcs] = React.useState<RCSScaffold | null>(null);
  const [loadingList, setLoadingList] = React.useState(false);
  const [loadingRCS, setLoadingRCS] = React.useState(false);
  const [error, setError] = React.useState<string | null>(null);
  const [zmotEvents, setZmotEvents] = React.useState<{ zmot_event_id: string; zmot_event: string; occurance: number }[]>([]);
  const [selectedZmotEvent, setSelectedZmotEvent] = React.useState<string>("");
  const [selectedEngagementMeta, setSelectedEngagementMeta] = React.useState<any>(null);

  // Load archetypes when product changes
  React.useEffect(() => {
    if (!selectedProductId) return;
    let alive = true;
    setLoadingList(true);
    setError(null);
    setArchetypes([]);
    setSelectedArch("");
    setZmotEvents([]);
    setSelectedZmotEvent("");
    setRcs(null);

    (async () => {
      try {
        const res = await fetch(`http://localhost:8000/get-archetypes/${selectedProductId}`, {
          headers: { Authorization: `Bearer ${token}` }
        });
        if (!res.ok) throw new Error(`Failed to load archetypes (${res.status})`);
        const data = await res.json();
        if (!alive) return;
        setArchetypes(data.archetypes || []);
      } catch (e: any) {
        if (alive) setError(e?.message || "Unable to load archetypes");
      } finally {
        if (alive) setLoadingList(false);
      }
    })();
    return () => { alive = false; };
  }, [selectedProductId, token]);

  // Load ZMOT events when archetype changes
  React.useEffect(() => {
    if (!selectedArch) {
      setZmotEvents([]);
      setSelectedZmotEvent("");
      return;
    }
    let alive = true;
    setZmotEvents([]);
    setSelectedZmotEvent("");
    (async () => {
      try {
        const res = await fetch(
          `http://localhost:8000/get-zmots-for-archetype/${selectedProductId}?archetype_id=${selectedArch}`,
          { headers: { Authorization: `Bearer ${token}` } }
        );
        if (!res.ok) throw new Error("Failed to load ZMOT events");
        const data = await res.json();
        if (!alive) return;
        setZmotEvents(Array.isArray(data) ? data : data.zmots || []);
      } catch (err) {
        if (alive) setZmotEvents([]);
      }
    })();
    return () => { alive = false; };
  }, [selectedArch, selectedProductId, token]);

  // Fetch RCS scaffold on any relevant change
  React.useEffect(() => {
    if (!selectedArch) {
      setRcs(null);
      return;
    }
    setLoadingRCS(true);
    setError(null);
    (async () => {
      try {
        const payload: any = { archetype_id: selectedArch };
        if (selectedZmotEvent) payload.zmot_event_id = selectedZmotEvent;
        if (selectedEngagementMeta) payload.engagement_meta = selectedEngagementMeta;
        const res = await fetch(`http://localhost:8000/get-reverse-case-study/${selectedProductId}`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            Authorization: `Bearer ${token}`,
          },
          body: JSON.stringify(payload),
        });
        if (!res.ok) throw new Error(`Failed to get reverse case study (${res.status})`);
        const data = await res.json();
        setRcs(data);
      } catch (e: any) {
        setError(e.message || "Failed to load reverse case study");
        setRcs(null);
      } finally {
        setLoadingRCS(false);
      }
    })();
  }, [selectedArch, selectedZmotEvent, selectedEngagementMeta, selectedProductId, token]);

  const current = archetypes.find(a => a.archetype_id === selectedArch);
  const title = current
    ? current.label ||
      [current.industry, current.funding_stage, current.geography].filter(Boolean).join(" | ")
    : "";

  return (
    <div className="w-full max-w-6xl mx-auto p-6">
      <h1 className="text-2xl font-semibold mb-4">Reverse Case Studies Generator (RCS)</h1>

      {/* Archetype Picker */}
      <div className="bg-white border rounded-2xl p-4 shadow-sm">
        <label className="block text-sm font-medium mb-2">Select archetype</label>
        <div className="flex items-center gap-3">
          <select
            className="w-full border rounded-xl px-3 py-2"
            value={selectedArch}
            onChange={e => setSelectedArch(e.target.value)}
            disabled={loadingList}
          >
            <option value="" disabled>
              {loadingList ? "Loading…" : "Choose an archetype…"}
            </option>
            {archetypes.map(a => (
              <option key={a.archetype_id} value={a.archetype_id}>
                {[
                  a.label,
                  [a.industry, a.funding_stage, a.geography].filter(Boolean).join(" | ")
                ].filter(Boolean)[0]}
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* ZMOT Picker */}
      {zmotEvents.length > 0 && (
        <div className="mt-4">
          <label className="block text-sm font-medium mb-2">Select ZMOT Event</label>
          <select
            className="w-full border rounded-xl px-3 py-2"
            value={selectedZmotEvent}
            onChange={e => setSelectedZmotEvent(e.target.value)}
          >
            <option value="" disabled>Choose a ZMOT event…</option>
            {zmotEvents.map(ev => (
              <option key={ev.zmot_event_id} value={ev.zmot_event_id}>{ev.zmot_event}</option>
            ))}
          </select>
        </div>
      )}

      {/* Errors */}
      {error && (
        <div className="mt-4 bg-red-50 text-red-700 border border-red-200 rounded-xl p-3">
          {error}
        </div>
      )}

      {/* RCS Scaffold */}
      {rcs && (
        <div className="mt-6 space-y-6">
            <div className="bg-white border rounded-2xl p-5 shadow-sm">
            <h2 className="text-xl font-semibold mb-2">Backend Output</h2>
            <pre className="text-xs bg-gray-50 p-4 rounded-xl overflow-x-auto">
                {JSON.stringify(rcs, null, 2)}
            </pre>
            </div>
        </div>
        )}

      {/* Empty state */}
      {!rcs && !loadingRCS && !error && (
        <div className="mt-6 text-sm text-gray-500">
          Select an archetype to generate the scaffold.
        </div>
      )}
    </div>
  );
}

function Card({ title, items }: { title: string; items: string[] }) {
  return (
    <div className="bg-white border rounded-2xl p-5 shadow-sm">
      <h3 className="font-medium mb-3">{title}</h3>
      <ul className="space-y-2">
        {items.slice(0, 6).map((t, i) => (
          <li key={i} className="text-sm">{t}</li>
        ))}
        {!items.length && <li className="text-sm text-gray-500">None</li>}
      </ul>
    </div>
  );
}

function PersonasCard({ personas }: { personas: { title: string; department: string; seniority: string }[] }) {
  return (
    <div className="bg-white border rounded-2xl p-5 shadow-sm">
      <h3 className="font-medium mb-3">Personas</h3>
      <ul className="space-y-2">
        {personas.slice(0, 6).map((p, i) => (
          <li key={i} className="text-sm">
            <span className="font-medium">{p.title || "—"}</span>
            <span className="text-gray-500"> • {p.department || "—"} • {p.seniority || "—"}</span>
          </li>
        ))}
        {!personas.length && <li className="text-sm text-gray-500">None</li>}
      </ul>
    </div>
  );
}