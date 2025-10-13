import { Box, FormControl, Grid, InputLabel, MenuItem, Select } from "@mui/material";
import React from "react";
import { useEffect, useState } from "react";

type Family = "industry" | "revenue" | "employees" | "funding" | "geography";
interface ICPOption {
    id: string;
    label: string;
    family: Family;
}

interface SelectedNode {
    id: string;
}

interface RCSConfiguratorProps {
  selectedProductId?: string;
  token: string;
  selectedNodes?: SelectedNode[];   // <- optional
  onRcsFetched: (data: any) => void;
}

const RCSConfigurator: React.FC<RCSConfiguratorProps> = ({
  selectedProductId,
  token,
  selectedNodes = [],             // <- default to []
  onRcsFetched
}) => {

    // ICP selectors
    const [icpOptions, setIcpOptions] = React.useState<Record<Family, ICPOption[]>>({
        industry: [],
        revenue: [],
        employees: [],
        funding: [],
        geography: [],
    });

    const [icpSelected, setIcpSelected] =
    React.useState<Record<Family, string | "">>({
        industry: "",
        revenue: "",
        employees: "",
        funding: "",
        geography: "",
    });
    const [err, setErr] = React.useState<string | null>(null);
    const [loading, setLoading] = React.useState<boolean>(false);
    const [shouldFetch, setShouldFetch] = useState(true);
    const [zmotSuggestions, setZmotSuggestions] = React.useState<{ id: string; label: string }[]>([]);
    const [selectedZmot, setSelectedZmot] = React.useState<string | "">("");
    const handleChange = (fam: Family, value: string) => {
        setIcpSelected(s => ({ ...s, [fam]: value }));
        setShouldFetch(true);
    };


    useEffect(() => {
        if (!selectedProductId || !token) return;
        let alive = true;
        (async () => {
            setLoading(true); setErr(null);
            try {
            const res = await fetch(`http://localhost:8000/get-icp-archetype-options/${selectedProductId}`, {
                headers: { Authorization: `Bearer ${token}` }
            });
            if (!res.ok) throw new Error(`Failed to load ICP attributes (${res.status})`);
            const data = await res.json();

            const keyMap: Record<string, Family> = {
                industry: "industry",
                revenue_range: "revenue",
                employee_range: "employees",
                funding_stage: "funding",
                geography: "geography",
            };

            const next: Record<Family, ICPOption[]> = {
                industry: [], revenue: [], employees: [], funding: [], geography: [],
            };

            Object.entries(data || {}).forEach(([backendKey, arr]) => {
                const fam = keyMap[backendKey];
                if (!fam || !Array.isArray(arr)) return;
                next[fam] = arr.map((opt: any) => ({
                id: opt.id,
                label: opt.label ?? opt.name ?? opt.id,
                family: fam,
                }));
            });

            if (alive) setIcpOptions(next);
            } catch (e: any) {
            if (alive) setErr(e?.message || "Unable to load ICP attributes");
            if (alive) setIcpOptions({ industry: [], revenue: [], employees: [], funding: [], geography: [] });
            } finally {
            if (alive) setLoading(false);
            }
        })();
        return () => { alive = false; };
        }, [selectedProductId, token]);

    useEffect(() => {
        // only mark dirty if node ids actually changed
        setShouldFetch(true);
    }, [JSON.stringify(selectedNodes.map(n => n.id))]);

    useEffect(() => {
        // fetch ZMOT suggestions whenever ICP selection changes
        const attrIds = Object.values(icpSelected).filter(Boolean) as string[];
        if (!selectedProductId) return;
        let alive = true;
        (async () => {
          if (attrIds.length === 0) {
            if (alive) setZmotSuggestions([]);
            return;
          }
          try {
            console.log("Fetching ZMOT suggestions for attributes", attrIds);
            const res = await fetch(`http://localhost:8000/get-zmots-for-attributes/${selectedProductId}`, {
              method: "POST",
              headers: { "Content-Type": "application/json", Authorization: `Bearer ${token}` },
              body: JSON.stringify({ attribute_ids: attrIds })
            });
            if (!res.ok) throw new Error("Failed to load ZMOT suggestions");
            const data = await res.json();
            console.log("Raw ZMOT suggestions data", data);
            // backend may return either { zmots: [...] } or an array directly
            const rawZmots = Array.isArray(data) ? data : data?.zmots ?? [];
            const zmots = (Array.isArray(rawZmots) ? rawZmots : []).map((z: any) => ({
              id: String(z.zmot_event_id ?? z.id ?? z.event_id ?? ""),
              label: String(z.zmot_label ?? z.zmot_event ?? z.label ?? z.name ?? z.id ?? ""),
            }));
            if (!alive) return;
            setZmotSuggestions(zmots);
            console.log("Fetched ZMOT suggestions", zmots);
          } catch {
            if (alive) setZmotSuggestions([]);
          }
        })();
        return () => { alive = false; };
      }, [icpSelected, selectedProductId, token]);

    useEffect(() => {
        if (!selectedProductId || !shouldFetch) return; // 🚫 skip if nothing new

        console.log("RCS fetch triggered", { 
            selectedProductId,
            shouldFetch,
            icpSelected, 
            selectedNodes 
        });
 
        const attrIds = Object.values(icpSelected).filter(Boolean);
        const selIds = Array.isArray(selectedNodes) ? selectedNodes.map(n => n.id) : [];
 
        const controller = new AbortController();
 
        (async () => {
            try {
            const payload: any = { attribute_ids: attrIds, selected_node_ids: selIds };
            if (selectedZmot) payload.zmot_event_id = selectedZmot;
            console.log("RCS payload to backend router", payload);
 
            const res = await fetch(`http://localhost:8000/get-reverse-case-study/${selectedProductId}`, {
                method: "POST",
                headers: {
                "Content-Type": "application/json",
                Authorization: `Bearer ${token}`,
                },
                body: JSON.stringify(payload),
                signal: controller.signal,
            });
 
            if (!res.ok) {
                console.warn("RCS fetch failed:", res.statusText);
                return;
            }
            const data = await res.json();
            console.log("RCS data response", data);
            onRcsFetched?.(data);
            } catch (e: unknown) {
            if (!(e instanceof Error && e.name === "AbortError")) console.warn("RCS fetch failed:", e);
            } finally {
            setShouldFetch(false);
            }
        })();
 
        return () => controller.abort();
        }, [shouldFetch, selectedProductId, token]);


    // render only pickers (no tabs)

    const familyPicker = (fam: Family, label: string) => {
        const opts = icpOptions?.[fam] ?? []; // <-- guard
        return (
            <FormControl sx={{ minWidth: 220 }}>
                <InputLabel id={`${fam}-label`}>{label}</InputLabel>
                <Select
                    labelId={`${fam}-label`}
                    value={icpSelected[fam]}
                    label={label}
                    onChange={e => handleChange(fam, e.target.value as string)}
                >
                    <MenuItem value=""><em>None</em></MenuItem>
                    {opts.map(opt => (
                        <MenuItem key={opt.id} value={opt.id}>{opt.label}</MenuItem>
                    ))}
                </Select>
            </FormControl>
        );
    };



    return (
        <Box sx={{ display: "flex", gap: 2, flexWrap: "wrap", mb: 2 }}>
            <Grid size={12} container spacing={2}>
                {familyPicker("industry", "Industry")}
                {familyPicker("revenue", "Revenue")}
                {familyPicker("employees", "Employees")}
                {familyPicker("funding", "Funding")}
                {familyPicker("geography", "Geography")}
            </Grid>
            <Grid size={12}>
                {/* ZMOT picker: chooses the most relevant ZMOT for selected attributes */}
                <FormControl sx={{ minWidth: 220, mt: 2 }}>
                    <InputLabel id="zmot-label">ZMOT</InputLabel>
                    <Select
                    labelId="zmot-label"
                    value={selectedZmot}
                    label="ZMOT"
                    onChange={(e) => {
                        const val = e.target.value as string;
                        setSelectedZmot(val);
                        // mark dirty so RCS fetch includes selected zmot
                        setShouldFetch(true);
                    }}
                    >
                    <MenuItem value=""><em>None</em></MenuItem>
                    {zmotSuggestions.map(z => (
                        <MenuItem key={z.id} value={z.id}>{z.label}</MenuItem>
                    ))}
                    </Select>
                </FormControl>
            </Grid>
        </Box>
    );
}

export default RCSConfigurator;


