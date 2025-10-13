import { Tabs, Tab, Box, Typography } from "@mui/material";
import RCSNetworkGraph from "./RCSGraph";
import RCSOverview from "./RCSOverview";
import RCSCampaignSequences from "./RCSCampaignSequences";
import RCSPersonaInvolvmentVisual from "./RCSPersonaInvolvmentVisual";
import RCSConcernsRecommendations from "./RCSConcernsRecommendations";
import { normalizeRcsSequences, normalizeRcsStrategy } from "../utils/rcs_normalize";
import { buildAssetsByPersona, buildLabelMaps } from "../utils/label_utils";
import React from "react";

export default function RCSTabs({ rcs, rcsGraph }: { rcs: any; rcsGraph: any }) {
  const [tab, setTab] = React.useState(0);
  const sequences = normalizeRcsSequences(rcs);
  const strategy = normalizeRcsStrategy(rcs);
  const labelMap = buildLabelMaps(rcs);
  const assetsByPersona = buildAssetsByPersona(rcs);

  return (
    <>
      <Tabs value={tab} onChange={(_, v) => setTab(v)} sx={{ mb: 2 }}>
        <Tab label="Graph" />
        <Tab label="Overview" />
        <Tab label="Campaign Sequences" />
        <Tab label="Persona Involvement" />
        <Tab label="Recommended Plays" />
      </Tabs>

      {tab === 0 && <RCSNetworkGraph graphData={rcsGraph} />}
      {tab === 1 && <RCSOverview rcs={rcs} />}
      {tab === 2 && <RCSCampaignSequences sequences={sequences} labelMap={labelMap} assetsByPersona={assetsByPersona} strategy={strategy} />}
      {tab === 3 && <RCSPersonaInvolvmentVisual personas={strategy.personas || []} />}
      {tab === 4 && <RCSConcernsRecommendations concerns={strategy.nextConcerns || []} />}
    </>
  );
}
