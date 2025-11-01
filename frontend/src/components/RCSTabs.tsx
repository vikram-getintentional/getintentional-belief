import { Tabs, Tab } from "@mui/material";
import RCSNetworkGraph from "./RCSGraph";
import RCSOverview from "./RCSOverview";
import RCSCampaignSequences from "./RCSCampaignSequences";
import RCSPersonaInvolvmentVisual from "./RCSPersonaInvolvmentVisual";
import RCSConcernsRecommendations from "./RCSConcernsRecommendations";
import React from "react";

export default function RCSTabs({ rcs, rcsGraph }: { rcs: any; rcsGraph: any }) {
  const [tab, setTab] = React.useState(1); // default to Overview

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
      {tab === 2 && <RCSCampaignSequences rcs={rcs} />}
      {tab === 3 && (
        <RCSPersonaInvolvmentVisual
          // prefer personas from frozen strategy when available
          rcs={rcs}
        />
      )}
      {tab === 4 && (
        <RCSConcernsRecommendations
          tactical={rcs?.tactical_update?.next_concerns || rcs?.tactical_update?.concern_backlog || []}
        />
      )}
    </>
  );
}
