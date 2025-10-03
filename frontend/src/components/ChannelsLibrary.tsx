import React from "react";
import { Card, CardContent, CardHeader, Typography, Grid } from "@mui/material";

type Channel = {
  id: string | number;
  name: string;
  type?: string;
  reach_score?: number | string;
  cost_index?: number | string;
};

interface ChannelsLibraryProps {
  channels: Channel[];
}

const ChannelsLibrary: React.FC<ChannelsLibraryProps> = ({ channels }) => (
  <Grid container spacing={2}>
    {channels.length === 0 && (
      <Grid size={12}>
        <Typography color="textSecondary">No channels found.</Typography>
      </Grid>
    )}
    {channels.map(channel => (
      <Grid size={12} key={channel.id}>
        <Card variant="outlined" sx={{ height: "100%" }}>
          <CardHeader
            title={
              <Typography variant="h6" gutterBottom>
                {channel.name}
              </Typography>
            }
            subheader={
              <Typography variant="body1" gutterBottom>
                {channel.type}
              </Typography>
            }
            sx={{ color: "indigo" }}
          />
          <CardContent>
            {channel.reach_score !== undefined && (
              <Typography variant="body2" gutterBottom>
                <b>Reach:</b> {channel.reach_score}
              </Typography>
            )}
            {channel.cost_index !== undefined && (
              <Typography variant="body2" gutterBottom>
                <b>Cost Index:</b> {channel.cost_index}
              </Typography>
            )}
          </CardContent>
        </Card>
      </Grid>
    ))}
  </Grid>
);

export default ChannelsLibrary;