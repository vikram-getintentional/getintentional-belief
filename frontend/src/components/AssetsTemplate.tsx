import React, { useEffect, useState } from "react";
import { Card, CardContent, CardHeader, Typography, TextField, Button, Box, Chip, Grid, Divider, Autocomplete, Slider } from "@mui/material";

interface Asset {
  id: string;
  name: string;
  format: string;
  cost?: number;
  est_production_cost?: number;
  time_to_produce?: string;
  best_used_for?: string;
  industries?: string[];
  personas?: string[];
  distribution_channels?: string[];
  notes?: string;
}

interface AssetsTemplateProps {
  productID: string;
}
const formatOptions = ["Long-form Text", "Short-form Text", "Infographic", "Video", "Podcast", "Webinar", "Template", "Checklist", "Case Study", "Whitepaper", "Ebook", "Blog Post", "Social Media Post", "Ad Unit", "Poster", "Email", "Landing Page", "Other"];
const deliveryOptions = ["Broadcast", "Personalized"];
const effectivenessRanges = [
  { min: 0, max: 30, label: "Wild Spray" },
  { min: 31, max: 50, label: "Hope and Pray" },
  { min: 51, max: 60, label: "Better than guessing" },
  { min: 61, max: 90, label: "Reliably Good" },
  { min: 91, max: 100, label: "Hyper Conversion" },
];

const initialForm = {
  name: "",
  format: "",
  delivery: "",
  effectiveness: 50,
  cost: "",
  est_production_cost: "",
  time_to_produce: "",
  best_used_for: "",
  industries: "",
  personas: "",
  distribution_channels: "",
  notes: "",
};

const getEffectivenessLabel = (value: number) => {
  const found = effectivenessRanges.find(range => value >= range.min && value <= range.max);
  return found ? found.label : "";
};



const AssetsTemplate = ({ productID }: AssetsTemplateProps) => {
  const [assets, setAssets] = useState<Asset[]>([]);
  const [form, setForm] = useState(initialForm);

  useEffect(() => {
    fetch(`/api/get-asset-templates/${productID}`)
      .then(res => res.json())
      .then(data => setAssets(data));
  }, [productID]);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setForm({ ...form, [e.target.name]: e.target.value });
  };

  const handleAddAsset = async () => {
    const payload = {
      ...form,
      cost: form.cost ? Number(form.cost) : undefined,
      est_production_cost: form.est_production_cost ? Number(form.est_production_cost) : undefined,
      industries: form.industries ? form.industries.split(",").map(s => s.trim()) : [],
      personas: form.personas ? form.personas.split(",").map(s => s.trim()) : [],
      distribution_channels: form.distribution_channels ? form.distribution_channels.split(",").map(s => s.trim()) : [],
    };
    await fetch(`/api/add-new-asset-template/${productID}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    // Re-fetch asset templates
    const res = await fetch(`/api/get-asset-templates/${productID}`);
    setAssets(await res.json());
    setForm(initialForm);
  };

  return (
    <Box>
      <Box mb={4}>
        <Typography variant="h6" gutterBottom>Add New Asset Type</Typography>
        <Grid container spacing={2}>
          <Grid size={12}>
            <TextField label="Asset Type" name="name" value={form.name} onChange={handleChange} fullWidth />
          </Grid>
          <Grid size={{ xs: 12, sm: 6 }}>
            <Autocomplete
                options={formatOptions}
                value={form.format}
                onChange={(_, newValue) => setForm({ ...form, format: newValue || "" })}
                renderInput={(params) => (
                    <TextField {...params} label="Format" fullWidth />
                )}
            />
          </Grid>
          <Grid size={{ xs: 12, sm: 6 }}>
            <Autocomplete
                options={deliveryOptions}
                value={form.delivery}
                onChange={(_, newValue) => setForm({ ...form, delivery: newValue || "" })}
                renderInput={(params) => (
                    <TextField {...params} label="Delivery" fullWidth />
                )}
            />
          </Grid>
          

        </Grid>

        <Divider sx={{ mt: 1.5, mb: 1.5 }} />
        <Typography variant="body1" gutterBottom xs={12}>Production Estimates</Typography>
        <Grid container spacing={2}>
          <Grid size={{ xs: 12, sm: 6 }}>
            <TextField label="Typical Production Cost (USD)" name="cost" value={form.cost} onChange={handleChange} fullWidth type="number" />
          </Grid>
          <Grid size={{ xs: 12, sm: 6 }}>
            <TextField label="Time to Produce (Days)" name="time_to_produce" value={form.time_to_produce} onChange={handleChange} fullWidth type="number" />
          </Grid>
        </Grid>
        <Divider sx={{ mt: 1.5, mb: 1.5 }} />
        <Typography variant="body1" gutterBottom xs={12}>Target & Expectations</Typography>
        <Grid container spacing={2}>
          <Grid size={{ xs: 6, sm: 4 }}>
            <TextField label="Typical Industries" name="industries" value={form.industries} onChange={handleChange} fullWidth />
          </Grid>
          <Grid size={{ xs: 6, sm: 4 }}>
            <TextField label="Target Personas" name="personas" value={form.personas} onChange={handleChange} fullWidth />
          </Grid>
          <Grid size={{ xs: 6, sm: 4 }}>
            <TextField label="Funnel Stage" name="funnel_stage" value={form.funnel_stage} onChange={handleChange} fullWidth />
          </Grid>
          <Grid size={{ xs: 6, sm: 4 }}>
            <Typography gutterBottom>Effectiveness</Typography>
            <TextField
                label="Effectiveness"
                value={getEffectivenessLabel(form.effectiveness)}
                InputProps={{ readOnly: true }}
                fullWidth
                sx={{ mb: 1 }}
            />
            <Slider
                value={form.effectiveness}
                onChange={(_, newValue) => setForm({ ...form, effectiveness: newValue as number })}
                step={1}
                min={0}
                max={100}
                valueLabelDisplay="auto"
                marks={[]} // No marks
            />
            </Grid>
        </Grid>
        <Divider sx={{ mt: 1.5, mb: 1.5 }} />
        <Typography variant="body1" gutterBottom xs={12}>Distribution & Channel</Typography>
        <Grid container spacing={2}>
          <Grid size={{ xs: 12, sm: 6 }}>
            <TextField label="Channels" name="distribution_channels" value={form.distribution_channels} onChange={handleChange} fullWidth />
          </Grid>
          <Grid size={{ xs: 12, sm: 6 }}>
            <TextField label="Works with Assets" name="other_assets" value={form.other_assets} onChange={handleChange} fullWidth />
          </Grid>
          <Grid size={{ xs: 12 }}>
            <TextField label="Notes" name="notes" value={form.notes} onChange={handleChange} fullWidth multiline minRows={2} />
          </Grid>
          <Grid size={{ xs: 12 }}>
            <Button variant="contained" color="primary" onClick={handleAddAsset}>Add Asset Type</Button>
          </Grid>
        </Grid>
      </Box>

      <Typography variant="h6" gutterBottom>Existing Asset Types</Typography>
      <Grid container spacing={2}>
        {assets.map(asset => (
          <Grid size={{ xs: 12, sm: 6, md: 4 }} key={asset.id}>
            <Card variant="outlined" sx={{ height: "100%", display: "flex", flexDirection: "column" }}>
              <CardHeader title={asset.name} subheader={asset.format} sx={{ bgcolor: "#f5f5f5" }} />
              <CardContent sx={{ flexGrow: 1 }}>
                {asset.cost !== undefined && (
                  <Typography variant="body2"><b>Cost:</b> {asset.cost}</Typography>
                )}
                {asset.est_production_cost !== undefined && (
                  <Typography variant="body2"><b>Estimated Production Cost:</b> {asset.est_production_cost}</Typography>
                )}
                {asset.time_to_produce && (
                  <Typography variant="body2"><b>Time to Produce:</b> {asset.time_to_produce}</Typography>
                )}
                {asset.best_used_for && (
                  <Typography variant="body2"><b>Best Used For:</b> {asset.best_used_for}</Typography>
                )}
                {asset.industries && asset.industries.length > 0 && (
                  <Typography variant="body2"><b>Industries:</b> {asset.industries.map((ind, idx) => <Chip key={idx} label={ind} size="small" sx={{ mr: 0.5 }} />)}</Typography>
                )}
                {asset.personas && asset.personas.length > 0 && (
                  <Typography variant="body2"><b>Personas:</b> {asset.personas.map((p, idx) => <Chip key={idx} label={p} size="small" sx={{ mr: 0.5 }} />)}</Typography>
                )}
                {asset.distribution_channels && asset.distribution_channels.length > 0 && (
                  <Typography variant="body2"><b>Distribution Channels:</b> {asset.distribution_channels.map((ch, idx) => <Chip key={idx} label={ch} size="small" sx={{ mr: 0.5 }} />)}</Typography>
                )}
                {asset.notes && (
                  <Typography variant="body2" color="textSecondary" sx={{ mt: 1 }}>
                    <b>Notes:</b> {asset.notes}
                  </Typography>
                )}
              </CardContent>
            </Card>
          </Grid>
        ))}
      </Grid>
    </Box>
  );
};

export default AssetsTemplate;
