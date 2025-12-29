import React, { useCallback, useEffect, useState } from "react";
import {
  Alert,
  Box,
  Button,
  Card,
  CardContent,
  CardHeader,
  Chip,
  CircularProgress,
  Divider,
  FormControl,
  InputLabel,
  MenuItem,
  Paper,
  Select,
  Stack,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  TextField,
  Toolbar,
  Typography,
} from "@mui/material";
import type {
  ThesisSegmentFeature,
  ThesisSegment,
  SegmentThesisSummary,
  ThesisBuilderCall,
  ThesisAttributeType,
  SegmentRuleCondition,
} from "../types/apiContracts";

const formatPercent = (value?: number | null) => {
  if (value === null || value === undefined) return "—";
  return `${(value * 100).toFixed(1)}%`;
};

type FeatureFormState = {
  name: string;
  key: string;
  type: ThesisAttributeType;
};

type SegmentBuilderForm = {
  name: string;
  description: string;
  priorWeight: number;
  conditions: SegmentRuleCondition[];
};

type ConditionDraft = {
  feature_key: string;
  valuesInput: string;
};

const initialFeatureForm: FeatureFormState = {
  name: "",
  key: "",
  type: "numeric",
};

const initialSegmentForm: SegmentBuilderForm = {
  name: "",
  description: "",
  priorWeight: 1,
  conditions: [],
};

const ThesisBuilderPage: React.FC = () => {
  const token =
    typeof window !== "undefined" ? localStorage.getItem("token") : null;
  const [products, setProducts] = useState<Array<{ id: string; name: string }>>([]);
  const [selectedProductId, setSelectedProductId] = useState<string>("");
  const [features, setFeatures] = useState<ThesisSegmentFeature[]>([]);
  const [segments, setSegments] = useState<ThesisSegment[]>([]);
  const [selectedSegmentId, setSelectedSegmentId] = useState<string | null>(null);
  const [segmentSummary, setSegmentSummary] = useState<SegmentThesisSummary | null>(null);
  const [calls, setCalls] = useState<ThesisBuilderCall[]>([]);
  const [featureForm, setFeatureForm] = useState<FeatureFormState>(initialFeatureForm);
  const [segmentForm, setSegmentForm] = useState<SegmentBuilderForm>(initialSegmentForm);
  const [conditionDraft, setConditionDraft] = useState<ConditionDraft>({
    feature_key: "",
    valuesInput: "",
  });
  const [alert, setAlert] = useState<{ type: "error" | "success"; message: string } | null>(null);
  const [loadingSummary, setLoadingSummary] = useState(false);
  const [loadingCalls, setLoadingCalls] = useState(false);
  const [callPayloadText, setCallPayloadText] = useState("");
  const [ingestingCall, setIngestingCall] = useState(false);
  const [transcriptText, setTranscriptText] = useState("");
  const [transcriptAccountId, setTranscriptAccountId] = useState("");
  const [transcriptExternalId, setTranscriptExternalId] = useState("");
  const [savingTranscript, setSavingTranscript] = useState(false);

  const getHeaders = useCallback((json = true) => {
    const headers: Record<string, string> = {};
    if (token) {
      headers.Authorization = `Bearer ${token}`;
    }
    if (json) {
      headers["Content-Type"] = "application/json";
    }
    return headers;
  }, [token]);

  const loadProducts = useCallback(async () => {
    if (!token) return;
    try {
      const meRes = await fetch("http://localhost:8000/me", { headers: getHeaders(false) });
      if (!meRes.ok) throw new Error("Unable to load profile");
      const meData = await meRes.json();
      if (!meData?.company_id) {
        throw new Error("Unable to resolve company");
      }
      const productsRes = await fetch(
        `http://localhost:8000/get-products/${meData.company_id}`,
        { headers: getHeaders(false) }
      );
      if (!productsRes.ok) throw new Error("Unable to load products");
      const productPayload = await productsRes.json();
      const items = productPayload.products ?? [];
      setProducts(items);
      if (items.length && !selectedProductId) {
        setSelectedProductId(items[0].id);
      }
    } catch (err) {
      console.error(err);
      setAlert({ type: "error", message: "Unable to load products for Thesis Builder." });
    }
  }, [getHeaders, selectedProductId, token]);

  const loadFeatures = useCallback(async () => {
    if (!selectedProductId || !token) return;
    try {
      const res = await fetch(
        `http://localhost:8000/api/thesis-builder/${selectedProductId}/features`,
        { headers: getHeaders() }
      );
      if (!res.ok) throw new Error("Failed to load features");
      const data = await res.json();
      setFeatures(data);
    } catch (err) {
      console.error(err);
      setAlert({ type: "error", message: "Failed to load thesis features." });
    }
  }, [getHeaders, selectedProductId, token]);

  const loadSegments = useCallback(async () => {
    if (!selectedProductId || !token) return;
    try {
      const res = await fetch(
        `http://localhost:8000/api/thesis-builder/${selectedProductId}/segments`,
        { headers: getHeaders() }
      );
      if (!res.ok) throw new Error("Failed to load segments");
      const data = await res.json();
      setSegments(data);
      setSelectedSegmentId((current) => {
        if (current && data.find((seg: ThesisSegment) => seg.id === current)) {
          return current;
        }
        return data[0]?.id ?? null;
      });
    } catch (err) {
      console.error(err);
      setAlert({ type: "error", message: "Unable to load thesis segments." });
    }
  }, [getHeaders, selectedProductId, token]);

  const loadCalls = useCallback(async () => {
    if (!selectedProductId || !token) return;
    setLoadingCalls(true);
    try {
      const res = await fetch(
        `http://localhost:8000/api/thesis-builder/${selectedProductId}/calls`,
        { headers: getHeaders() }
      );
      if (!res.ok) throw new Error("Failed to load call history");
      const data = await res.json();
      setCalls(data);
    } catch (err) {
      console.error(err);
      setAlert({ type: "error", message: "Unable to refresh call history." });
    } finally {
      setLoadingCalls(false);
    }
  }, [getHeaders, selectedProductId, token]);

  useEffect(() => {
    loadProducts();
  }, [loadProducts]);

  useEffect(() => {
    if (!selectedProductId) return;
    loadFeatures();
    loadSegments();
    loadCalls();
  }, [selectedProductId, loadFeatures, loadSegments, loadCalls]);

  useEffect(() => {
    if (!selectedProductId || !selectedSegmentId) {
      setSegmentSummary(null);
      return;
    }
    setLoadingSummary(true);
    (async () => {
      try {
        const res = await fetch(
          `http://localhost:8000/api/thesis-builder/${selectedProductId}/segments/${selectedSegmentId}/summary`,
          { headers: getHeaders() }
        );
        if (!res.ok) throw new Error("Failed to load segment summary");
        const summary = await res.json();
        setSegmentSummary(summary);
      } catch (err) {
        console.error(err);
        setAlert({ type: "error", message: "Unable to load the selected segment summary." });
      } finally {
        setLoadingSummary(false);
      }
    })();
  }, [selectedProductId, selectedSegmentId, getHeaders]);

  const handleAddFeature = async () => {
    if (!selectedProductId || !token) return;
    if (!featureForm.name || !featureForm.key) {
      setAlert({ type: "error", message: "Feature name and key are required." });
      return;
    }
    try {
      const res = await fetch(
        `http://localhost:8000/api/thesis-builder/${selectedProductId}/features`,
        {
          method: "POST",
          headers: getHeaders(),
          body: JSON.stringify(featureForm),
        }
      );
      if (!res.ok) throw new Error("Failed to create feature");
      const created = await res.json();
      setFeatures((prev) => [created, ...prev]);
      setFeatureForm(initialFeatureForm);
      setAlert({ type: "success", message: "Feature saved." });
    } catch (err) {
      console.error(err);
      setAlert({ type: "error", message: "Unable to save that feature." });
    }
  };

  const handleAddCondition = () => {
    if (!conditionDraft.feature_key) return;
    const newCondition: SegmentRuleCondition = {
      feature_key: conditionDraft.feature_key,
      op: "IN",
      values: conditionDraft.valuesInput
        .split(",")
        .map((value) => value.trim())
        .filter(Boolean),
    };
    setSegmentForm((prev) => ({
      ...prev,
      conditions: [...prev.conditions, newCondition],
    }));
    setConditionDraft({ feature_key: "", valuesInput: "" });
  };

  const handleRemoveCondition = (index: number) => {
    setSegmentForm((prev) => ({
      ...prev,
      conditions: prev.conditions.filter((_, idx) => idx !== index),
    }));
  };

  const handleCreateSegment = async () => {
    if (!selectedProductId || !token) return;
    if (!segmentForm.name || segmentForm.conditions.length === 0) {
      setAlert({ type: "error", message: "Segments require a name and at least one condition." });
      return;
    }
    try {
      const payload = {
        name: segmentForm.name,
        description: segmentForm.description,
        prior_weight: segmentForm.priorWeight,
        rules: { conditions: segmentForm.conditions },
      };
      const res = await fetch(
        `http://localhost:8000/api/thesis-builder/${selectedProductId}/segments`,
        {
          method: "POST",
          headers: getHeaders(),
          body: JSON.stringify(payload),
        }
      );
      if (!res.ok) throw new Error("Failed to create segment");
      const created = await res.json();
      setSegments((prev) => [created, ...prev]);
      setSelectedSegmentId(created.id);
      setSegmentForm(initialSegmentForm);
      setAlert({ type: "success", message: "Segment created." });
    } catch (err) {
      console.error(err);
      setAlert({ type: "error", message: "Unable to create that segment." });
    }
  };

  const handleAcceptDrift = async () => {
    if (!selectedProductId || !selectedSegmentId || !segmentSummary || !token) return;
    const components: Array<Record<string, any>> = [];
    segmentSummary.distributions.attributes.forEach((entry) => {
      components.push({
        component_type: "attribute",
        feature_key: entry.feature_key,
      });
    });
    segmentSummary.distributions.pains.forEach((entry) => {
      components.push({
        component_type: "pain",
        parent_pain_id: entry.parent_pain_id ?? null,
      });
    });
    segmentSummary.distributions.zmots.forEach((entry) => {
      components.push({
        component_type: "zmot",
        parent_pain_id: entry.parent_pain_id,
      });
    });
    segmentSummary.distributions.aspirations.forEach((entry) => {
      components.push({
        component_type: "aspiration",
        parent_pain_id: entry.parent_pain_id,
      });
    });
    segmentSummary.distributions.beliefs.forEach((entry) => {
      components.push({
        component_type: "belief",
        parent_pain_id: entry.parent_pain_id,
        parent_aspiration_id: entry.parent_aspiration_id,
      });
    });
    try {
      const res = await fetch(
        `http://localhost:8000/api/thesis-builder/${selectedProductId}/segments/${selectedSegmentId}/accept-drift`,
        {
          method: "POST",
          headers: getHeaders(),
          body: JSON.stringify({ components }),
        }
      );
      if (!res.ok) throw new Error("Failed to accept drift");
      setAlert({ type: "success", message: "Thesis updated to reflect the new evidence." });
      if (selectedSegmentId) {
        setLoadingSummary(true);
        const summaryRes = await fetch(
          `http://localhost:8000/api/thesis-builder/${selectedProductId}/segments/${selectedSegmentId}/summary`,
          { headers: getHeaders() }
        );
        if (summaryRes.ok) {
          const newSummary = await summaryRes.json();
          setSegmentSummary(newSummary);
        }
        setLoadingSummary(false);
      }
    } catch (err) {
      console.error(err);
      setAlert({ type: "error", message: "Unable to accept that drift." });
    }
  };

  const handleIngestCall = async () => {
    if (!selectedProductId || !token) return;
    if (!callPayloadText.trim()) {
      setAlert({ type: "error", message: "Please paste a canonical call JSON before submitting." });
      return;
    }
    let payload: Record<string, any>;
    try {
      payload = JSON.parse(callPayloadText);
    } catch (err) {
      setAlert({ type: "error", message: "Invalid JSON payload." });
      return;
    }
    if (transcriptText.trim()) {
      payload.transcript_text = transcriptText.trim();
    }
    setIngestingCall(true);
    try {
      const res = await fetch(
        `http://localhost:8000/api/thesis-builder/${selectedProductId}/calls`,
        {
          method: "POST",
          headers: getHeaders(),
          body: JSON.stringify(payload),
        }
      );
      if (!res.ok) {
        const text = await res.text();
        throw new Error(text || "Failed to ingest call");
      }
      setAlert({ type: "success", message: "Call saved and thesis updated." });
      setCallPayloadText("");
      loadCalls();
      if (selectedSegmentId) {
        setLoadingSummary(true);
        const summaryRes = await fetch(
          `http://localhost:8000/api/thesis-builder/${selectedProductId}/segments/${selectedSegmentId}/summary`,
          { headers: getHeaders() }
        );
        if (summaryRes.ok) {
          const newSummary = await summaryRes.json();
          setSegmentSummary(newSummary);
        }
        setLoadingSummary(false);
      }
    } catch (err) {
      console.error(err);
      setAlert({
        type: "error",
        message:
          "Unable to ingest that call." +
          (err instanceof Error && err.message ? ` ${err.message}` : ""),
      });
    } finally {
      setIngestingCall(false);
    }
  };

  const handleSaveTranscript = async () => {
    if (!selectedProductId || !token) return;
    if (!transcriptText.trim()) {
      setAlert({ type: "error", message: "Please paste a call transcript before saving." });
      return;
    }
    setSavingTranscript(true);
    try {
      const payload: Record<string, any> = {
        transcript_text: transcriptText.trim(),
        attribute_bins: {},
      };
      if (transcriptAccountId.trim()) {
        payload.account_id = transcriptAccountId.trim();
      }
      if (transcriptExternalId.trim()) {
        payload.external_call_id = transcriptExternalId.trim();
      }
      const res = await fetch(
        `http://localhost:8000/api/thesis-builder/${selectedProductId}/calls/transcript`,
        {
          method: "POST",
          headers: getHeaders(),
          body: JSON.stringify(payload),
        }
      );
      if (!res.ok) {
        const text = await res.text();
        throw new Error(text || "Failed to save transcript");
      }
      setAlert({ type: "success", message: "Transcript saved to the database." });
      setTranscriptText("");
      setTranscriptAccountId("");
      setTranscriptExternalId("");
      loadCalls();
    } catch (err) {
      console.error(err);
      setAlert({
        type: "error",
        message:
          "Unable to save that transcript." +
          (err instanceof Error && err.message ? ` ${err.message}` : ""),
      });
    } finally {
      setSavingTranscript(false);
    }
  };

  return (
    <Box component="main" sx={{ flexGrow: 1, p: 3 }}>
      <Toolbar />
      <Stack spacing={2}>
        <Box sx={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
          <Typography variant="h4" fontWeight={700}>
            Thesis Builder
          </Typography>
          <FormControl size="small" sx={{ minWidth: 220 }}>
            <InputLabel id="thesis-product-label">Select Product</InputLabel>
            <Select
              labelId="thesis-product-label"
              label="Select Product"
              value={selectedProductId}
              onChange={(event) => setSelectedProductId(event.target.value)}
            >
              {products.map((product) => (
                <MenuItem key={product.id} value={product.id}>
                  {product.name}
                </MenuItem>
              ))}
            </Select>
          </FormControl>
        </Box>
        {alert && (
          <Alert severity={alert.type} onClose={() => setAlert(null)}>
            {alert.message}
          </Alert>
        )}
        <Stack spacing={3}>
          <Card>
            <CardHeader title="Segment Features" />
            <CardContent>
              <Stack spacing={2}>
                <Stack spacing={1}>
                  {features.length ? (
                    features.map((feature) => (
                      <Chip
                        key={feature.id}
                        label={`${feature.name} (${feature.key}) — ${feature.type}`}
                        variant="outlined"
                      />
                    ))
                  ) : (
                    <Typography variant="body2" color="text.secondary">
                      No features defined yet.
                    </Typography>
                  )}
                </Stack>
                <Divider />
                <Stack spacing={2}>
                  <TextField
                    label="Feature Name"
                    value={featureForm.name}
                    onChange={(event) =>
                      setFeatureForm((prev) => ({ ...prev, name: event.target.value }))
                    }
                    size="small"
                    fullWidth
                  />
                  <TextField
                    label="Feature Key"
                    value={featureForm.key}
                    onChange={(event) =>
                      setFeatureForm((prev) => ({ ...prev, key: event.target.value }))
                    }
                    size="small"
                    fullWidth
                    helperText="Use the canonical bin key (e.g. gmv_bin)."
                  />
                  <FormControl size="small" fullWidth>
                    <InputLabel id="feature-type-label">Type</InputLabel>
                    <Select
                      labelId="feature-type-label"
                      label="Type"
                      value={featureForm.type}
                      onChange={(event) =>
                        setFeatureForm((prev) => ({
                          ...prev,
                          type: event.target.value as ThesisAttributeType,
                        }))
                      }
                    >
                      <MenuItem value="numeric">Numeric</MenuItem>
                      <MenuItem value="categorical">Categorical</MenuItem>
                      <MenuItem value="boolean">Boolean</MenuItem>
                    </Select>
                  </FormControl>
                  <Button fullWidth variant="contained" onClick={handleAddFeature}>
                    Add Feature
                  </Button>
                </Stack>
              </Stack>
            </CardContent>
          </Card>
          <Card>
            <CardHeader title="Segments" />
            <CardContent>
              <Stack spacing={2}>
                <Stack spacing={1}>
                  <TextField
                    label="Segment Name"
                    value={segmentForm.name}
                    onChange={(event) =>
                      setSegmentForm((prev) => ({ ...prev, name: event.target.value }))
                    }
                    size="small"
                    fullWidth
                  />
                  <TextField
                    label="Description"
                    value={segmentForm.description}
                    onChange={(event) =>
                      setSegmentForm((prev) => ({ ...prev, description: event.target.value }))
                    }
                    size="small"
                    fullWidth
                    multiline
                    minRows={2}
                  />
                  <TextField
                    label="Prior Weight"
                    type="number"
                    value={segmentForm.priorWeight}
                    onChange={(event) =>
                      setSegmentForm((prev) => ({
                        ...prev,
                        priorWeight: Number(event.target.value) || 1,
                      }))
                    }
                    size="small"
                    fullWidth
                  />
                </Stack>
                <Divider />
                <Typography variant="subtitle2">Rule Builder</Typography>
                <Stack spacing={1}>
                  <FormControl size="small">
                    <InputLabel id="condition-feature-label">Feature</InputLabel>
                    <Select
                      labelId="condition-feature-label"
                      label="Feature"
                      value={conditionDraft.feature_key}
                      onChange={(event) =>
                        setConditionDraft((prev) => ({
                          ...prev,
                          feature_key: event.target.value,
                        }))
                      }
                    >
                      {features.map((feature) => (
                        <MenuItem key={feature.id} value={feature.key}>
                          {feature.name} ({feature.key})
                        </MenuItem>
                      ))}
                    </Select>
                  </FormControl>
                  <TextField
                    label="Values (comma separated)"
                    value={conditionDraft.valuesInput}
                    onChange={(event) =>
                      setConditionDraft((prev) => ({
                        ...prev,
                        valuesInput: event.target.value,
                      }))
                    }
                    size="small"
                  />
                  <Button variant="outlined" size="small" onClick={handleAddCondition}>
                    Add Condition
                  </Button>
                  <Stack direction="row" spacing={1} flexWrap="wrap">
                    {segmentForm.conditions.map((condition, index) => (
                      <Chip
                        key={`${condition.feature_key}-${index}`}
                        label={`${condition.feature_key} IN ${condition.values?.join(", ") || "ANY"}`}
                        onDelete={() => handleRemoveCondition(index)}
                        size="small"
                      />
                    ))}
                    {!segmentForm.conditions.length && (
                      <Typography variant="caption" color="text.secondary">
                        Add at least one condition before saving.
                      </Typography>
                    )}
                  </Stack>
                </Stack>
                <Button variant="contained" onClick={handleCreateSegment}>
                  Create Segment
                </Button>
                <Divider />
                <TableContainer component={Paper} variant="outlined">
                  <Table size="small">
                    <TableHead>
                      <TableRow>
                        <TableCell>Name</TableCell>
                        <TableCell>Description</TableCell>
                        <TableCell>Prior Weight</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {segments.map((segment) => (
                        <TableRow
                          key={segment.id}
                          hover
                          selected={segment.id === selectedSegmentId}
                          onClick={() => setSelectedSegmentId(segment.id)}
                        >
                          <TableCell>{segment.name}</TableCell>
                          <TableCell>{segment.description || "—"}</TableCell>
                          <TableCell>{segment.prior_weight.toFixed(1)}</TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </TableContainer>
              </Stack>
            </CardContent>
          </Card>
          <Card>
            <CardHeader title="Segment Thesis" />
            <CardContent>
              {loadingSummary ? (
                <CircularProgress />
              ) : segmentSummary ? (
                <Stack spacing={2}>
                  <Stack direction="row" justifyContent="space-between" alignItems="center">
                    <Typography variant="h6">{segmentSummary.segment.name}</Typography>
                    <Button variant="outlined" onClick={handleAcceptDrift}>
                      Accept Updated Thesis
                    </Button>
                  </Stack>
                  <Typography variant="body2" color="text.secondary">
                    {segmentSummary.segment.description || "No description provided."}
                  </Typography>
                  <Divider />
                  <Stack spacing={2}>
                    {segmentSummary.distributions.attributes.map((entry) => (
                      <Stack spacing={1} key={entry.feature_key}>
                        <Stack direction="row" alignItems="center" justifyContent="space-between">
                          <Typography variant="subtitle2">{entry.feature_key}</Typography>
                          <Chip label={entry.drift_level} size="small" />
                        </Stack>
                        <LinearProgress variant="determinate" value={entry.drift * 100} />
                        {Object.keys({ ...entry.prior, ...entry.posterior }).map((category) => (
                          <Stack
                            key={`${entry.feature_key}-${category}`}
                            direction="row"
                            justifyContent="space-between"
                          >
                            <Typography variant="body2">{category}</Typography>
                            <Typography variant="body2" color="text.secondary">
                              {formatPercent(entry.prior[category])} → {formatPercent(entry.posterior[category])}
                            </Typography>
                          </Stack>
                        ))}
                      </Stack>
                    ))}
                    {segmentSummary.distributions.pains.map((entry) => (
                      <Stack key={`pain-${entry.parent_pain_id || "root"}`}>
                        <Stack direction="row" alignItems="center" justifyContent="space-between">
                          <Typography variant="subtitle2">Pain {entry.parent_pain_id || "primary"}</Typography>
                          <Chip label={entry.drift_level} size="small" />
                        </Stack>
                        <LinearProgress variant="determinate" value={entry.drift * 100} />
                      </Stack>
                    ))}
                    {segmentSummary.distributions.zmots.map((entry) => (
                      <Stack key={`zmot-${entry.parent_pain_id}`}>
                        <Stack direction="row" alignItems="center" justifyContent="space-between">
                          <Typography variant="subtitle2">ZMOT for {entry.parent_pain_id}</Typography>
                          <Chip label={entry.drift_level} size="small" />
                        </Stack>
                        <LinearProgress variant="determinate" value={entry.drift * 100} />
                      </Stack>
                    ))}
                    {segmentSummary.distributions.aspirations.map((entry) => (
                      <Stack key={`asp-${entry.parent_pain_id}`}>
                        <Stack direction="row" alignItems="center" justifyContent="space-between">
                          <Typography variant="subtitle2">Aspiration for {entry.parent_pain_id}</Typography>
                          <Chip label={entry.drift_level} size="small" />
                        </Stack>
                        <LinearProgress variant="determinate" value={entry.drift * 100} />
                      </Stack>
                    ))}
                    {segmentSummary.distributions.beliefs.map((entry) => (
                      <Stack key={`belief-${entry.parent_pain_id}-${entry.parent_aspiration_id}`}>
                        <Stack direction="row" alignItems="center" justifyContent="space-between">
                          <Typography variant="subtitle2">
                            Belief for {entry.parent_pain_id}/{entry.parent_aspiration_id}
                          </Typography>
                          <Chip label={entry.drift_level} size="small" />
                        </Stack>
                        <LinearProgress variant="determinate" value={entry.drift * 100} />
                      </Stack>
                    ))}
                  </Stack>
                </Stack>
              ) : (
                <Typography variant="body2" color="text.secondary">
                  Select or create a segment to visualize its thesis and drift metrics.
                </Typography>
              )}
            </CardContent>
          </Card>
          <Card>
            <CardHeader title="Call History & Ingestion" />
            <CardContent>
              <Stack spacing={2}>
                <Typography variant="body2" color="text.secondary">
                  Paste a canonical call payload to ingest it into the thesis model. The JSON
                  should match the canonical_summary + metadata shape described in the API
                  contract.
                </Typography>
                <TextField
                  label="Canonical Call JSON"
                  multiline
                  minRows={6}
                  value={callPayloadText}
                  onChange={(event) => setCallPayloadText(event.target.value)}
                  placeholder={`{
  "account_id": "acc_123",
  "external_call_id": "gong_abc",
  "attribute_bins": { "gmv_bin": "gmv_1_10m", "store_count_bin": "stores_1_3" },
  "stakeholders": [],
  "canonical_summary": {
    "pains": [],
    "zmots": [],
    "aspirations": [],
    "beliefs": []
  }
}`}
                  variant="outlined"
                  fullWidth
                  InputLabelProps={{ shrink: true }}
                />
                <Button
                  variant="contained"
                  onClick={handleIngestCall}
                  disabled={ingestingCall}
                >
                  {ingestingCall ? "Ingesting..." : "Ingest Call"}
                </Button>
                <Divider />
                <Typography variant="body2" color="text.secondary">
                  Upload a full call transcript (saved independently).
                </Typography>
                <Stack spacing={1}>
                  <TextField
                    label="Account ID (optional)"
                    value={transcriptAccountId}
                    onChange={(event) => setTranscriptAccountId(event.target.value)}
                    size="small"
                    fullWidth
                  />
                  <TextField
                    label="External Call ID (optional)"
                    value={transcriptExternalId}
                    onChange={(event) => setTranscriptExternalId(event.target.value)}
                    size="small"
                    fullWidth
                  />
                </Stack>
                <TextField
                  label="Call Transcript"
                  multiline
                  minRows={6}
                  value={transcriptText}
                  onChange={(event) => setTranscriptText(event.target.value)}
                  variant="outlined"
                  fullWidth
                />
                <Button
                  variant="outlined"
                  onClick={handleSaveTranscript}
                  disabled={savingTranscript}
                >
                  {savingTranscript ? "Saving transcript..." : "Save Transcript"}
                </Button>
                <Divider />
                {loadingCalls ? (
                  <CircularProgress />
                ) : calls.length ? (
                  <TableContainer component={Paper} variant="outlined">
                    <Table size="small">
                      <TableHead>
                        <TableRow>
                          <TableCell>Date</TableCell>
                          <TableCell>Account</TableCell>
                          <TableCell>Segment</TableCell>
                          <TableCell>Attributes</TableCell>
                        </TableRow>
                      </TableHead>
                      <TableBody>
                        {calls.map((call) => (
                          <TableRow key={call.call_id}>
                            <TableCell>
                              {new Date(call.created_at).toLocaleString()}
                            </TableCell>
                            <TableCell>{call.account_id || call.external_call_id || "—"}</TableCell>
                            <TableCell>
                              {segments.find((segment) => segment.id === call.segment_id)?.name || "—"}
                            </TableCell>
                            <TableCell>
                              {Object.entries(call.attribute_bins)
                                .map(([key, value]) => `${key}: ${value}`)
                                .join(", ")}
                            </TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </TableContainer>
                ) : (
                  <Typography variant="body2" color="text.secondary">
                    Calls you save through the Thesis Builder will show up here with their canonical evidence.
                  </Typography>
                )}
              </Stack>
            </CardContent>
          </Card>
        </Stack>
      </Stack>
    </Box>
  );
};

export default ThesisBuilderPage;
