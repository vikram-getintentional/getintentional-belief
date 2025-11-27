import { ChangeEvent, useEffect, useState } from "react";
import {
  Alert,
  Box,
  Button,
  Container,
  Dialog,
  DialogActions,
  DialogContent,
  DialogTitle,
  FormControl,
  Grid,
  InputLabel,
  MenuItem,
  Select,
  Snackbar,
  Tab,
  Tabs,
  TextField,
  Typography,
} from "@mui/material";
import AssetsLibrary from "../components/AssetsLibrary";
import ChannelsLibrary from "../components/ChannelsLibrary";

type EnumOption = { value: string; label: string };

type AssetOptionGroups = {
  category: EnumOption[];
  content_type: EnumOption[];
  time_to_consume: EnumOption[];
  depth: EnumOption[];
  sales_call_stage: EnumOption[];
};

type ChannelOptionGroups = {
  channel_type: EnumOption[];
  delivery_mode: EnumOption[];
};

type AssetFormState = {
  category: string;
  content_type: string;
  time_to_consume: string;
  depth: string;
  description: string;
  notes: string;
  callStage: string;
};

type ChannelFormState = {
  channel_type: string;
  delivery_mode: string;
  reach_score_estimate: string;
  notes: string;
};

type SnackState = {
  open: boolean;
  message: string;
  severity: "success" | "error" | "info" | "warning";
};

const emptyAssetOptions: AssetOptionGroups = {
  category: [],
  content_type: [],
  time_to_consume: [],
  depth: [],
  sales_call_stage: [],
};

const emptyChannelOptions: ChannelOptionGroups = {
  channel_type: [],
  delivery_mode: [],
};

const createDefaultAssetForm = (): AssetFormState => ({
  category: "",
  content_type: "",
  time_to_consume: "",
  depth: "",
  description: "",
  notes: "",
  callStage: "",
});

const createDefaultChannelForm = (): ChannelFormState => ({
  channel_type: "",
  delivery_mode: "",
  reach_score_estimate: "",
  notes: "",
});

const Arsenal = () => {
  const [products, setProducts] = useState<{ id: string; name: string }[]>([]);
  const [selectedProductId, setSelectedProductId] = useState<string | null>(null);
  const [statusMsg, setStatusMsg] = useState("");
  const token = typeof window !== "undefined" ? localStorage.getItem("token") : null;

  const [arsenalAssets, setArsenalAssets] = useState<any[]>([]);
  const [arsenalChannels, setArsenalChannels] = useState<any[]>([]);
  const [tabIndex, setTabIndex] = useState(0);
  const [assetMetaGaps, setAssetMetaGaps] = useState(0);
  const [channelMetaGaps, setChannelMetaGaps] = useState(0);

  const [assetOptions, setAssetOptions] = useState<AssetOptionGroups>(emptyAssetOptions);
  const [channelOptions, setChannelOptions] = useState<ChannelOptionGroups>(emptyChannelOptions);

  const [editingAsset, setEditingAsset] = useState<any | null>(null);
  const [editingChannel, setEditingChannel] = useState<any | null>(null);
  const [assetForm, setAssetForm] = useState<AssetFormState>(createDefaultAssetForm());
  const [channelForm, setChannelForm] = useState<ChannelFormState>(createDefaultChannelForm());
  const [assetSaving, setAssetSaving] = useState(false);
  const [channelSaving, setChannelSaving] = useState(false);

  const [snack, setSnack] = useState<SnackState>({
    open: false,
    message: "",
    severity: "success",
  });

  const recomputeMetaGaps = (assetsList: any[], channelsList: any[]) => {
    setAssetMetaGaps(assetsList.filter((asset) => !asset.metadata_complete).length);
    setChannelMetaGaps(channelsList.filter((channel) => !channel.metadata_complete).length);
  };

  useEffect(() => {
    if (!token) return;
    const fetchCompanyAndProducts = async () => {
      try {
        const meRes = await fetch("http://localhost:8000/me", {
          headers: { Authorization: `Bearer ${token}` },
        });
        if (!meRes.ok) throw new Error("Failed to load user");
        const meData = await meRes.json();

        const prodRes = await fetch(
          `http://localhost:8000/get-products/${meData.company_id}`,
          {
            headers: { Authorization: `Bearer ${token}` },
          }
        );
        if (!prodRes.ok) throw new Error("Failed to load products");
        const prodData = await prodRes.json();
        const list = prodData.products || [];
        if (!list.length) {
          setStatusMsg("No products found. Please run Value Prop first.");
          return;
        }
        setProducts(list);
        setSelectedProductId((prev) => prev || list[0].id);
      } catch (err) {
        setStatusMsg("Error fetching company or products.");
      }
    };
    fetchCompanyAndProducts();
  }, [token]);

  useEffect(() => {
    if (!token) return;
    const fetchOptions = async () => {
      try {
        const res = await fetch("http://localhost:8000/arsenal/options", {
          headers: { Authorization: `Bearer ${token}` },
        });
        if (!res.ok) throw new Error();
        const data = await res.json();
        const assetOpts = data.asset ?? {};
        const channelOpts = data.channel ?? {};
        setAssetOptions({
          category: assetOpts.category ?? [],
          content_type: assetOpts.content_type ?? [],
          time_to_consume: assetOpts.time_to_consume ?? [],
          depth: assetOpts.depth ?? [],
          sales_call_stage: assetOpts.sales_call_stage ?? [],
        });
        setChannelOptions({
          channel_type: channelOpts.channel_type ?? [],
          delivery_mode: channelOpts.delivery_mode ?? [],
        });
      } catch (err) {
        console.warn("Failed to load arsenal options", err);
      }
    };
    fetchOptions();
  }, [token]);

  useEffect(() => {
    if (!selectedProductId || !token) return;
    const fetchArsenal = async () => {
      try {
        const res = await fetch(
          `http://localhost:8000/get-arsenal-library/${selectedProductId}`,
          { headers: { Authorization: `Bearer ${token}` } }
        );
        if (!res.ok) throw new Error("Failed to load arsenal");
        const data = await res.json();
        const assets = data.arsenal.assets ?? [];
        const channels = data.arsenal.channels ?? [];
        setArsenalAssets(assets);
        setArsenalChannels(channels);
        recomputeMetaGaps(assets, channels);
      } catch (err) {
        setStatusMsg("Error fetching Arsenal Library.");
      }
    };
    fetchArsenal();
  }, [selectedProductId, token]);

  useEffect(() => {
    if (editingAsset) {
      setAssetForm({
        category: editingAsset.category || "",
        content_type: editingAsset.content_type || "",
        time_to_consume: editingAsset.time_to_consume || "",
        depth: editingAsset.depth || "",
        description: editingAsset.description || "",
        notes: editingAsset.notes || "",
        callStage: editingAsset.call_stage || "",
      });
    } else {
      setAssetForm(createDefaultAssetForm());
    }
  }, [editingAsset]);

  useEffect(() => {
    if (editingChannel) {
      setChannelForm({
        channel_type: editingChannel.channel_type || "",
        delivery_mode: editingChannel.delivery_mode || "",
        reach_score_estimate:
          editingChannel.reach_score_estimate === null ||
          editingChannel.reach_score_estimate === undefined
            ? ""
            : String(editingChannel.reach_score_estimate),
        notes: editingChannel.notes || "",
      });
    } else {
      setChannelForm(createDefaultChannelForm());
    }
  }, [editingChannel]);

  const handleAssetFieldChange = (field: keyof AssetFormState, value: string) => {
    setAssetForm((prev) => {
      const next = { ...prev, [field]: value };
      if (field === "category" && value !== "sales_call" && prev.callStage) {
        next.callStage = "";
      }
      return next;
    });
  };

  const handleChannelFieldChange = (field: keyof ChannelFormState, value: string) => {
    setChannelForm((prev) => ({ ...prev, [field]: value }));
  };

  const handleAssetSave = async () => {
    if (!editingAsset || !token) return;
    setAssetSaving(true);
    try {
      const res = await fetch(
        `http://localhost:8000/arsenal/assets/${editingAsset.id}`,
        {
          method: "PATCH",
          headers: {
            Authorization: `Bearer ${token}`,
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            category: assetForm.category || null,
            content_type: assetForm.content_type || null,
            time_to_consume: assetForm.time_to_consume || null,
            depth: assetForm.depth || null,
            description: assetForm.description,
            notes: assetForm.notes,
            call_stage: assetForm.callStage || null,
          }),
        }
      );
      const data = await res.json().catch(() => null);
      if (!res.ok) {
        throw new Error((data && data.detail) || "Failed to update asset metadata");
      }
      const updated = data.asset;
      setArsenalAssets((prev) => {
        const next = prev.map((asset) => (asset.id === updated.id ? updated : asset));
        recomputeMetaGaps(next, arsenalChannels);
        return next;
      });
      setSnack({ open: true, message: "Asset metadata updated", severity: "success" });
      setEditingAsset(null);
    } catch (err: any) {
      setSnack({
        open: true,
        message: err?.message || "Failed to update asset metadata",
        severity: "error",
      });
    } finally {
      setAssetSaving(false);
    }
  };

  const handleChannelSave = async () => {
    if (!editingChannel || !token) return;
    setChannelSaving(true);
    try {
      const reachValue = channelForm.reach_score_estimate.trim();
      let reachPayload: number | null = null;
      if (reachValue !== "") {
        const parsed = Number(reachValue);
        if (Number.isNaN(parsed)) {
          throw new Error("Reach score must be a number");
        }
        reachPayload = parsed;
      }

      const res = await fetch(
        `http://localhost:8000/arsenal/channels/${editingChannel.id}`,
        {
          method: "PATCH",
          headers: {
            Authorization: `Bearer ${token}`,
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            channel_type: channelForm.channel_type || null,
            delivery_mode: channelForm.delivery_mode || null,
            reach_score_estimate: reachPayload,
            notes: channelForm.notes,
          }),
        }
      );
      const data = await res.json().catch(() => null);
      if (!res.ok) {
        throw new Error((data && data.detail) || "Failed to update channel metadata");
      }
      const updated = data.channel;
      setArsenalChannels((prev) => {
        const next = prev.map((channel) => (channel.id === updated.id ? updated : channel));
        recomputeMetaGaps(arsenalAssets, next);
        return next;
      });
      setSnack({ open: true, message: "Channel metadata updated", severity: "success" });
      setEditingChannel(null);
    } catch (err: any) {
      setSnack({
        open: true,
        message: err?.message || "Failed to update channel metadata",
        severity: "error",
      });
    } finally {
      setChannelSaving(false);
    }
  };

  const handleSnackClose = () => {
    setSnack((prev) => ({ ...prev, open: false }));
  };

  return (
    <Container maxWidth="md">
      <Typography variant="h4" gutterBottom>
        Arsenal Library
      </Typography>

      {products.length > 1 && (
        <Box mb={3}>
          <FormControl fullWidth>
            <InputLabel>Select Product</InputLabel>
            <Select
              value={selectedProductId || ""}
              label="Select Product"
              onChange={(event) => setSelectedProductId(event.target.value as string)}
            >
              <MenuItem value="" disabled>
                Select a product
              </MenuItem>
              {products.map((product) => (
                <MenuItem key={product.id} value={product.id}>
                  {product.name}
                </MenuItem>
              ))}
            </Select>
          </FormControl>
        </Box>
      )}

      {statusMsg && (
        <Typography color="error" mb={2}>
          {statusMsg}
        </Typography>
      )}

      {(assetMetaGaps > 0 || channelMetaGaps > 0) && (
        <Alert severity="warning" sx={{ mb: 3 }}>
          {assetMetaGaps > 0 && (
            <span>
              {assetMetaGaps} asset{assetMetaGaps === 1 ? "" : "s"} need metadata.
            </span>
          )}
          {assetMetaGaps > 0 && channelMetaGaps > 0 && <span>&nbsp;•&nbsp;</span>}
          {channelMetaGaps > 0 && (
            <span>
              {channelMetaGaps} channel{channelMetaGaps === 1 ? "" : "s"} need metadata.
            </span>
          )}
        </Alert>
      )}

      <Tabs
        value={tabIndex}
        onChange={(_, newValue) => setTabIndex(newValue)}
        indicatorColor="primary"
        textColor="primary"
        variant="fullWidth"
        sx={{ mb: 3 }}
      >
        <Tab label="Assets" />
        <Tab label="Channels" />
        <Tab label="Campaigns" />
      </Tabs>

      <Box hidden={tabIndex !== 0}>
        <Typography variant="h5" gutterBottom>
          Assets
        </Typography>
        <Grid container spacing={2} mb={4} alignItems="stretch">
          <AssetsLibrary assets={arsenalAssets} onEdit={setEditingAsset} />
        </Grid>
      </Box>

      <Box hidden={tabIndex !== 1}>
        <Typography variant="h5" gutterBottom>
          Channels
        </Typography>
        <Grid container spacing={2}>
          <ChannelsLibrary channels={arsenalChannels} onEdit={setEditingChannel} />
        </Grid>
      </Box>

      <Dialog
        open={Boolean(editingAsset)}
        onClose={() => !assetSaving && setEditingAsset(null)}
        maxWidth="sm"
        fullWidth
      >
        <DialogTitle>Edit Asset Metadata</DialogTitle>
        <DialogContent dividers>
          <Box sx={{ display: "grid", gap: 2 }}>
            <FormControl fullWidth>
              <InputLabel id="asset-category-label">Category</InputLabel>
              <Select
                labelId="asset-category-label"
                label="Category"
                value={assetForm.category}
                onChange={(event) =>
                  handleAssetFieldChange("category", event.target.value as string)
                }
              >
                <MenuItem value="">
                  <em>Unspecified</em>
                </MenuItem>
                {assetOptions.category.map((option) => (
                  <MenuItem key={option.value} value={option.value}>
                    {option.label}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>

            <FormControl fullWidth>
              <InputLabel id="asset-content-type-label">Content Type</InputLabel>
              <Select
                labelId="asset-content-type-label"
                label="Content Type"
                value={assetForm.content_type}
                onChange={(event) =>
                  handleAssetFieldChange("content_type", event.target.value as string)
                }
              >
                <MenuItem value="">
                  <em>Unspecified</em>
                </MenuItem>
                {assetOptions.content_type.map((option) => (
                  <MenuItem key={option.value} value={option.value}>
                    {option.label}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>

            {assetForm.category === "sales_call" && (
              <FormControl fullWidth>
                <InputLabel id="asset-sales-stage-label">Sales Call Stage</InputLabel>
                <Select
                  labelId="asset-sales-stage-label"
                  label="Sales Call Stage"
                  value={assetForm.callStage}
                  onChange={(event) =>
                    handleAssetFieldChange("callStage", event.target.value as string)
                  }
                >
                  <MenuItem value="">
                    <em>Select stage</em>
                  </MenuItem>
                  {assetOptions.sales_call_stage.map((option) => (
                    <MenuItem key={option.value} value={option.value}>
                      {option.label}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            )}

            <FormControl fullWidth>
              <InputLabel id="asset-time-label">Time To Consume</InputLabel>
              <Select
                labelId="asset-time-label"
                label="Time To Consume"
                value={assetForm.time_to_consume}
                onChange={(event) =>
                  handleAssetFieldChange("time_to_consume", event.target.value as string)
                }
              >
                <MenuItem value="">
                  <em>Unspecified</em>
                </MenuItem>
                {assetOptions.time_to_consume.map((option) => (
                  <MenuItem key={option.value} value={option.value}>
                    {option.label}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>

            <FormControl fullWidth>
              <InputLabel id="asset-depth-label">Depth</InputLabel>
              <Select
                labelId="asset-depth-label"
                label="Depth"
                value={assetForm.depth}
                onChange={(event) => handleAssetFieldChange("depth", event.target.value as string)}
              >
                <MenuItem value="">
                  <em>Unspecified</em>
                </MenuItem>
                {assetOptions.depth.map((option) => (
                  <MenuItem key={option.value} value={option.value}>
                    {option.label}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>

            <TextField
              label="Description"
              value={assetForm.description}
              onChange={(event: ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) =>
                handleAssetFieldChange("description", event.target.value)
              }
              multiline
              minRows={2}
            />

            <TextField
              label="Notes"
              value={assetForm.notes}
              onChange={(event: ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) =>
                handleAssetFieldChange("notes", event.target.value)
              }
              multiline
              minRows={2}
            />
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setEditingAsset(null)} disabled={assetSaving}>
            Cancel
          </Button>
          <Button onClick={handleAssetSave} variant="contained" disabled={assetSaving}>
            Save
          </Button>
        </DialogActions>
      </Dialog>

      <Dialog
        open={Boolean(editingChannel)}
        onClose={() => !channelSaving && setEditingChannel(null)}
        maxWidth="sm"
        fullWidth
      >
        <DialogTitle>Edit Channel Metadata</DialogTitle>
        <DialogContent dividers>
          <Box sx={{ display: "grid", gap: 2 }}>
            <FormControl fullWidth>
              <InputLabel id="channel-type-label">Channel Type</InputLabel>
              <Select
                labelId="channel-type-label"
                label="Channel Type"
                value={channelForm.channel_type}
                onChange={(event) =>
                  handleChannelFieldChange("channel_type", event.target.value as string)
                }
              >
                <MenuItem value="">
                  <em>Unspecified</em>
                </MenuItem>
                {channelOptions.channel_type.map((option) => (
                  <MenuItem key={option.value} value={option.value}>
                    {option.label}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>

            <FormControl fullWidth>
              <InputLabel id="channel-delivery-label">Delivery Mode</InputLabel>
              <Select
                labelId="channel-delivery-label"
                label="Delivery Mode"
                value={channelForm.delivery_mode}
                onChange={(event) =>
                  handleChannelFieldChange("delivery_mode", event.target.value as string)
                }
              >
                <MenuItem value="">
                  <em>Unspecified</em>
                </MenuItem>
                {channelOptions.delivery_mode.map((option) => (
                  <MenuItem key={option.value} value={option.value}>
                    {option.label}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>

            <TextField
              label="Reach Score Estimate"
              value={channelForm.reach_score_estimate}
              onChange={(event: ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) =>
                handleChannelFieldChange("reach_score_estimate", event.target.value)
              }
              type="number"
              inputProps={{ step: 0.01, min: 0, max: 1 }}
              helperText="Optional. Enter a value between 0 and 1."
            />

            <TextField
              label="Notes"
              value={channelForm.notes}
              onChange={(event: ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) =>
                handleChannelFieldChange("notes", event.target.value)
              }
              multiline
              minRows={2}
            />
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setEditingChannel(null)} disabled={channelSaving}>
            Cancel
          </Button>
          <Button onClick={handleChannelSave} variant="contained" disabled={channelSaving}>
            Save
          </Button>
        </DialogActions>
      </Dialog>

      <Snackbar
        open={snack.open}
        autoHideDuration={4000}
        onClose={handleSnackClose}
        anchorOrigin={{ vertical: "bottom", horizontal: "center" }}
      >
        <Alert onClose={handleSnackClose} severity={snack.severity} sx={{ width: "100%" }}>
          {snack.message}
        </Alert>
      </Snackbar>
    </Container>
  );
};

export default Arsenal;
