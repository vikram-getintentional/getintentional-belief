import { useCallback, useEffect, useMemo, useState } from "react";
import { Add as AddIcon } from "@mui/icons-material";
import {
  Box,
  Button,
  Card,
  CardContent,
  CardHeader,
  Grid,
  MenuItem,
  Stack,
  TextField,
  Typography,
} from "@mui/material";
import ValuePropCard from "../components/ValuePropCard";
import { alpha, useTheme } from "@mui/material/styles";

import ConfirmWebsite from "../components/ConfirmWebsite";

const API_BASE = "http://localhost:8000";

type Capability = {
  id?: string;
  coreness: number;
  centrality: number;
  name: string;
  description?: string;
};

type ProductData = {
  product_node_id: string;
  summary: string;
  plg_flag?: boolean;
  capabilities: Capability[];
};

const formatMetricValue = (value?: number) =>
  typeof value === "number" ? value.toFixed(2) : "—";

const ValueProposition = () => {
  const [_, setEmail] = useState('');
  const [companyName, setCompanyName] = useState('');

  const [companyId, setCompanyId] = useState<string | null>(null);
  const [products, setProducts] = useState<{ id: string; name: string }[]>([]);
  const [selectedProductId, setSelectedProductId] = useState<string | null>(null);
  const [productData, setProductData] = useState<ProductData | null>(null);
  const [statusMsg, setStatusMsg] = useState("");
  const token = localStorage.getItem("token");
  const [showConfirmWebsite, setShowConfirmWebsite] = useState(false);
  const [summaryDraft, setSummaryDraft] = useState("");
  const [capabilityDrafts, setCapabilityDrafts] =
    useState<Record<string, { name: string; description: string }>>({});
  const [newCapability, setNewCapability] = useState({ name: "", description: "" });
  const [editingCapabilityId, setEditingCapabilityId] = useState<string | null>(null);
  const [isNewCapabilityOpen, setIsNewCapabilityOpen] = useState(false);
  const [isSavingAll, setIsSavingAll] = useState(false);
  const [saveAllStatus, setSaveAllStatus] = useState("");
  const theme = useTheme();
  const handleAnalyzeSuccess = (data: any) => {
    console.log("Analyze succeeded:", data);
    if (data?.product_node_id) {
      setSelectedProductId(data.product_node_id);
    };
    setShowConfirmWebsite(false);
  };

  const renderConfirmWebsite = () => {
      return (
        <>
          {console.log("Rendering ConfirmWebsite component with productID:", selectedProductId)}
          <div className="p-8">
            <h2 className="text-2xl font-bold mb-6">Confirm Website</h2>
            <ConfirmWebsite onAnalyzeSuccess={handleAnalyzeSuccess} />
          </div>
        </>
      );
  };

  const handleCapabilityFieldChange = (capId: string, field: "name" | "description", value: string) => {
    setCapabilityDrafts((prev) => ({
      ...prev,
      [capId]: {
        ...(prev[capId] || { name: "", description: "" }),
        [field]: value,
      },
    }));
  };

  const changedCapabilityIds = useMemo(() => {
    if (!productData) {
      return [];
    }
    return (productData.capabilities || [])
      .map((cap) => {
        const capId = cap.id || cap.name;
        const draft = capabilityDrafts[capId];
        if (!draft) return null;
        const originalDescription = cap.description || "";
        if (draft.name !== cap.name || draft.description !== originalDescription) {
          return capId;
        }
        return null;
      })
      .filter((id): id is string => Boolean(id));
  }, [productData, capabilityDrafts]);

  const updateSummaryOnServer = async () => {
    if (!selectedProductId) return;
    const res = await fetch(
      `${API_BASE}/value-prop/update-summary/${selectedProductId}`,
      {
        method: "POST",
        headers: {
          Authorization: `Bearer ${token}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ summary: summaryDraft }),
      }
    );
    if (!res.ok) {
      throw new Error("Failed to save summary");
    }
  };

  const updateCapabilityOnServer = async (capId: string) => {
    if (!selectedProductId) return;
    const draft = capabilityDrafts[capId];
    if (!draft) return;
    const res = await fetch(
      `${API_BASE}/value-prop/capabilities/${selectedProductId}/${capId}`,
      {
        method: "PATCH",
        headers: {
          Authorization: `Bearer ${token}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          name: draft.name,
          description: draft.description,
        }),
      }
    );
    if (!res.ok) {
      throw new Error("Failed to update capability");
    }
  };

  const createCapabilityOnServer = async () => {
    if (!selectedProductId) return;
    const trimmedName = newCapability.name.trim();
    const trimmedDescription = newCapability.description.trim();
    const res = await fetch(`${API_BASE}/value-prop/capabilities/${selectedProductId}`, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${token}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        name: trimmedName,
        description: trimmedDescription,
      }),
    });
    if (!res.ok) {
      throw new Error("Failed to create capability");
    }
  };

  const handleSaveAll = async () => {
    if (!selectedProductId || !productData) {
      return;
    }
    const summaryChanged = summaryDraft !== (productData.summary || "");
    const newCapabilityName = newCapability.name.trim();
    const newCapabilityDescription = newCapability.description.trim();
    const hasNewCapabilityDraft = newCapabilityName !== "" || newCapabilityDescription !== "";
    if (hasNewCapabilityDraft && newCapabilityName === "") {
      setSaveAllStatus("Provide a name before saving the new capability.");
      return;
    }

    const shouldAddCapability = newCapabilityName !== "";
    if (!summaryChanged && changedCapabilityIds.length === 0 && !shouldAddCapability) {
      return;
    }

    setIsSavingAll(true);
    setSaveAllStatus("Saving changes...");
    try {
      if (summaryChanged) {
        await updateSummaryOnServer();
      }
      for (const capId of changedCapabilityIds) {
        await updateCapabilityOnServer(capId);
      }
      if (shouldAddCapability) {
        await createCapabilityOnServer();
      }
      await fetchProductData();
      setSaveAllStatus("All changes saved.");
      setStatusMsg("");
      setEditingCapabilityId(null);
      if (shouldAddCapability) {
        setNewCapability({ name: "", description: "" });
        setIsNewCapabilityOpen(false);
      }
    } catch (err) {
      console.error(err);
      setSaveAllStatus("Failed to save changes.");
      setStatusMsg("Error saving changes.");
    } finally {
      setIsSavingAll(false);
    }
  };

  // Fetch company and products on mount
  useEffect(() => {
    const fetchCompanyAndProducts = async () => {
      try {
        const meRes = await fetch("http://localhost:8000/me", {
          headers: { Authorization: `Bearer ${token}` },
        });
        const meData = await meRes.json();
        setCompanyId(meData.company_id);
        if (meData.email) {
          setEmail(meData.email);
          setCompanyName(meData.company_name);
        }

        const prodRes = await fetch(
          `http://localhost:8000/get-products/${meData.company_id}`,
          { headers: { Authorization: `Bearer ${token}` } }
        );
        const prodData = await prodRes.json();
        console.log("Backend Products data:", prodData);
        if (!prodData.products || prodData.products.length === 0) {
          setShowConfirmWebsite(true);
          console.log("No products found, prompting for website confirmation.");
          return;
        }
        setProducts(prodData.products);

        console.log("Products fetched length:", prodData.products.length);

        // Auto-select if only one product
        if (prodData.products.length === 1) {
          setSelectedProductId(prodData.products[0].id);
          console.log("Auto selection of product ID")
        }
      } catch (err) {
        setStatusMsg("Error fetching company or products.");
      }
    };
    fetchCompanyAndProducts();
  }, [token]);

  const fetchProductData = useCallback(async () => {
    if (!selectedProductId) return;
    setStatusMsg("Loading product data...");
    try {
      const res = await fetch(
        `${API_BASE}/get-product-capabilities/${selectedProductId}`,
        { headers: { Authorization: `Bearer ${token}` } }
      );
      if (!res.ok) {
        throw new Error("Failed to fetch product data.");
      }
      const data = await res.json();
      console.log("Backend Product data:", data);
      if (!data || !data.product_node_id) {
        setProductData(null);
        setStatusMsg("No product data found.");
        setShowConfirmWebsite(true);
        return;
      }
      setProductData(data);
      setStatusMsg("");
    } catch (err) {
      setStatusMsg("Error fetching product data.");
      setProductData(null);
    }
  }, [selectedProductId, token]);

  useEffect(() => {
    if (selectedProductId) {
      fetchProductData();
    }
  }, [selectedProductId, fetchProductData]);

  useEffect(() => {
    if (!productData) return;
    setSummaryDraft(productData.summary || "");
    const drafts: Record<string, { name: string; description: string }> = {};
    (productData.capabilities || []).forEach((cap) => {
      const capId = cap.id || cap.name;
      drafts[capId] = {
        name: cap.name,
        description: cap.description || "",
      };
    });
    setCapabilityDrafts(drafts);
  }, [productData]);

  if (showConfirmWebsite) {
    return renderConfirmWebsite();
  }

  const summaryHasChanges =
    Boolean(productData) && summaryDraft !== (productData?.summary || "");
  const newCapabilityHasContent =
    newCapability.name.trim() !== "" || newCapability.description.trim() !== "";
  const hasPendingEdits =
    summaryHasChanges || changedCapabilityIds.length > 0 || newCapabilityHasContent;
  const newCapabilityNameMissing =
    isNewCapabilityOpen &&
    newCapability.description.trim() !== "" &&
    newCapability.name.trim() === "";
  const isEditingSummary = editingCapabilityId === "summary";
  const summaryCardBackground = isEditingSummary
    ? alpha(theme.palette.primary.main, 0.08)
    : theme.palette.background.paper;
  const summaryCardBorder = isEditingSummary ? theme.palette.primary.main : theme.palette.divider;

  return (
    <Box sx={{ p: { xs: 2, md: 4 }, display: "flex", flexDirection: "column", gap: 4 }}>
      <Stack
        direction={{ xs: "column", md: "row" }}
        spacing={2}
        alignItems="flex-end"
        justifyContent="space-between"
      >
        <Box>
          <Typography variant="h5" fontWeight={600}>
            Product Overview
          </Typography>
          <Typography variant="body2" color="text.secondary">
            Tap a card to edit summary or capabilities.
          </Typography>
        </Box>
        {(hasPendingEdits || saveAllStatus) && (
          <Stack spacing={0.5} alignItems="flex-end">
            {hasPendingEdits && (
              <Button
                variant="contained"
                color="primary"
                onClick={handleSaveAll}
                disabled={isSavingAll}
              >
                {isSavingAll ? "Saving changes..." : "Save All Changes"}
              </Button>
            )}
            {saveAllStatus && (
              <Typography variant="caption" color="text.secondary">
                {saveAllStatus}
              </Typography>
            )}
          </Stack>
        )}
      </Stack>

      {products.length > 1 && (
        <TextField
          select
          label="Select Product"
          size="small"
          value={selectedProductId || ""}
          onChange={(e) => setSelectedProductId(e.target.value)}
          sx={{ maxWidth: 320 }}
        >
          <MenuItem value="" disabled>
            Select a product
          </MenuItem>
          {products.map((p) => (
            <MenuItem key={p.id} value={p.id}>
              {p.name}
            </MenuItem>
          ))}
        </TextField>
      )}

      {statusMsg && (
        <Typography variant="body2" color="error">
          {statusMsg}
        </Typography>
      )}

      {productData && (
        <Stack spacing={4}>
          <ValuePropCard
            product_node_id={productData.product_node_id}
            summary={productData.summary}
            plg_flag={productData.plg_flag}
          />

          <Card
            variant="outlined"
            sx={{
              borderRadius: 3,
              cursor: "pointer",
              borderColor: summaryCardBorder,
              backgroundColor: summaryCardBackground,
              transition: "0.2s",
              boxShadow: isEditingSummary ? theme.shadows[2] : theme.shadows[0],
            }}
            onClick={() => setEditingCapabilityId("summary")}
          >
            <CardHeader
              title="Value Prop Summary"
              subheader="Tap to edit"
              titleTypographyProps={{ variant: "subtitle1", color: "primary.main", fontWeight: 600 }}
              subheaderTypographyProps={{ variant: "caption", color: "text.secondary", letterSpacing: 0.5 }}
            />
            <CardContent sx={{ pt: 0, minHeight: 140 }}>
              {editingCapabilityId === "summary" ? (
                <TextField
                  multiline
                  minRows={3}
                  value={summaryDraft}
                  onChange={(e) => setSummaryDraft(e.target.value)}
                  placeholder="Describe your product value prop..."
                  fullWidth
                  variant="standard"
                  InputProps={{ disableUnderline: true }}
                  autoFocus
                  onClick={(e) => e.stopPropagation()}
                />
              ) : (
                <Typography variant="body2" color="text.secondary">
                  {summaryDraft || "Describe your product value prop..."}
                </Typography>
              )}
            </CardContent>
          </Card>

          <Box>
            <Stack direction="row" alignItems="center" justifyContent="space-between" mb={2}>
              <Typography variant="h6" color="primary" fontWeight={600}>
                Capabilities
              </Typography>
              <Typography variant="caption" color="text.secondary" letterSpacing={0.5}>
                Click a card to edit
              </Typography>
            </Stack>
            <Grid container spacing={2}>
              {(productData.capabilities || []).length === 0 && (
                <Grid item xs={12}>
                  <Typography variant="body2" color="text.secondary">
                    No capabilities found yet. Add one below to get started.
                  </Typography>
                </Grid>
              )}
              {(productData.capabilities || []).map((cap) => {
                const capId = cap.id || cap.name;
                const isEditing = editingCapabilityId === capId;
                const draft = capabilityDrafts[capId];
                const cardBackground = isEditing
                  ? alpha(theme.palette.primary.main, 0.08)
                  : theme.palette.background.paper;
                const cardBorder = isEditing ? theme.palette.primary.main : theme.palette.divider;
                return (
                  <Grid item xs={12} md={6} key={capId}>
                    <Card
                      variant="outlined"
                      sx={{
                        minHeight: 200,
                        borderRadius: 3,
                        cursor: "pointer",
                        borderColor: cardBorder,
                        backgroundColor: cardBackground,
                        transition: "0.2s",
                        boxShadow: isEditing ? theme.shadows[2] : theme.shadows[0],
                      }}
                      onClick={() => setEditingCapabilityId(capId)}
                    >
                      <CardContent sx={{ p: 3, minHeight: 200 }}>
                        <Typography variant="caption" color="text.secondary">
                          Coreness {formatMetricValue(cap.coreness)} · Centrality{" "}
                          {formatMetricValue(cap.centrality)}
                        </Typography>
                        {isEditing ? (
                          <Stack
                            spacing={1.5}
                            mt={1}
                            onClick={(e) => e.stopPropagation()}
                          >
                            <TextField
                              label="Name"
                              size="small"
                              variant="standard"
                              InputProps={{ disableUnderline: true }}
                              value={draft?.name ?? cap.name}
                              onChange={(e) =>
                                handleCapabilityFieldChange(capId, "name", e.target.value)
                              }
                              fullWidth
                              autoFocus
                            />
                            <TextField
                              label="Description"
                              size="small"
                              variant="standard"
                              InputProps={{ disableUnderline: true }}
                              multiline
                              minRows={3}
                              value={draft?.description ?? cap.description ?? ""}
                              onChange={(e) =>
                                handleCapabilityFieldChange(capId, "description", e.target.value)
                              }
                              fullWidth
                            />
                            <Typography variant="caption" color="text.secondary">
                              Edits are saved together via the button above.
                            </Typography>
                          </Stack>
                        ) : (
                          <Stack spacing={1} mt={2}>
                            <Typography variant="subtitle1" fontWeight={600} color="text.primary">
                              {draft?.name || cap.name}
                            </Typography>
                            <Typography
                              variant="body2"
                              color="text.secondary"
                              sx={{ minHeight: 60 }}
                            >
                              {draft?.description ||
                                cap.description ||
                                "Add a description to highlight what this capability does."}
                            </Typography>
                          </Stack>
                        )}
                      </CardContent>
                    </Card>
                  </Grid>
                );
              })}
              <Grid item xs={12} md={6}>
                <Card
                  variant="outlined"
                  sx={{
                    minHeight: 200,
                    borderRadius: 3,
                    cursor: "pointer",
                    borderColor: isNewCapabilityOpen ? "primary.main" : "divider",
                    backgroundColor: isNewCapabilityOpen ? "background.paper" : "background.default",
                    transition: "0.2s",
                  }}
                  onClick={() => setIsNewCapabilityOpen(true)}
                >
                  {!isNewCapabilityOpen ? (
                    <CardContent
                      sx={{
                        display: "flex",
                        flexDirection: "column",
                        alignItems: "center",
                        justifyContent: "center",
                        gap: 1,
                        minHeight: 180,
                      }}
                    >
                      <AddIcon fontSize="large" color="disabled" />
                      <Typography variant="button" color="text.secondary" letterSpacing={1.2}>
                        Add capability
                      </Typography>
                    </CardContent>
                  ) : (
                    <CardContent
                      sx={{ display: "flex", flexDirection: "column", gap: 2 }}
                      onClick={(e) => e.stopPropagation()}
                    >
                      <Stack direction="row" justifyContent="space-between" alignItems="center">
                        <Typography variant="caption" color="text.secondary" letterSpacing={0.7}>
                          New capability
                        </Typography>
                        <Button
                          size="small"
                          onClick={(e) => {
                            e.stopPropagation();
                            setIsNewCapabilityOpen(false);
                            setNewCapability({ name: "", description: "" });
                          }}
                        >
                          Cancel
                        </Button>
                      </Stack>
                      <TextField
                        label="Name"
                        size="small"
                        value={newCapability.name}
                        onChange={(e) =>
                          setNewCapability((prev) => ({ ...prev, name: e.target.value }))
                        }
                        fullWidth
                        error={newCapabilityNameMissing}
                        helperText={
                          newCapabilityNameMissing ? "Provide a name before saving." : undefined
                        }
                      />
                      <TextField
                        label="Description"
                        size="small"
                        multiline
                        minRows={3}
                        value={newCapability.description}
                        onChange={(e) =>
                          setNewCapability((prev) => ({ ...prev, description: e.target.value }))
                        }
                        fullWidth
                      />
                      <Typography variant="caption" color="text.secondary">
                        Click Save All Changes to persist this entry.
                      </Typography>
                    </CardContent>
                  )}
                </Card>
              </Grid>
            </Grid>
          </Box>
        </Stack>
      )}
    </Box>
  );
};

export default ValueProposition;
