import { ChangeEvent, useEffect, useMemo, useState } from "react";
import {
  Alert,
  Autocomplete,
  Box,
  Button,
  Chip,
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
import Stack from "@mui/material/Stack";
import AssetsLibrary from "../components/AssetsLibrary";
import CategoryLibrary from "../components/CategoryLibrary";
import type {
  ArsenalAssetCategoryMeta,
  ArsenalChannelCategoryMeta,
} from "../types/apiContracts";

type EnumOption = { value: string; label: string };

type AssetOptionGroups = {
  category: EnumOption[];
  content_type: EnumOption[];
  time_to_consume: EnumOption[];
  depth: EnumOption[];
  sales_call_stage: EnumOption[];
  belief_stage: EnumOption[];
};

type ChannelOptionGroups = {
  channel_type: EnumOption[];
  delivery_mode: EnumOption[];
};

const BELIEF_STAGE_OPTIONS: EnumOption[] = [
  { value: "Problem Realization", label: "Problem Realization" },
  { value: "Pain Realization", label: "Pain Realization" },
  { value: "Resolution Discovery", label: "Resolution Discovery" },
  { value: "Execution Guidance", label: "Execution Guidance" },
];

const ORG_CONVERSION_OPTIONS: EnumOption[] = [
  { value: "early", label: "Early Cycle" },
  { value: "mid", label: "Mid Cycle" },
  { value: "late", label: "Late Cycle" },
];

const dedupeStrings = (values: Array<string | null | undefined>): string[] => {
  const seen = new Set<string>();
  const out: string[] = [];
  values.forEach((value) => {
    if (!value) return;
    const trimmed = value.trim();
    if (!trimmed || seen.has(trimmed)) return;
    seen.add(trimmed);
    out.push(trimmed);
  });
  return out;
};

const formatOptionLabel = (value: string) => {
  const cleaned = value.replace(/[_\s]+/g, " ").trim();
  if (!cleaned) return value;
  return cleaned
    .split(" ")
    .filter(Boolean)
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
};

const buildLabelMap = (options: EnumOption[]) => {
  return options.reduce<Record<string, string>>((acc, option) => {
    const key = option.value.toLowerCase();
    acc[key] = option.label || formatOptionLabel(option.value);
    return acc;
  }, {});
};

const labelFromMap = (map: Record<string, string>, value: string) => {
  if (!value) return "";
  return map[value.toLowerCase()] || formatOptionLabel(value);
};

type AssetFormState = {
  name: string;
  category: string;
  content_type: string;
  time_to_consume: string;
  depth: string;
  description: string;
  notes: string;
  callStage: string;
  targetPersonas: string[];
  targetAccountSegments: string[];
  targetStages: string[];
  targetConcerns: string[];
  orgConversionMaturity: string;
  typicalChannels: string[];
  channelId: string;
  activities: string[];
};

type ChannelFormState = {
  name: string;
  channel_type: string;
  delivery_mode: string;
  reach_score_estimate: string;
  notes: string;
  targetPersonas: string[];
  targetAccountSegments: string[];
  targetStages: string[];
  orgConversionMaturity: string;
  typicalAssets: string[];
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
  belief_stage: BELIEF_STAGE_OPTIONS,
};

const emptyChannelOptions: ChannelOptionGroups = {
  channel_type: [],
  delivery_mode: [],
};

const createDefaultAssetForm = (): AssetFormState => ({
  name: "",
  category: "",
  content_type: "",
  time_to_consume: "",
  depth: "",
  description: "",
  notes: "",
  callStage: "",
  targetPersonas: [],
  targetAccountSegments: [],
  targetStages: [],
  targetConcerns: [],
  orgConversionMaturity: "",
  typicalChannels: [],
  channelId: "",
  activities: [],
});

const createDefaultChannelForm = (): ChannelFormState => ({
  name: "",
  channel_type: "",
  delivery_mode: "",
  reach_score_estimate: "",
  notes: "",
  targetPersonas: [],
  targetAccountSegments: [],
  targetStages: [],
  orgConversionMaturity: "",
  typicalAssets: [],
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
  const [assetCategories, setAssetCategories] = useState<ArsenalAssetCategoryMeta[]>([]);
  const [channelCategories, setChannelCategories] = useState<ArsenalChannelCategoryMeta[]>([]);

  const [assetOptions, setAssetOptions] = useState<AssetOptionGroups>(emptyAssetOptions);
  const [channelOptions, setChannelOptions] = useState<ChannelOptionGroups>(emptyChannelOptions);

  const [editingAsset, setEditingAsset] = useState<any | null>(null);
  const [editingChannel, setEditingChannel] = useState<any | null>(null);
  const [assetModalMode, setAssetModalMode] = useState<"edit" | "create">("edit");
  const [channelModalMode, setChannelModalMode] = useState<"edit" | "create">("edit");
  const [assetForm, setAssetForm] = useState<AssetFormState>(createDefaultAssetForm());
  const [channelForm, setChannelForm] = useState<ChannelFormState>(createDefaultChannelForm());
  const [assetSaving, setAssetSaving] = useState(false);
  const [channelSaving, setChannelSaving] = useState(false);

  const [snack, setSnack] = useState<SnackState>({
    open: false,
    message: "",
    severity: "success",
  });
  const [assetCategoryInput, setAssetCategoryInput] = useState("");
  const [assetContentTypeInput, setAssetContentTypeInput] = useState("");
  const [channelTypeInput, setChannelTypeInput] = useState("");
  const [deliveryModeInput, setDeliveryModeInput] = useState("");

  const assetCategoryValues = useMemo(
    () => assetOptions.category.map((option) => option.value),
    [assetOptions.category]
  );
  const assetCategoryLabelMap = useMemo(
    () => buildLabelMap(assetOptions.category),
    [assetOptions.category]
  );
  const assetContentTypeValues = useMemo(
    () => assetOptions.content_type.map((option) => option.value),
    [assetOptions.content_type]
  );
  const assetContentTypeLabelMap = useMemo(
    () => buildLabelMap(assetOptions.content_type),
    [assetOptions.content_type]
  );
  const channelTypeValues = useMemo(
    () => channelOptions.channel_type.map((option) => option.value),
    [channelOptions.channel_type]
  );
  const channelTypeLabelMap = useMemo(
    () => buildLabelMap(channelOptions.channel_type),
    [channelOptions.channel_type]
  );
  const deliveryModeValues = useMemo(
    () => channelOptions.delivery_mode.map((option) => option.value),
    [channelOptions.delivery_mode]
  );
  const deliveryModeLabelMap = useMemo(
    () => buildLabelMap(channelOptions.delivery_mode),
    [channelOptions.delivery_mode]
  );

  const assetStageOptions = useMemo(
    () =>
      (assetOptions.belief_stage && assetOptions.belief_stage.length > 0
        ? assetOptions.belief_stage
        : BELIEF_STAGE_OPTIONS
      ).map((option) => option.label || option.value),
    [assetOptions.belief_stage]
  );

  const assetPersonaOptions = useMemo(() => {
    const usagePersonas =
      (editingAsset?.usage?.personas || []).map(
        (persona: { label?: string | null; id: string }) => persona.label || persona.id
      ) || [];
    return dedupeStrings([...assetForm.targetPersonas, ...usagePersonas]);
  }, [assetForm.targetPersonas, editingAsset]);

  const assetSegmentOptions = useMemo(() => {
    const usageSegments =
      (editingAsset?.usage?.segments || []).map((segment: { label: string }) => segment.label) || [];
    return dedupeStrings([...assetForm.targetAccountSegments, ...usageSegments]);
  }, [assetForm.targetAccountSegments, editingAsset]);

  const channelStageOptions = assetStageOptions;
  const isCreatingAsset = Boolean(editingAsset && assetModalMode === "create");
  const isCreatingChannel = Boolean(editingChannel && channelModalMode === "create");

  useEffect(() => {
    setAssetCategoryInput(
      assetForm.category
        ? labelFromMap(assetCategoryLabelMap, assetForm.category)
        : ""
    );
  }, [assetForm.category, assetCategoryLabelMap]);

  useEffect(() => {
    setAssetContentTypeInput(
      assetForm.content_type
        ? labelFromMap(assetContentTypeLabelMap, assetForm.content_type)
        : ""
    );
  }, [assetForm.content_type, assetContentTypeLabelMap]);

  useEffect(() => {
    setChannelTypeInput(
      channelForm.channel_type
        ? labelFromMap(channelTypeLabelMap, channelForm.channel_type)
        : ""
    );
  }, [channelForm.channel_type, channelTypeLabelMap]);

  useEffect(() => {
    setDeliveryModeInput(
      channelForm.delivery_mode
        ? labelFromMap(deliveryModeLabelMap, channelForm.delivery_mode)
        : ""
    );
  }, [channelForm.delivery_mode, deliveryModeLabelMap]);

  const assetUsageStageLabels = useMemo(
    () => (editingAsset?.usage?.stages || []).map((stage: { label: string }) => stage.label) || [],
    [editingAsset]
  );
  const assetUsageConcernLabels = useMemo(
    () => (editingAsset?.usage?.concerns || []).map((concern: { label: string }) => concern.label) || [],
    [editingAsset]
  );
  const assetUsagePersonaMap = useMemo(() => {
    const map = new Map<string, string>();
    (editingAsset?.usage?.personas || []).forEach(
      (persona: { id: string; label?: string | null }) => map.set(persona.id, persona.label || persona.id)
    );
    return map;
  }, [editingAsset]);

  const assetConcernOptions = useMemo(() => {
    return dedupeStrings([...assetForm.targetConcerns, ...assetUsageConcernLabels]);
  }, [assetForm.targetConcerns, assetUsageConcernLabels]);

  const channelPersonaOptions = useMemo(() => {
    const usagePersonas =
      (editingChannel?.usage?.personas || []).map(
        (persona: { label?: string | null; id: string }) => persona.label || persona.id
      ) || [];
    return dedupeStrings([...channelForm.targetPersonas, ...usagePersonas]);
  }, [channelForm.targetPersonas, editingChannel]);

  const channelSegmentOptions = useMemo(() => {
    const usageSegments =
      (editingChannel?.usage?.segments || []).map((segment: { label: string }) => segment.label) || [];
    return dedupeStrings([...channelForm.targetAccountSegments, ...usageSegments]);
  }, [channelForm.targetAccountSegments, editingChannel]);

  const channelUsageStageLabels = useMemo(
    () => (editingChannel?.usage?.stages || []).map((stage: { label: string }) => stage.label) || [],
    [editingChannel]
  );
  const channelUsagePersonaMap = useMemo(() => {
    const map = new Map<string, string>();
    (editingChannel?.usage?.personas || []).forEach(
      (persona: { id: string; label?: string | null }) => map.set(persona.id, persona.label || persona.id)
    );
    return map;
  }, [editingChannel]);

  const channelLabelMap = useMemo(() => {
    const map: Record<string, string> = {};
    arsenalChannels.forEach((channel) => {
      map[channel.id] = channel.name || channel.slug || channel.id;
    });
    return map;
  }, [arsenalChannels]);

  const channelLookup = useMemo(() => {
    const map: Record<string, { channel_category_meta?: { key?: string } }> = {};
    arsenalChannels.forEach((channel) => {
      if (channel.id) {
        map[channel.id] = channel;
      }
    });
    return map;
  }, [arsenalChannels]);

  const channelOptionsForSelect = useMemo(() => {
    const base = arsenalChannels.map((channel) => channel.id);
    const seen = new Set(base);
    assetForm.typicalChannels.forEach((id) => {
      if (!seen.has(id)) {
        base.push(id);
        seen.add(id);
      }
    });
    return base;
  }, [arsenalChannels, assetForm.typicalChannels]);

  const assetLabelMap = useMemo(() => {
    const map: Record<string, string> = {};
    arsenalAssets.forEach((asset) => {
      map[asset.id] = asset.name || asset.slug || asset.id;
    });
    return map;
  }, [arsenalAssets]);

  const assetOptionsForSelect = useMemo(() => {
    const base = arsenalAssets.map((asset) => asset.id);
    const seen = new Set(base);
    channelForm.typicalAssets.forEach((id) => {
      if (!seen.has(id)) {
        base.push(id);
        seen.add(id);
      }
    });
    return base;
  }, [arsenalAssets, channelForm.typicalAssets]);

  const activityOptions = useMemo(() => {
    const existing = assetForm.activities || [];
    const derived = Array.isArray(editingAsset?.activity_labels)
      ? (editingAsset?.activity_labels as string[])
      : [];
    const autoActivity =
      editingAsset?.derived_metadata?.activity?.label &&
      !derived.includes(editingAsset.derived_metadata.activity.label)
        ? [editingAsset.derived_metadata.activity.label]
        : [];
    return dedupeStrings([...existing, ...derived, ...autoActivity]);
  }, [assetForm.activities, editingAsset]);

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
          belief_stage: assetOpts.belief_stage ?? BELIEF_STAGE_OPTIONS,
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
        const assetCategoryEntries = data.arsenal.asset_categories ?? [];
        const channelCategoryEntries = data.arsenal.channel_categories ?? [];
        setArsenalAssets(assets);
        setArsenalChannels(channels);
        setAssetCategories(assetCategoryEntries);
        setChannelCategories(channelCategoryEntries);
        recomputeMetaGaps(assets, channels);
      } catch (err) {
        setStatusMsg("Error fetching Arsenal Library.");
      }
    };
    fetchArsenal();
  }, [selectedProductId, token]);

  useEffect(() => {
    if (editingAsset) {
      const primaryChannelId =
        editingAsset.primary_channel?.id ||
        editingAsset.typical_channels?.[0]?.id ||
        "";
      const derivedActivity =
        editingAsset.activity_labels && editingAsset.activity_labels.length
          ? editingAsset.activity_labels
          : editingAsset.derived_metadata?.activity?.label
          ? [editingAsset.derived_metadata.activity.label]
          : [];
      const derivedName =
        editingAsset.derived_metadata?.asset_name?.label ||
        editingAsset.name ||
        "";
      setAssetForm({
        name: derivedName,
        category: editingAsset.category || "",
        content_type: editingAsset.content_type || "",
        time_to_consume: editingAsset.time_to_consume || "",
        depth: editingAsset.depth || "",
        description: editingAsset.description || "",
        notes: editingAsset.notes || "",
        callStage: editingAsset.call_stage || "",
        targetPersonas: dedupeStrings(
          (editingAsset.target_personas || []).map(
            (value: string) => assetUsagePersonaMap.get(value) || value
          )
        ),
        targetAccountSegments: editingAsset.target_account_segments || [],
        targetStages: dedupeStrings([
          ...(editingAsset.target_belief_stages || []),
          ...assetUsageStageLabels,
        ]),
        targetConcerns: dedupeStrings([
          ...(editingAsset.target_concerns || []),
          ...assetUsageConcernLabels,
        ]),
        orgConversionMaturity:
          editingAsset.org_conversion_maturity?.code ||
          editingAsset.funnel_stage?.code ||
          "",
        typicalChannels: (editingAsset.typical_channels || []).map(
          (entry: { id: string }) => entry.id
        ),
        channelId: primaryChannelId || "",
        activities: dedupeStrings(derivedActivity),
      });
    } else {
      setAssetForm(createDefaultAssetForm());
    }
  }, [editingAsset, assetUsagePersonaMap, assetUsageStageLabels, assetUsageConcernLabels]);

  useEffect(() => {
    if (editingChannel) {
      setChannelForm({
        name: editingChannel.name || "",
        channel_type: editingChannel.channel_type || "",
        delivery_mode: editingChannel.delivery_mode || "",
        reach_score_estimate:
          editingChannel.reach_score_estimate === null ||
          editingChannel.reach_score_estimate === undefined
            ? ""
            : String(editingChannel.reach_score_estimate),
        notes: editingChannel.notes || "",
        targetPersonas: dedupeStrings(
          (editingChannel.target_personas || []).map(
            (value: string) => channelUsagePersonaMap.get(value) || value
          )
        ),
        targetAccountSegments: editingChannel.target_account_segments || [],
        targetStages: dedupeStrings([
          ...(editingChannel.target_belief_stages || []),
          ...channelUsageStageLabels,
        ]),
        orgConversionMaturity:
          editingChannel.org_conversion_maturity?.code ||
          editingChannel.funnel_stage?.code ||
          "",
        typicalAssets: (editingChannel.typical_assets || []).map(
          (entry: { id: string }) => entry.id
        ),
      });
    } else {
      setChannelForm(createDefaultChannelForm());
    }
  }, [
    editingChannel,
    channelUsagePersonaMap,
    channelUsageStageLabels,
  ]);

  const channelCategoryAssets = useMemo(() => {
    const map: Record<string, Array<{ assetId: string; score: number }>> = {};
    arsenalAssets.forEach((asset) => {
      const score = asset.usage?.total_engagements || 0;
      const categoryKeys = (asset.typical_channels || [])
        .map((entry) => channelLookup[entry.id])
        .filter((channel): channel is { channel_category_meta?: { key?: string } } => Boolean(channel))
        .map((channel) => channel.channel_category_meta?.key)
        .filter((key): key is string => Boolean(key));
      const uniqueKeys = Array.from(new Set(categoryKeys));
      uniqueKeys.forEach((key) => {
        const bucket = map[key] || [];
        bucket.push({ assetId: asset.id, score });
        map[key] = bucket;
      });
    });
    return map;
  }, [arsenalAssets, channelLookup]);

  const handleAssetFieldChange = (
    field: keyof AssetFormState,
    value: string,
    opts?: { displayLabel?: string }
  ) => {
    const enumFields: Array<keyof AssetFormState> = [
      "category",
      "content_type",
      "time_to_consume",
      "depth",
    ];
    let normalized = value;
    if (enumFields.includes(field)) {
      normalized = value.trim();
    }
    setAssetForm((prev) => {
      const next = { ...prev, [field]: normalized };
      if (field === "category" && normalized !== "sales_call" && prev.callStage) {
        next.callStage = "";
      }
      return next;
    });
    if (enumFields.includes(field) && normalized) {
      setAssetOptions((prev) => {
        const optionKeyMap: Record<
          "category" | "content_type" | "time_to_consume" | "depth",
          keyof AssetOptionGroups
        > = {
          category: "category",
          content_type: "content_type",
          time_to_consume: "time_to_consume",
          depth: "depth",
        };
        const optionKey =
          optionKeyMap[field as "category" | "content_type" | "time_to_consume" | "depth"];
        const list = prev[optionKey] || [];
        const exists = list.some(
          (option) => option.value.toLowerCase() === normalized.toLowerCase()
        );
        if (exists) return prev;
        const updatedList = [
          ...list,
          {
            value: normalized,
            label: opts?.displayLabel?.trim() || formatOptionLabel(normalized),
          },
        ];
        return {
          ...prev,
          [optionKey]: updatedList,
        };
      });
    }
  };

  const handleAssetListChange = (
    field: "targetPersonas" | "targetAccountSegments" | "targetStages" | "targetConcerns",
    values: string[]
  ) => {
    setAssetForm((prev) => ({
      ...prev,
      [field]: dedupeStrings(values.map((value) => value.trim()).filter(Boolean)),
    }));
  };

  const handleAssetIdListChange = (field: "typicalChannels", values: string[]) => {
    setAssetForm((prev) => ({
      ...prev,
      [field]: values.map((value) => value.trim()).filter(Boolean),
    }));
  };

  const handleChannelFieldChange = (
    field: keyof ChannelFormState,
    value: string,
    opts?: { displayLabel?: string }
  ) => {
    const enumFields: Array<keyof ChannelFormState> = ["channel_type", "delivery_mode"];
    let normalized = value;
    if (enumFields.includes(field)) {
      normalized = value.trim();
    } else if (field === "name") {
      normalized = value.trim();
    }
    setChannelForm((prev) => ({ ...prev, [field]: normalized }));
    if (enumFields.includes(field) && normalized) {
      setChannelOptions((prev) => {
        const optionKeyMap: Record<
          "channel_type" | "delivery_mode",
          keyof ChannelOptionGroups
        > = {
          channel_type: "channel_type",
          delivery_mode: "delivery_mode",
        };
        const optionKey = optionKeyMap[field as "channel_type" | "delivery_mode"];
        const list = prev[optionKey] || [];
        const exists = list.some(
          (option) => option.value.toLowerCase() === normalized.toLowerCase()
        );
        if (exists) return prev;
        const updatedList = [
          ...list,
          {
            value: normalized,
            label: opts?.displayLabel?.trim() || formatOptionLabel(normalized),
          },
        ];
        return {
          ...prev,
          [optionKey]: updatedList,
        };
      });
    }
  };

  const handleChannelListChange = (
    field: "targetPersonas" | "targetAccountSegments" | "targetStages",
    values: string[]
  ) => {
    setChannelForm((prev) => ({
      ...prev,
      [field]: dedupeStrings(values.map((value) => value.trim()).filter(Boolean)),
    }));
  };

  const handleChannelIdListChange = (field: "typicalAssets", values: string[]) => {
    setChannelForm((prev) => ({
      ...prev,
      [field]: values.map((value) => value.trim()).filter(Boolean),
    }));
  };

  const handleCreateAssetClick = () => {
    if (!selectedProductId) {
      setSnack({
        open: true,
        message: "Select a product before creating assets.",
        severity: "error",
      });
      return;
    }
    setAssetModalMode("create");
    setEditingAsset({
      id: "__new__",
      product_id: selectedProductId,
      usage: {},
      __isNew: true,
    });
  };

  const handleCreateChannelClick = () => {
    if (!selectedProductId) {
      setSnack({
        open: true,
        message: "Select a product before creating channels.",
        severity: "error",
      });
      return;
    }
    setChannelModalMode("create");
    setEditingChannel({
      id: "__new__",
      product_id: selectedProductId,
      usage: {},
      __isNew: true,
    });
  };

  const handleAssetSave = async () => {
    if (!editingAsset || !token) return;
    if (assetModalMode === "create" && !selectedProductId) {
      setSnack({
        open: true,
        message: "Select a product before creating assets.",
        severity: "error",
      });
      return;
    }
    setAssetSaving(true);
    try {
      const payload = {
        name: assetForm.name || null,
        category: assetForm.category || null,
        content_type: assetForm.content_type || null,
        time_to_consume: assetForm.time_to_consume || null,
        depth: assetForm.depth || null,
        description: assetForm.description,
        notes: assetForm.notes,
        call_stage: assetForm.callStage || null,
        target_personas: assetForm.targetPersonas,
        target_account_segments: assetForm.targetAccountSegments,
        target_belief_stages: assetForm.targetStages,
        target_concerns: assetForm.targetConcerns,
        org_conversion_maturity: assetForm.orgConversionMaturity || null,
        typical_channels: assetForm.typicalChannels,
        primary_channel_id: assetForm.channelId || null,
        activity_labels: assetForm.activities,
      };
      const endpoint =
        assetModalMode === "create"
          ? "http://localhost:8000/arsenal/assets"
          : `http://localhost:8000/arsenal/assets/${editingAsset.id}`;
      const method = assetModalMode === "create" ? "POST" : "PATCH";
      const res = await fetch(endpoint, {
        method,
        headers: {
          Authorization: `Bearer ${token}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify(
          assetModalMode === "create"
            ? { product_id: selectedProductId, ...payload }
            : payload
        ),
      });
      const data = await res.json().catch(() => null);
      if (!res.ok) {
        throw new Error((data && data.detail) || "Failed to update asset metadata");
      }
      const updated = data.asset;
      setArsenalAssets((prev) => {
        let next: any[];
        if (assetModalMode === "create") {
          next = [updated, ...prev];
        } else {
          next = prev.map((asset) => (asset.id === updated.id ? updated : asset));
        }
        recomputeMetaGaps(next, arsenalChannels);
        return next;
      });
      setSnack({
        open: true,
        message:
          assetModalMode === "create"
            ? "Asset created"
            : "Asset metadata updated",
        severity: "success",
      });
      setAssetModalMode("edit");
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
    if (channelModalMode === "create" && !selectedProductId) {
      setSnack({
        open: true,
        message: "Select a product before creating channels.",
        severity: "error",
      });
      return;
    }
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

      const payload = {
        name: channelForm.name || null,
        channel_type: channelForm.channel_type || null,
        delivery_mode: channelForm.delivery_mode || null,
        reach_score_estimate: reachPayload,
        notes: channelForm.notes,
        target_personas: channelForm.targetPersonas,
        target_account_segments: channelForm.targetAccountSegments,
        target_belief_stages: channelForm.targetStages,
        target_concerns: channelForm.targetConcerns,
        org_conversion_maturity: channelForm.orgConversionMaturity || null,
        typical_assets: channelForm.typicalAssets,
      };

      const endpoint =
        channelModalMode === "create"
          ? "http://localhost:8000/arsenal/channels"
          : `http://localhost:8000/arsenal/channels/${editingChannel.id}`;
      const method = channelModalMode === "create" ? "POST" : "PATCH";
      const res = await fetch(endpoint, {
        method,
        headers: {
          Authorization: `Bearer ${token}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify(
          channelModalMode === "create"
            ? { product_id: selectedProductId, ...payload }
            : payload
        ),
      });
      const data = await res.json().catch(() => null);
      if (!res.ok) {
        throw new Error((data && data.detail) || "Failed to update channel metadata");
      }
      const updated = data.channel;
      setArsenalChannels((prev) => {
        let next: any[];
        if (channelModalMode === "create") {
          next = [updated, ...prev];
        } else {
          next = prev.map((channel) => (channel.id === updated.id ? updated : channel));
        }
        recomputeMetaGaps(arsenalAssets, next);
        return next;
      });
      setSnack({
        open: true,
        message:
          channelModalMode === "create"
            ? "Channel created"
            : "Channel metadata updated",
        severity: "success",
      });
      setChannelModalMode("edit");
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

  const handleAssetApprove = async (asset: any) => {
    if (!token) return;
    try {
      const res = await fetch(`http://localhost:8000/arsenal/assets/${asset.id}/approve`, {
        method: "POST",
        headers: {
          Authorization: `Bearer ${token}`,
          "Content-Type": "application/json",
        },
      });
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      const updated = data.asset;
      setArsenalAssets((prev) => {
        const next = prev.map((item) => (item.id === updated.id ? updated : item));
        recomputeMetaGaps(next, arsenalChannels);
        return next;
      });
      if (editingAsset?.id === updated.id) {
        setEditingAsset(updated);
      }
      setSnack({
        open: true,
        message: "Asset approved",
        severity: "success",
      });
    } catch (err) {
      console.error("Asset approval failed", err);
      setSnack({ open: true, message: "Unable to approve asset", severity: "error" });
    }
  };

  const handleChannelApprove = async (channel: any) => {
    if (!token) return;
    try {
      const res = await fetch(`http://localhost:8000/arsenal/channels/${channel.id}/approve`, {
        method: "POST",
        headers: {
          Authorization: `Bearer ${token}`,
          "Content-Type": "application/json",
        },
      });
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      const updated = data.channel;
      setArsenalChannels((prev) => {
        const next = prev.map((item) => (item.id === updated.id ? updated : item));
        recomputeMetaGaps(arsenalAssets, next);
        return next;
      });
      if (editingChannel?.id === updated.id) {
        setEditingChannel(updated);
      }
      setSnack({
        open: true,
        message: "Channel approved",
        severity: "success",
      });
    } catch (err) {
      console.error("Channel approval failed", err);
      setSnack({ open: true, message: "Unable to approve channel", severity: "error" });
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

      <Box
        sx={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          gap: 2,
          flexWrap: "wrap",
          mb: 3,
        }}
      >
        <Tabs
          value={tabIndex}
          onChange={(_, newValue) => setTabIndex(newValue)}
          indicatorColor="primary"
          textColor="primary"
          variant="fullWidth"
          sx={{ minWidth: 240 }}
        >
          <Tab label="Arsenal Repo" />
          <Tab label="Asset Categories" />
          <Tab label="Channel Categories" />
        </Tabs>
        <Stack direction="row" spacing={1}>
          {tabIndex === 0 && (
            <Button variant="contained" onClick={handleCreateAssetClick} disabled={!selectedProductId}>
              New Asset
            </Button>
          )}
          {tabIndex === 2 && (
            <Button variant="contained" onClick={handleCreateChannelClick} disabled={!selectedProductId}>
              New Channel
            </Button>
          )}
        </Stack>
      </Box>

      <Box hidden={tabIndex !== 0}>
        <Typography variant="h5" gutterBottom>
          Arsenal Repo
        </Typography>
        <AssetsLibrary
          assets={arsenalAssets}
          onEdit={(asset) => {
            setAssetModalMode("edit");
            setEditingAsset(asset);
          }}
          onApprove={handleAssetApprove}
          channelLookup={channelLookup}
        />
      </Box>

      <Box hidden={tabIndex !== 1}>
        <Typography variant="h5" gutterBottom>
          Asset Categories
        </Typography>
        <CategoryLibrary
          categories={assetCategories}
          type="asset"
          assetLabelMap={assetLabelMap}
        />
      </Box>

      <Box hidden={tabIndex !== 2}>
        <Typography variant="h5" gutterBottom>
          Channel Categories
        </Typography>
        <CategoryLibrary
          categories={channelCategories}
          type="channel"
          channelCategoryAssets={channelCategoryAssets}
          assetLabelMap={assetLabelMap}
        />
      </Box>

      <Dialog
        open={Boolean(editingAsset)}
        onClose={() => {
          if (assetSaving) return;
          setAssetModalMode("edit");
          setEditingAsset(null);
        }}
        maxWidth="sm"
        fullWidth
      >
        <DialogTitle>{isCreatingAsset ? "Create Asset" : "Edit Asset Metadata"}</DialogTitle>
        <DialogContent dividers>
          <Box sx={{ display: "grid", gap: 2 }}>
            {!isCreatingAsset && editingAsset?.id && (
              <TextField label="Asset ID" value={editingAsset.id} InputProps={{ readOnly: true }} />
            )}
            <TextField
              label="Asset title"
              value={assetForm.name}
              onChange={(event: ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) =>
                handleAssetFieldChange("name", event.target.value)
              }
              required
            />

            <Autocomplete<string, false, false, true>
              freeSolo
              selectOnFocus
              clearOnBlur
              handleHomeEndKeys
              options={assetCategoryValues}
              value={assetForm.category || null}
              inputValue={assetCategoryInput}
              onInputChange={(_, newInput) => setAssetCategoryInput(newInput)}
              onChange={(_, newValue) => {
                if (!newValue) {
                  setAssetCategoryInput("");
                  handleAssetFieldChange("category", "");
                  return;
                }
                const label =
                  labelFromMap(assetCategoryLabelMap, newValue) ||
                  newValue.trim() ||
                  assetCategoryInput;
                setAssetCategoryInput(label);
                handleAssetFieldChange("category", newValue.trim(), {
                  displayLabel: label,
                });
              }}
              renderOption={(props, option) => (
                <li {...props}>{labelFromMap(assetCategoryLabelMap, option)}</li>
              )}
              getOptionLabel={(option) =>
                typeof option === "string"
                  ? labelFromMap(assetCategoryLabelMap, option)
                  : labelFromMap(assetCategoryLabelMap, option)
              }
              renderInput={(params) => (
                <TextField
                  {...params}
                  label="Category"
                  placeholder="Select or type a category"
                  required
                />
              )}
            />

            <Autocomplete<string, false, false, true>
              freeSolo
              selectOnFocus
              clearOnBlur
              handleHomeEndKeys
              options={assetContentTypeValues}
              value={assetForm.content_type || null}
              inputValue={assetContentTypeInput}
              onInputChange={(_, newInput) => setAssetContentTypeInput(newInput)}
              onChange={(_, newValue) => {
                if (!newValue) {
                  setAssetContentTypeInput("");
                  handleAssetFieldChange("content_type", "");
                  return;
                }
                const label =
                  labelFromMap(assetContentTypeLabelMap, newValue) ||
                  newValue.trim() ||
                  assetContentTypeInput;
                setAssetContentTypeInput(label);
                handleAssetFieldChange("content_type", newValue.trim(), {
                  displayLabel: label,
                });
              }}
              renderOption={(props, option) => (
                <li {...props}>{labelFromMap(assetContentTypeLabelMap, option)}</li>
              )}
              getOptionLabel={(option) =>
                typeof option === "string"
                  ? labelFromMap(assetContentTypeLabelMap, option)
                  : labelFromMap(assetContentTypeLabelMap, option)
              }
              renderInput={(params) => (
                <TextField
                  {...params}
                  label="Content Type"
                  placeholder="Select or type a content type"
                  required
                />
              )}
            />

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

            <Autocomplete<EnumOption, false, false, false>
              options={assetOptions.depth}
              value={
                assetOptions.depth.find(
                  (option) =>
                    option.value.toLowerCase() === assetForm.depth.toLowerCase()
                ) || null
              }
              onChange={(_, option) =>
                handleAssetFieldChange("depth", option?.value || "")
              }
              getOptionLabel={(option) => option.label || option.value}
              isOptionEqualToValue={(option, value) => option.value === value.value}
              renderInput={(params) => (
                <TextField
                  {...params}
                  label="Depth"
                  placeholder="Select a depth"
                />
              )}
            />

            <Autocomplete<string, true, false, true>
              multiple
              freeSolo
              options={assetPersonaOptions}
              value={assetForm.targetPersonas}
              onChange={(_, value) =>
                handleAssetListChange(
                  "targetPersonas",
                  (value as string[]).map((entry) => entry.trim()).filter(Boolean)
                )
              }
              renderInput={(params) => (
                <TextField
                  {...params}
                  label="Target personas"
                  placeholder="Add persona labels"
                />
              )}
            />

            <Autocomplete<string, true, false, true>
              multiple
              freeSolo
              options={assetSegmentOptions}
              value={assetForm.targetAccountSegments}
              onChange={(_, value) =>
                handleAssetListChange(
                  "targetAccountSegments",
                  (value as string[]).map((entry) => entry.trim()).filter(Boolean)
                )
              }
              renderInput={(params) => (
                <TextField
                  {...params}
                  label="Target account segments"
                  placeholder="Add segments like 'Industry: Finance'"
                />
              )}
            />

            <Autocomplete<string, true, false, true>
              multiple
              freeSolo
              options={dedupeStrings([...assetStageOptions, ...assetUsageStageLabels])}
              value={assetForm.targetStages}
              onChange={(_, value) =>
                handleAssetListChange(
                  "targetStages",
                  (value as string[]).map((entry) => entry.trim()).filter(Boolean)
                )
              }
              renderInput={(params) => (
                <TextField
                  {...params}
                  label="Target belief stages"
                  placeholder="Select stages this asset supports"
                />
              )}
            />

            <FormControl fullWidth>
              <InputLabel id="asset-org-conversion-label">
                Org Conversion Maturity
              </InputLabel>
              <Select
                labelId="asset-org-conversion-label"
                value={assetForm.orgConversionMaturity}
                label="Org Conversion Maturity"
                onChange={(event) =>
                  handleAssetFieldChange(
                    "orgConversionMaturity",
                    event.target.value as string
                  )
                }
              >
                <MenuItem value="">
                  <em>Not set</em>
                </MenuItem>
                {ORG_CONVERSION_OPTIONS.map((option) => (
                  <MenuItem key={option.value} value={option.value}>
                    {option.label}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>

            <Autocomplete<string, true, false, true>
              multiple
              freeSolo
              options={assetConcernOptions}
              value={assetForm.targetConcerns}
              onChange={(_, value) =>
                handleAssetListChange(
                  "targetConcerns",
                  (value as string[]).map((entry) => entry.trim()).filter(Boolean)
                )
              }
              renderInput={(params) => (
                <TextField
                  {...params}
                  label="Key concerns / pains"
                  placeholder="Add pains, jobs, or triggers this asset addresses"
                />
              )}
            />

            <FormControl fullWidth>
              <InputLabel id="asset-primary-channel-label">Primary channel</InputLabel>
              <Select
                labelId="asset-primary-channel-label"
                label="Primary channel"
                value={assetForm.channelId}
                onChange={(event) =>
                  setAssetForm((prev) => ({ ...prev, channelId: event.target.value as string }))
                }
              >
                <MenuItem value="">
                  <em>None</em>
                </MenuItem>
                {arsenalChannels.map((channel) => (
                  <MenuItem key={channel.id} value={channel.id}>
                    {channel.name || channel.slug || channel.id}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>

            <Autocomplete<string, true, false, false>
              multiple
              options={channelOptionsForSelect}
              value={assetForm.typicalChannels}
              onChange={(_, value) =>
                handleAssetIdListChange(
                  "typicalChannels",
                  (value as string[]).map((entry) => entry.trim()).filter(Boolean)
                )
              }
              getOptionLabel={(option) => channelLabelMap[option] || option}
              renderTags={(value, getTagProps) =>
                value.map((option, index) => (
                  <Chip
                    {...getTagProps({ index })}
                    label={channelLabelMap[option] || option}
                    size="small"
                  />
                ))
              }
              renderInput={(params) => (
                <TextField
                  {...params}
                  label="Typical distribution channels"
                  placeholder="Select channels that usually deliver this asset"
                />
              )}
            />

            <Autocomplete<string, true, false, true>
              multiple
              freeSolo
              options={activityOptions}
              value={assetForm.activities}
              onChange={(_, value) =>
                setAssetForm((prev) => ({
                  ...prev,
                  activities: (value as string[]).map((entry) => entry.trim()).filter(Boolean),
                }))
              }
              renderInput={(params) => (
                <TextField {...params} label="Activities" placeholder="e.g., Download, Attend demo" />
              )}
            />

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
          <Button
            onClick={() => {
              setAssetModalMode("edit");
              setEditingAsset(null);
            }}
            disabled={assetSaving}
          >
            Cancel
          </Button>
          <Button onClick={handleAssetSave} variant="contained" disabled={assetSaving}>
            {isCreatingAsset ? "Create" : "Save"}
          </Button>
        </DialogActions>
      </Dialog>

      <Dialog
        open={Boolean(editingChannel)}
        onClose={() => {
          if (channelSaving) return;
          setChannelModalMode("edit");
          setEditingChannel(null);
        }}
        maxWidth="sm"
        fullWidth
      >
        <DialogTitle>{isCreatingChannel ? "Create Channel" : "Edit Channel Metadata"}</DialogTitle>
        <DialogContent dividers>
          <Box sx={{ display: "grid", gap: 2 }}>
            <TextField
              label="Channel name"
              value={channelForm.name}
              onChange={(event: ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) =>
                handleChannelFieldChange("name", event.target.value)
              }
              required
            />

            <Autocomplete<string, false, false, true>
              freeSolo
              selectOnFocus
              clearOnBlur
              handleHomeEndKeys
              options={channelTypeValues}
              value={channelForm.channel_type || null}
              inputValue={channelTypeInput}
              onInputChange={(_, newInput) => setChannelTypeInput(newInput)}
              onChange={(_, newValue) => {
                if (!newValue) {
                  setChannelTypeInput("");
                  handleChannelFieldChange("channel_type", "");
                  return;
                }
                const label =
                  labelFromMap(channelTypeLabelMap, newValue) ||
                  newValue.trim() ||
                  channelTypeInput;
                setChannelTypeInput(label);
                handleChannelFieldChange("channel_type", newValue.trim(), {
                  displayLabel: label,
                });
              }}
              renderOption={(props, option) => (
                <li {...props}>{labelFromMap(channelTypeLabelMap, option)}</li>
              )}
              getOptionLabel={(option) =>
                typeof option === "string"
                  ? labelFromMap(channelTypeLabelMap, option)
                  : labelFromMap(channelTypeLabelMap, option)
              }
              renderInput={(params) => (
                <TextField
                  {...params}
                  label="Channel Type"
                  placeholder="Select or type a channel type"
                  required
                />
              )}
            />

            <Autocomplete<string, false, false, true>
              freeSolo
              selectOnFocus
              clearOnBlur
              handleHomeEndKeys
              options={deliveryModeValues}
              value={channelForm.delivery_mode || null}
              inputValue={deliveryModeInput}
              onInputChange={(_, newInput) => setDeliveryModeInput(newInput)}
              onChange={(_, newValue) => {
                if (!newValue) {
                  setDeliveryModeInput("");
                  handleChannelFieldChange("delivery_mode", "");
                  return;
                }
                const label =
                  labelFromMap(deliveryModeLabelMap, newValue) ||
                  newValue.trim() ||
                  deliveryModeInput;
                setDeliveryModeInput(label);
                handleChannelFieldChange("delivery_mode", newValue.trim(), {
                  displayLabel: label,
                });
              }}
              renderOption={(props, option) => (
                <li {...props}>{labelFromMap(deliveryModeLabelMap, option)}</li>
              )}
              getOptionLabel={(option) =>
                typeof option === "string"
                  ? labelFromMap(deliveryModeLabelMap, option)
                  : labelFromMap(deliveryModeLabelMap, option)
              }
              renderInput={(params) => (
                <TextField
                  {...params}
                  label="Delivery Mode"
                  placeholder="Select or type a delivery mode"
                  required
                />
              )}
            />

            <Autocomplete<string, true, false, true>
              multiple
              freeSolo
              options={channelPersonaOptions}
              value={channelForm.targetPersonas}
              onChange={(_, value) =>
                handleChannelListChange(
                  "targetPersonas",
                  (value as string[]).map((entry) => entry.trim()).filter(Boolean)
                )
              }
              renderInput={(params) => (
                <TextField
                  {...params}
                  label="Target personas"
                  placeholder="Add persona labels"
                />
              )}
            />

            <Autocomplete<string, true, false, true>
              multiple
              freeSolo
              options={channelSegmentOptions}
              value={channelForm.targetAccountSegments}
              onChange={(_, value) =>
                handleChannelListChange(
                  "targetAccountSegments",
                  (value as string[]).map((entry) => entry.trim()).filter(Boolean)
                )
              }
              renderInput={(params) => (
                <TextField
                  {...params}
                  label="Target account segments"
                  placeholder="Add segments like 'Industry: Finance'"
                />
              )}
            />

            <Autocomplete<string, true, false, true>
              multiple
              freeSolo
              options={dedupeStrings([...channelStageOptions, ...channelUsageStageLabels])}
              value={channelForm.targetStages}
              onChange={(_, value) =>
                handleChannelListChange(
                  "targetStages",
                  (value as string[]).map((entry) => entry.trim()).filter(Boolean)
                )
              }
              renderInput={(params) => (
                <TextField
                  {...params}
                  label="Target belief stages"
                  placeholder="Select stages this channel supports"
                />
              )}
            />

            <Autocomplete<string, true, false, false>
              multiple
              options={assetOptionsForSelect}
              value={channelForm.typicalAssets}
              onChange={(_, value) =>
                handleChannelIdListChange(
                  "typicalAssets",
                  (value as string[]).map((entry) => entry.trim()).filter(Boolean)
                )
              }
              getOptionLabel={(option) => assetLabelMap[option] || option}
              renderTags={(value, getTagProps) =>
                value.map((option, index) => (
                  <Chip
                    {...getTagProps({ index })}
                    label={assetLabelMap[option] || option}
                    size="small"
                  />
                ))
              }
              renderInput={(params) => (
                <TextField
                  {...params}
                  label="Typically paired assets"
                  placeholder="Select assets this channel usually distributes"
                />
              )}
            />
            <FormControl fullWidth>
              <InputLabel id="channel-org-conversion-label">
                Org Conversion Maturity
              </InputLabel>
              <Select
                labelId="channel-org-conversion-label"
                value={channelForm.orgConversionMaturity}
                label="Org Conversion Maturity"
                onChange={(event) =>
                  handleChannelFieldChange(
                    "orgConversionMaturity",
                    event.target.value as string
                  )
                }
              >
                <MenuItem value="">
                  <em>Not set</em>
                </MenuItem>
                {ORG_CONVERSION_OPTIONS.map((option) => (
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
          <Button
            onClick={() => {
              setChannelModalMode("edit");
              setEditingChannel(null);
            }}
            disabled={channelSaving}
          >
            Cancel
          </Button>
          <Button onClick={handleChannelSave} variant="contained" disabled={channelSaving}>
            {isCreatingChannel ? "Create" : "Save"}
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
