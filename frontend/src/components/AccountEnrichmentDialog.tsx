import React, { useEffect, useMemo, useState } from "react";
import {
  Box,
  Button,
  Chip,
  CircularProgress,
  Dialog,
  DialogActions,
  DialogContent,
  DialogTitle,
  Divider,
  IconButton,
  Stack,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  TextField,
  Tooltip,
  Typography,
  MenuItem,
  Paper,
} from "@mui/material";
import CloseIcon from "@mui/icons-material/Close";

export interface EnrichmentAccount {
  id?: string;
  account_name: string;
}

type MatchPayload = {
  match_id: string;
  person_id?: string | null;
  person_name?: string | null;
  person_title?: string | null;
  person_department?: string | null;
  person_seniority?: string | null;
  match_confidence?: number | null;
  source?: string | null;
  notes?: string | null;
};

type CandidatePayload = {
  person_id: string;
  person_name: string;
  person_title?: string | null;
  person_department?: string | null;
  person_seniority?: string | null;
  confidence?: number | null;
  engagement_count?: number | null;
  reasons?: string[];
};

type PersonaRequirement = {
  persona_id: string;
  persona_label: string;
  persona_title?: string | null;
  persona_department?: string | null;
  persona_seniority?: string | null;
  expected_in_deal: number;
  expected_in_deal_pct: number;
  stage_index: number;
  stage_label: string;
  matched_people: MatchPayload[];
  suggested_people: CandidatePayload[];
};

type PersonaCandidate = {
  label: string;
  normalized_label: string;
  persona_title?: string | null;
  persona_department?: string | null;
  persona_seniority?: string | null;
  account_occurrences: number;
  account_first_seen?: string | null;
  account_last_seen?: string | null;
  global_account_count?: number;
  global_occurrences?: number;
  suggested_persona_id?: string | null;
  suggested_persona_label?: string | null;
  similarity?: number | null;
  wolves_score?: number | null;
  wolves_delta_bp?: number | null;
};

type EnrichmentSummary = {
  required_personas: number;
  personas_with_matches: number;
  matched_people: number;
  total_people: number;
  coverage_pct: number;
  coverage_ratio: number;
};

type EnrichmentResponse = {
  account_id: string;
  product_id: string;
  blueprint_generated_at?: string | null;
  persona_requirements: PersonaRequirement[];
  summary: EnrichmentSummary;
  persona_candidates?: PersonaCandidate[];
};

type MatchEditorState = {
  persona: PersonaRequirement;
  existingMatch?: MatchPayload;
};

type AccountEnrichmentDialogProps = {
  open: boolean;
  onClose: () => void;
  account: EnrichmentAccount | null;
  productId?: string | null;
  token?: string | null;
};

const formatPercent = (value?: number | null) => {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  return `${Math.round(value)}%`;
};

const formatConfidence = (value?: number | null) => {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  return `${Math.round(value * 100)}%`;
};

const formatDate = (value?: string | null) => {
  if (!value) return "—";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "—";
  return date.toLocaleDateString();
};

const buildPersonDescriptor = (
  name?: string | null,
  title?: string | null,
  department?: string | null,
  seniority?: string | null
) => {
  return [name, title, department, seniority].filter(Boolean).join(" · ");
};

const describeMatch = (match: MatchPayload) => {
  const descriptor = buildPersonDescriptor(
    match.person_name,
    match.person_title,
    match.person_department,
    match.person_seniority
  );
  if (descriptor) return descriptor;
  if (match.notes && match.notes.trim()) return match.notes.trim();
  return "Unnamed person";
};

const AccountEnrichmentDialog: React.FC<AccountEnrichmentDialogProps> = ({
  open,
  onClose,
  account,
  productId,
  token,
}) => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [data, setData] = useState<EnrichmentResponse | null>(null);
  const [editorState, setEditorState] = useState<MatchEditorState | null>(null);
  const [selectedCandidateId, setSelectedCandidateId] = useState<string>("");
  const [manualNotes, setManualNotes] = useState<string>("");
  const [manualConfidence, setManualConfidence] = useState<string>("");
  const [manualName, setManualName] = useState<string>("");
  const [manualTitle, setManualTitle] = useState<string>("");
  const [manualDepartment, setManualDepartment] = useState<string>("");
  const [manualSeniority, setManualSeniority] = useState<string>("");
  const [candidateActionLoading, setCandidateActionLoading] = useState<string | null>(null);

  const authHeader = useMemo(() => ({
    Authorization: token ? `Bearer ${token}` : "",
    "Content-Type": "application/json",
  }), [token]);

  useEffect(() => {
    if (!open || !account?.id || !productId || !token) {
      setData(null);
      setError(null);
      return;
    }

    const controller = new AbortController();
    setLoading(true);
    setError(null);
    (async () => {
      try {
        const res = await fetch(
          `http://localhost:8000/accounts/${account.id}/enrichment?product_id=${productId}`,
          { headers: { Authorization: `Bearer ${token}` }, signal: controller.signal }
        );
        if (!res.ok) {
          throw new Error(await res.text());
        }
        const payload = (await res.json()) as EnrichmentResponse;
        console.log('Enrichment payload', payload);
        setData(payload);
      } catch (err: any) {
        if (err?.name === "AbortError") {
          return;
        }
        console.error("Failed to load enrichment", err);
        setError(err?.message || "Failed to load enrichment");
      } finally {
        setLoading(false);
      }
    })();
    return () => controller.abort();
  }, [open, account?.id, productId, token]);

  const resetEditor = () => {
    setEditorState(null);
    setSelectedCandidateId("");
    setManualNotes("");
    setManualConfidence("");
    setManualName("");
    setManualTitle("");
    setManualDepartment("");
    setManualSeniority("");
    setError(null);
  };

  const handleOpenEditor = (
    persona: PersonaRequirement,
    match?: MatchPayload,
    candidateId?: string
  ) => {
    const candidate = candidateId
      ? persona.suggested_people.find((c) => c.person_id === candidateId)
      : undefined;
    setEditorState({ persona, existingMatch: match });
    if (match?.person_id) {
      setSelectedCandidateId(match.person_id);
    } else if (candidate) {
      setSelectedCandidateId(candidate.person_id);
    } else {
      setSelectedCandidateId("");
    }
    const initialNotes =
      match?.notes ||
      (candidate?.reasons && candidate.reasons.length
        ? candidate.reasons.join("; ")
        : "");
    const initialConfidence =
      match?.match_confidence ?? candidate?.confidence ?? null;
    setManualNotes(initialNotes || "");
    setManualConfidence(
      initialConfidence !== null && initialConfidence !== undefined
        ? String(initialConfidence)
        : ""
    );
    setManualName(match?.person_name || candidate?.person_name || "");
    setManualTitle(match?.person_title || candidate?.person_title || "");
    setManualDepartment(
      match?.person_department || candidate?.person_department || ""
    );
    setManualSeniority(
      match?.person_seniority || candidate?.person_seniority || ""
    );
    setError(null);
  };

  const refresh = async () => {
    if (!account?.id || !productId || !token) return;
    try {
      const res = await fetch(
        `http://localhost:8000/accounts/${account.id}/enrichment?product_id=${productId}`,
        { headers: { Authorization: `Bearer ${token}` } }
      );
      if (!res.ok) throw new Error(await res.text());
      const payload = (await res.json()) as EnrichmentResponse;
      setData(payload);
    } catch (err: any) {
      console.error("Failed to refresh enrichment", err);
      setError(err?.message || "Failed to refresh enrichment");
    }
  };

  const handleSaveMatch = async () => {
    if (!editorState || !account?.id || !productId || !token) return;
    const { persona, existingMatch } = editorState;
    const selectedCandidate = persona.suggested_people.find(
      (candidate) => candidate.person_id === selectedCandidateId
    );

    if (!selectedCandidate && !existingMatch && !manualName.trim()) {
      setError("Provide a person name or choose a suggestion to match this persona.");
      return;
    }

    setError(null);

    try {
      const manualConfidenceValue = manualConfidence.trim();
      let confidenceValue: number | null;
      if (selectedCandidate) {
        confidenceValue = selectedCandidate.confidence ?? null;
      } else if (manualConfidenceValue) {
        const numericConfidence = Number(manualConfidenceValue);
        if (Number.isNaN(numericConfidence)) {
          setError("Confidence must be a number between 0 and 1.");
          return;
        }
        confidenceValue = numericConfidence;
      } else {
        confidenceValue = null;
      }

      const clean = (value: string) => {
        const trimmed = value.trim();
        return trimmed.length ? trimmed : null;
      };

      const manualNameClean = clean(manualName);
      const manualTitleClean = clean(manualTitle);
      const manualDepartmentClean = clean(manualDepartment);
      const manualSeniorityClean = clean(manualSeniority);
      const manualNotesClean = clean(manualNotes);
      const reasonNotes =
        selectedCandidate?.reasons && selectedCandidate.reasons.length
          ? selectedCandidate.reasons.join("; ")
          : null;
      const existingNotesClean =
        existingMatch?.notes && existingMatch.notes.trim()
          ? existingMatch.notes.trim()
          : null;
      const notes = manualNotesClean || reasonNotes || existingNotesClean || null;

      const payload = {
        product_id: productId,
        persona_id: persona.persona_id,
        persona_label: persona.persona_label,
        stage: persona.stage_label,
        person_id: selectedCandidate?.person_id || existingMatch?.person_id || null,
        match_confidence: confidenceValue,
        notes,
        match_id: existingMatch?.match_id,
        person_name:
          manualNameClean ||
          selectedCandidate?.person_name ||
          existingMatch?.person_name ||
          null,
        person_title:
          manualTitleClean ||
          selectedCandidate?.person_title ||
          existingMatch?.person_title ||
          null,
        person_department:
          manualDepartmentClean ||
          selectedCandidate?.person_department ||
          existingMatch?.person_department ||
          null,
        person_seniority:
          manualSeniorityClean ||
          selectedCandidate?.person_seniority ||
          existingMatch?.person_seniority ||
          null,
      };

      setLoading(true);
      const res = await fetch(
        `http://localhost:8000/accounts/${account.id}/enrichment/matches`,
        {
          method: "POST",
          headers: authHeader,
          body: JSON.stringify(payload),
        }
      );
      if (!res.ok) {
        throw new Error(await res.text());
      }
      await refresh();
      resetEditor();
    } catch (err: any) {
      console.error("Failed to save match", err);
      setError(err?.message || "Failed to save match");
    } finally {
      setLoading(false);
    }
  };

  const handleDeleteMatch = async (match: MatchPayload) => {
    if (!account?.id || !productId || !token) return;
    try {
      setLoading(true);
      const res = await fetch(
        `http://localhost:8000/accounts/${account.id}/enrichment/matches/${match.match_id}?product_id=${productId}`,
        { method: "DELETE", headers: { Authorization: `Bearer ${token}` } }
      );
      if (!res.ok) {
        throw new Error(await res.text());
      }
      await refresh();
    } catch (err: any) {
      console.error("Failed to delete match", err);
      setError(err?.message || "Failed to delete match");
    } finally {
      setLoading(false);
    }
  };

  const handleCandidateAction = async (
    candidate: PersonaCandidate,
    action: "add" | "match"
  ) => {
    if (!account?.id || !productId || !token) return;
    if (action === "match" && !candidate.suggested_persona_id) return;
    const actionKey = `${action}-${candidate.normalized_label}`;
    try {
      setCandidateActionLoading(actionKey);
      const payload = {
        product_id: productId,
        label: candidate.label,
        title: candidate.persona_title,
        department: candidate.persona_department,
        seniority: candidate.persona_seniority,
        action,
        target_persona_id: action === "match" ? candidate.suggested_persona_id : undefined,
      };
      const res = await fetch(
        `http://localhost:8000/accounts/${account.id}/enrichment/personas`,
        {
          method: "POST",
          headers: authHeader,
          body: JSON.stringify(payload),
        }
      );
      if (!res.ok) {
        throw new Error(await res.text());
      }
      await refresh();
    } catch (err: any) {
      console.error("Failed to apply candidate action", err);
      setError(err?.message || "Failed to apply candidate action");
    } finally {
      setCandidateActionLoading(null);
    }
  };

  const coverageLabel = useMemo(() => {
    if (!data?.summary) return "";
    const { coverage_pct, personas_with_matches, required_personas } = data.summary;
    return `${coverage_pct}% enriched (${personas_with_matches}/${required_personas} personas)`;
  }, [data?.summary]);

  return (
    <Dialog open={open} onClose={onClose} fullWidth maxWidth="lg">
      <DialogTitle sx={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
        <span>Account Enrichment · {account?.account_name || "Account"}</span>
        <IconButton onClick={onClose} size="small">
          <CloseIcon fontSize="small" />
        </IconButton>
      </DialogTitle>
      <DialogContent dividers>
        {loading && !data ? (
          <Box sx={{ display: "flex", justifyContent: "center", py: 4 }}>
            <CircularProgress size={32} />
          </Box>
        ) : null}
        {error && (
          <Box sx={{ mb: 2 }}>
            <Typography color="error" variant="body2">
              {error}
            </Typography>
          </Box>
        )}
        {data && (
          <Stack spacing={2}>
            <Box>
              <Typography variant="subtitle2" color="text.secondary">
                Enrichment Summary
              </Typography>
              <Typography variant="h6">{coverageLabel}</Typography>
              <Typography variant="body2" color="text.secondary">
                Matched people: {data.summary.matched_people} · Total people tracked: {data.summary.total_people}
              </Typography>
            </Box>

            {data.persona_candidates && data.persona_candidates.length > 0 && (
              <Box>
                <Typography variant="subtitle2" color="text.secondary">
                  Persona candidates from recent engagements
                </Typography>
                <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
                  These titles were observed in engagement data but are not yet mapped to your graph. Promote them or
                  match them to an existing persona.
                </Typography>
                <Stack spacing={1.5}>
                  {data.persona_candidates.map((candidate) => {
                    const addKey = `add-${candidate.normalized_label}`;
                    const matchKey = `match-${candidate.normalized_label}`;
                    const addLoading = candidateActionLoading === addKey;
                    const matchLoading = candidateActionLoading === matchKey;
                    return (
                      <Paper
                        key={candidate.normalized_label}
                        variant="outlined"
                        sx={{ p: 1.5, borderColor: "primary.100" }}
                      >
                        <Stack
                          direction={{ xs: "column", sm: "row" }}
                          spacing={1}
                          alignItems={{ xs: "flex-start", sm: "center" }}
                          justifyContent="space-between"
                        >
                          <Box>
                            <Typography variant="subtitle1">{candidate.label}</Typography>
                            <Typography variant="body2" color="text.secondary">
                              {[
                                candidate.persona_title,
                                candidate.persona_department,
                                candidate.persona_seniority,
                              ]
                                .filter(Boolean)
                                .join(" · ") || "—"}
                            </Typography>
                            <Typography variant="caption" color="text.secondary">
                              {candidate.account_occurrences} engagements in this account ·{" "}
                              {candidate.global_account_count || 0} accounts overall
                              {candidate.account_last_seen
                                ? ` · Last seen ${formatDate(candidate.account_last_seen)}`
                                : ""}
                            </Typography>
                            <Stack direction="row" spacing={1} sx={{ mt: 0.5, flexWrap: "wrap" }}>
                              {candidate.suggested_persona_label && (
                                <Chip
                                  size="small"
                                  label={`Suggested: ${candidate.suggested_persona_label}`}
                                  variant="outlined"
                                />
                              )}
                              {typeof candidate.similarity === "number" && (
                                <Chip
                                  size="small"
                                  label={`Similarity ${formatConfidence(candidate.similarity)}`}
                                  variant="outlined"
                                />
                              )}
                              {typeof candidate.wolves_score === "number" && (
                                <Chip
                                  size="small"
                                  label={`Wolves ${formatConfidence(candidate.wolves_score)}`}
                                  color="success"
                                  variant="outlined"
                                />
                              )}
                              {typeof candidate.wolves_delta_bp === "number" && (
                                <Chip
                                  size="small"
                                  label={`Δ ${Math.round(candidate.wolves_delta_bp)} bp`}
                                  variant="outlined"
                                />
                              )}
                            </Stack>
                          </Box>
                          <Stack direction="row" spacing={1}>
                            <Button
                              size="small"
                              variant="contained"
                              onClick={() => handleCandidateAction(candidate, "add")}
                              disabled={addLoading}
                            >
                              {addLoading ? "Adding..." : "Add Persona"}
                            </Button>
                            {candidate.suggested_persona_id && (
                              <Button
                                size="small"
                                variant="outlined"
                                onClick={() => handleCandidateAction(candidate, "match")}
                                disabled={matchLoading}
                              >
                                {matchLoading ? "Matching..." : `Match to ${candidate.suggested_persona_label}`}
                              </Button>
                            )}
                          </Stack>
                        </Stack>
                      </Paper>
                    );
                  })}
                </Stack>
              </Box>
            )}

            <Divider />

            <TableContainer>
              <Table size="small">
                <TableHead>
                  <TableRow>
                    <TableCell>Persona</TableCell>
                    <TableCell>Expected in Deal</TableCell>
                    <TableCell>Stage</TableCell>
                    <TableCell>Matched People</TableCell>
                    <TableCell>Suggestions</TableCell>
                    <TableCell width={120} align="right">
                      Actions
                    </TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {data.persona_requirements.map((persona) => (
                    <TableRow key={persona.persona_id} hover>
                      <TableCell>
                        <Typography variant="subtitle2">{persona.persona_label}</Typography>
                        <Typography variant="caption" color="text.secondary">
                          {persona.persona_title ? persona.persona_title : ""}
                          {persona.persona_department ? ` · ${persona.persona_department}` : ""}
                          {persona.persona_seniority ? ` · ${persona.persona_seniority}` : ""}
                        </Typography>
                      </TableCell>
                      <TableCell>{formatPercent(persona.expected_in_deal_pct)}</TableCell>
                      <TableCell>{persona.stage_label}</TableCell>
                      <TableCell>
                        <Stack direction="row" spacing={0.5} sx={{ flexWrap: "wrap" }}>
                      {persona.matched_people.length === 0 ? (
                        <Typography variant="caption" color="text.secondary">
                          None
                        </Typography>
                      ) : null}
                      {persona.matched_people.map((match) => {
                        const label = describeMatch(match);
                        const confidenceText =
                          match.match_confidence === null ||
                          match.match_confidence === undefined
                            ? ""
                            : ` (${formatConfidence(match.match_confidence)})`;
                        const chipLabel = `${label}${confidenceText}`;
                        const tooltip =
                          match.notes && match.notes.trim()
                            ? match.notes.trim()
                            : undefined;
                        return (
                          <Tooltip key={match.match_id} title={tooltip || ""} arrow>
                            <Chip
                              label={chipLabel}
                              onDelete={() => handleDeleteMatch(match)}
                              onClick={() => handleOpenEditor(persona, match)}
                              variant="outlined"
                              size="small"
                              sx={{ mr: 0.5, mb: 0.5, cursor: "pointer" }}
                            />
                          </Tooltip>
                        );
                      })}
                    </Stack>
                  </TableCell>
                  <TableCell>
                    <Stack direction="row" spacing={0.5} sx={{ flexWrap: "wrap" }}>
                      {persona.suggested_people.length === 0 ? (
                            <Typography variant="caption" color="text.secondary">
                              No suggestions
                            </Typography>
                          ) : null}
                          {persona.suggested_people.map((candidate) => {
                            const descriptor =
                              buildPersonDescriptor(
                                candidate.person_name,
                                candidate.person_title,
                                candidate.person_department,
                                candidate.person_seniority
                              ) || candidate.person_name;
                            const confidenceText =
                              candidate.confidence === null ||
                              candidate.confidence === undefined
                                ? ""
                                : ` (${formatConfidence(candidate.confidence)})`;
                            const tooltip =
                              candidate.reasons && candidate.reasons.length
                                ? candidate.reasons.join(" • ")
                                : "Suggested";
                            return (
                              <Tooltip key={candidate.person_id} title={tooltip} arrow>
                                <Chip
                                  label={`${descriptor}${confidenceText}`}
                                  size="small"
                                  variant="outlined"
                                  clickable
                                  onClick={() =>
                                    handleOpenEditor(persona, undefined, candidate.person_id)
                                  }
                                  sx={{ mr: 0.5, mb: 0.5, cursor: "pointer" }}
                                />
                              </Tooltip>
                            );
                          })}
                        </Stack>
                      </TableCell>
                      <TableCell align="right">
                        <Button
                          size="small"
                          variant="contained"
                          onClick={() => handleOpenEditor(persona)}
                        >
                          {persona.matched_people.length ? "Manage Matches" : "Add Match"}
                        </Button>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          </Stack>
        )}
      </DialogContent>
      <DialogActions>
        <Button onClick={onClose} color="inherit">
          Close
        </Button>
      </DialogActions>

      {/* Match editor dialog */}
      <Dialog open={Boolean(editorState)} onClose={resetEditor} maxWidth="sm" fullWidth>
        <DialogTitle>Match Persona</DialogTitle>
        <DialogContent dividers>
          {editorState ? (
            <Stack spacing={2}>
              <Box>
                <Typography variant="subtitle1">{editorState.persona.persona_label}</Typography>
                <Typography variant="body2" color="text.secondary">
                  Stage: {editorState.persona.stage_label} · Expected: {formatPercent(editorState.persona.expected_in_deal_pct)}
                </Typography>
              </Box>
              {editorState.persona.matched_people.length ? (
                <Box>
                  <Typography variant="subtitle2">Existing matches</Typography>
                  <Typography variant="caption" color="text.secondary">
                    Click a person to edit details or remove them from this persona.
                  </Typography>
                  <Stack direction="row" spacing={0.5} sx={{ flexWrap: "wrap", mt: 1 }}>
                    {editorState.persona.matched_people.map((match) => {
                      const label = describeMatch(match);
                      const confidenceText =
                        match.match_confidence === null ||
                        match.match_confidence === undefined
                          ? ""
                          : ` (${formatConfidence(match.match_confidence)})`;
                      const chipLabel = `${label}${confidenceText}`;
                      const tooltip =
                        match.notes && match.notes.trim()
                          ? match.notes.trim()
                          : "Click to edit · Delete to remove";
                      return (
                        <Tooltip key={match.match_id} title={tooltip} arrow>
                          <Chip
                            label={chipLabel}
                            onClick={() => handleOpenEditor(editorState.persona, match)}
                            onDelete={() => handleDeleteMatch(match)}
                            variant="outlined"
                            size="small"
                            sx={{ mr: 0.5, mb: 0.5, cursor: "pointer" }}
                          />
                        </Tooltip>
                      );
                    })}
                  </Stack>
                </Box>
              ) : null}

              <Box>
                <Typography variant="subtitle2">Select a suggested person</Typography>
                <TextField
                  select
                  fullWidth
                  size="small"
                  value={selectedCandidateId}
                  onChange={(e) => setSelectedCandidateId(e.target.value)}
                  helperText="Optional — choose from suggestions or leave blank for manual match"
                >
                  <MenuItem value="">-- None --</MenuItem>
                  {editorState.persona.suggested_people.map((candidate) => (
                    <MenuItem key={candidate.person_id} value={candidate.person_id}>
                      {candidate.person_name}
                      {candidate.person_title ? ` · ${candidate.person_title}` : ""}
                      {candidate.confidence ? ` (${formatConfidence(candidate.confidence)})` : ""}
                    </MenuItem>
                  ))}
                </TextField>
              </Box>

              <Box>
                <Typography variant="subtitle2">Manual person details</Typography>
                <Typography variant="body2" color="text.secondary">
                  Add details for someone not already tracked or refine a suggested match.
                </Typography>
                <Stack spacing={1.5} sx={{ mt: 1 }}>
                  <TextField
                    label="Person Name"
                    fullWidth
                    size="small"
                    value={manualName}
                    onChange={(e) => setManualName(e.target.value)}
                    required={!selectedCandidateId}
                    helperText={
                      selectedCandidateId
                        ? "Optional when adjusting a suggested person"
                        : "Required when adding a new person"
                    }
                  />
                  <TextField
                    label="Title"
                    fullWidth
                    size="small"
                    value={manualTitle}
                    onChange={(e) => setManualTitle(e.target.value)}
                  />
                  <TextField
                    label="Department"
                    fullWidth
                    size="small"
                    value={manualDepartment}
                    onChange={(e) => setManualDepartment(e.target.value)}
                  />
                  <TextField
                    label="Seniority"
                    fullWidth
                    size="small"
                    value={manualSeniority}
                    onChange={(e) => setManualSeniority(e.target.value)}
                  />
                </Stack>
              </Box>

              <Box>
                <Typography variant="subtitle2">Confidence (0-1)</Typography>
                <TextField
                  fullWidth
                  size="small"
                  type="number"
                  inputProps={{ step: 0.05, min: 0, max: 1 }}
                  value={manualConfidence}
                  onChange={(e) => setManualConfidence(e.target.value)}
                  placeholder="Optional — override or add confidence"
                />
              </Box>

              <Box>
                <Typography variant="subtitle2">Notes</Typography>
                <TextField
                  fullWidth
                  size="small"
                  multiline
                  minRows={2}
                  value={manualNotes}
                  onChange={(e) => setManualNotes(e.target.value)}
                  placeholder="Add context when selecting manually"
                />
              </Box>
            </Stack>
          ) : null}
        </DialogContent>
        <DialogActions>
          <Button onClick={resetEditor} color="inherit">
            Cancel
          </Button>
          <Button onClick={handleSaveMatch} variant="contained" disabled={loading}>
            Save Match
          </Button>
        </DialogActions>
      </Dialog>
    </Dialog>
  );
};

export default AccountEnrichmentDialog;
