# Decision Intelligence API Contracts

These contracts describe the JSON payloads returned by the backend for the portfolio execution planner, account-level plans, insights feed, personas atlas, and ICP overview. Each contract stays backward-compatible with the frozen v2.0 Execution Plan spec (`meta/portfolio/themes`) while explicitly surfacing wolves (decision drivers) and subsidy cascades (enablement burdens). They are ready to paste into TypeScript interfaces or Pydantic models.

## 1. ComprehensiveExecutionPlan (Marketing Planner) — `GET /portfolio/{portfolio_id}/execution-plan`

### Key Sections

- `meta`: version (`2.1`), generation timestamp, and the canonical `beliefScale`.
- `portfolio`: overview of the target portfolio with `keyStats`, `decisionDrivers` (wolf share, blocking beliefs), `subsidyMap` (wolf → downstream subsidy personas), `beliefProgression` (path, timing, drop-off), and `painThemes`.
- `themes`: reused from v2.0 (`id`, `name`, `objective`, `campaigns`) with brand-new belief-led fields.

### New v2.1 Fields

- `beliefShifts` (per campaign):
  - Links a belief intervention to a persona, pain, wolf, and unblock target.
  - Includes `expectedLiftPct` and `confidence`.
- `arsenalTable` (extended):
  - References `beliefShiftId` and provides `orgStage`, `avgTimeToImpactDays`, and lift metrics.

#### Example snippet
```
"beliefShifts": [
  {
    "targetPersonaId": "persona_it",
    "fromBelief": "Problem",
    "toBelief": "SolutionViable",
    "corePainId": "pain_security_gaps",
    "wolfPersonaId": "persona_cfo",
    "expectedLiftPct": 0.041
  }
]
```

## 2. AccountPlan — `GET /accounts/{account_id}/plan`

### Core payload

- `meta`: timestamp and canonical belief scale.
- `account`: current deal metrics (belief stage, momentum, readiness, predicted stall).
- `decisionMap`: maps the primary gate (wolf), secondary gates, champion, and supporting persons with belief, influence, and gate type.
- `enablementQueue`: persona-wise fears, needed evidence, risk, and status.
- `nextBestActions`: ranked belief shifts with assets, confidence, lift, urgency, and the unblock narrative.
- `outcomeSimulator`: shows current win/loss expectations plus outcomes if the recommended NBAs execute or are ignored.

### Modeling guidance

- `momentum` and `status` are string enums (e.g., `MEDIUM`, `RED`, `AMBER`).
- All belief stages should align with the `meta.beliefScale`.
- `recommendedWindowDays` frames the execution cadence for NBAs.

## 3. InsightsInbox — `GET /insights/inbox`

### Payload structure

- `insights`: array of cards with `id`, `type` (`DECISION_DRIVER_SHIFT`, `SUBSIDY_FAILURE`, etc.), `severity`, `title`, `description`, `scope`, `evidence`, `implication`, `recommendedAction`, `createdAt`, and `tags`.
- `scope` optionally constrains the insight (ICP IDs, segments, products).
- `evidence` can include metrics before/after change, sample size, time window, affected personas, and correlations.

### Usage tip

- Feed UX cards with `implication`/`recommendedAction` so operators know how to adjust strategy immediately.

## 4. PersonasAtlas — `GET /personas/atlas`

### Highlights

- Each persona includes canonical metadata: `departments`, example titles, `seniority`, `coalitionRole` (e.g., `WOLF`, `CHAMPION`), entry and common pains, win occurrence, activation time, and example accounts.
- `metaSignals` describe subsidy dynamics (who subsidizes them and blocking risk if neglected).

## 5. ICPOverview — `GET /icps/overview`

### Payload summary

- `icps`: typed list with `industry`, `employeeRange`, `geo`, `techStackSignals`, `typicalWolfPersonaId/Name`, win rate, sales cycle, deal size, subsidy burden score, dominant pains, emerging trends, and status (`CORE`, `EMERGING`, `DECLINING`).
- Use `avgSubsidyBurdenScore` to rank enablement effort per ICP.

## Implementing models

When translating to TypeScript or Pydantic:

1. Reuse the shared `beliefScale` array across endpoints to keep validation consistent.
2. Use enums where possible (`beliefStage`, `momentum`, `severity`, `status`, `coalitionRole`).
3. Link `beliefShifts` ↔ `asesnalTable`/`nextBestActions` via IDs to trace lift and unblock flows.
4. Keep `expectedLiftPct`/`confidence` as decimals between `0.0` and `1.0`.

The TypeScript definitions in `frontend/src/types/apiContracts.ts` reflect this contract so frontend engineers can import the shapes directly (`ComprehensiveExecutionPlan`, `AccountPlanContract`, `InsightsInbox`, `PersonasAtlas`, `ICPOverview`) and keep components type-safe as the backend rolls out the v2.1 payload.

Backward compatibility is preserved because v2.1 only adds fields. Existing v2.0 clients can safely ignore unknown keys while upgraded clients can model the new belief-led structures explicitly.
