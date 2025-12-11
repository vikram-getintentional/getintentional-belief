# Live Decision Intelligence UI Contract

This document captures the operational UI contract for the marketing and sales portfolio intelligence product. It is intended for design and engineering to deliver screens that reveal who decides, who needs subsidies, what beliefs block deals, and the highest ROI actions at every level. Technical language such as belief layers, SHM, Bayesian decay, and centrality remain underpinning theory; the UI surfaces actions, impacts, and learning without exposing the philosophy directly.

---

## 1. Marketing Planner (Portfolio Level)

**Core Objective:** Decide what to run, for which accounts, in what sequence to maximize portfolio conversion. This panel answers “If I execute this plan across these accounts, what happens?”

### 1.1 Portfolio Summary
- **Objective:** Give leadership a single-glance truth about the plan’s impact.
- **Top Strip KPI Cards:**
  - Total Target Accounts
  - % Fully Enriched
  - Avg Org Belief State
  - % Accounts with Wolf Identified
  - Expected Conversion %
  - Expected Pipeline $
  - Median Time to First Conversion
- **User Insight Questions:**
  - Is this plan strategically healthy or delusional?
  - Is enrichment holding back execution?
  - Are wolves even visible yet?

### 1.2 Strategy Thesis
- **Objective:** Explain why this plan should work using only learned truth.
- **A. Dominant Decision Drivers (Wolf Layer):** Table shows persona, win-share as final gate, median close time, typical blocking belief.
- **B. Subsidy Load Map (Enablement Burden):** Table shows wolf persona, downstream personas to enable, avg personas per enablement, failure risk.
- **C. Core Belief Progression:** Timeline ZMOT → Problem Recognition → Internal Barrier → Champion → Approval → Implementation with median time between each stage and drop-off %.
- **D. Dominant Pain Themes:** Table mapping canonical pain, % of wins, and stage where it appears.

### 1.3 Execution Plan (Belief-Led)
- **Objective:** Convert strategy into belief shift operations instead of asset spam.
- **Grid:** Each row defines target persona, belief shift (From → To), core pain, best asset type, channel, expected lift, confidence, and whom it unblocks.
- **User Actions:** Identify low-asset-strength belief gaps, allocate budget to high-lift/low-confidence gaps, stop targeting late-stage wolves without subsidy alignment.

### 1.4 Campaign Gantt
- **Objective:** Show sequencing discipline across the campaign.
- **Overarching Schedule:**
  - Weeks 1–2: ZMOT Seeding (Operators)
  - Weeks 3–5: Problem Amplification (Managers)
  - Weeks 6–9: Risk & Barrier Breaking (IT + Finance)
  - Weeks 10–12: Wolf Conversion (CFO / CEO)

---

## 2. Account Plan (Deal Level)

**Core Objective:** Win THIS account by resolving the real gate and subsidizing the right people.

### 2.1 Account Snapshot
- Show current belief stage, live wolf persona, champion strength (0–100), enablement readiness (0–100), deal momentum (Bayesian), predicted stall window (days).
- Include quick insight identifying whether the deal is persuasion- or enablement-blocked.

### 2.2 Decision Map (Wolf + Coalition)
- Table with role, persona, belief, influence, gate type; mark primary gate (wolf), secondary supporters, and the champion.
- User Action: Anchor sequencing on the primary gate.

### 2.3 Subsidy / Enablement Queue
- Table covering persona, what they fear, what they need, risk if ignored, status indicator (e.g., 🔴/🟠).
- Insight: Explain how this enables the wolf and prevents false-negative interpretations.

### 2.4 Next Best Actions (NBA)
- Ranked list showing priority, target persona, belief shift goal, asset type, channel, urgency (“Why Now”), and what each action enables.
- Emphasize this is the weekly execution checklist and nothing else should take precedence.

### 2.5 Outcome Simulator
- Display change in win probability if NBAs execute vs. stalls if ignored (e.g., P(Win) 41%→59%, expected close 21–28 days, predicted stall in 14 days if ignored).

---

## 3. Insights Inbox (Learning Layer)

**Objective:** Surface system learnings that should change strategy.
- Each insight card shows: What changed, evidence (“Why”), implication (“What it means”), and recommended change.
- Sample cards include wolf shifts (e.g., CTO becoming final gate) and subsidy failures (RevOps enablement underfunded).

---

## 4. Personas (Canonical Market Roles)

**Objective:** Display stable, canonical market truths rather than ephemeral campaign roles.
- For each persona, list canonical name, departments, example titles, entry belief stage, common pains, coalition role (driver/champion/blocker/user), occurrence rate in wins, average time to activation, and example accounts.
- **User Actions:** Decide who to enrich in new accounts, validate persona relevance, remove noisy duplicates.

---

## 5. ICPs (Market Archetypes)

**Objective:** Surface the archetypes where the machine consistently works.
- Table per ICP with industry, size band, tech stack trigger, wolf persona, median win %, and sales cycle.
- Additional panels highlight Top 5 emerging ICPs, Top 5 declining ICPs, and ICPs with the highest subsidy burden.

---

## 6. Final Design Truth

Your system is no longer focused on personas and assets only. Instead, it now answers:
1. Who actually decides?
2. Who must be subsidized?
3. What belief is blocking a deal?
4. What is the highest ROI action right now?

Design+engineering should build interfaces that answer these questions directly with the data captured above.
