# 📐 Mathematical Breakdown Report
## F1 Bayesian Predictor & Live Tracker 2026
> **Use Case**: Monaco Grand Prix 2026 — Qualifying & Race Simulation  
> **Scope**: Complete mathematical pipeline from raw FP3 telemetry → starting grid prediction → race probability

---

## 🧭 System Overview

The system applies a **three-layer probabilistic pipeline**:

```
FP3/SQ Telemetry
    │
    ▼
┌──────────────────────────────────────┐
│  LAYER 1: Bayesian ELO Update        │  ← Bayes' Theorem (head-to-head)
│  Prior × Likelihood → Posterior ELO  │
└─────────────────┬────────────────────┘
                  │ Updated ELO (Posterior)
                  ▼
┌──────────────────────────────────────┐
│  LAYER 2: LightGBM LTR + Quantile    │  ← LambdaMART + Quantile Regression
│  Qualifying Grid Rank + Lap Time CI  │
└─────────────────┬────────────────────┘
                  │ Predicted Starting Grid
                  ▼
┌──────────────────────────────────────┐
│  LAYER 3: Monte Carlo Race Simulator │  ← Vectorized NumPy (10,000 sims)
│  Win % / Podium % / DNF %            │
└──────────────────────────────────────┘
```

---

## 🔵 LAYER 1 — Bayesian ELO Update

### 1.1 Concept

Before each race weekend, every driver carries a **Prior ELO rating** representing their historical baseline performance. Once FP3 data arrives, the system runs a **Bayesian update** using head-to-head teammate lap time comparisons to produce a **Posterior ELO** reflecting their current weekend form.

> **Why teammate-only comparison?** Because both drivers share the same car specification, circuit allocation, and weather. A teammate head-to-head isolates *driver skill* from *car advantage*, which is the purest signal of current form.

### 1.2 Prior — $P(A)$

Each driver's base prior is defined in `src/utils.py`:

$$\text{Prior}_{\text{driver}} = \text{base\_elo} + \text{junior\_bonus}$$

| Driver | Base ELO | Junior Bonus | Effective Prior |
|---|---|---|---|
| Kimi Antonelli (ANT) | 1860 | 0 | **1860** |
| George Russell (RUS) | 1840 | 0 | **1840** |
| Max Verstappen (VER) | 1900 | 0 | **1900** |
| Gabriel Bortoleto (BOR) | 1400 | +50 | **1450** |
| Arvid Lindblad (LIN) | 1380 | +40 | **1420** |

> Junior bonus compensates for lack of historical head-to-head race data, reflecting the premium placed on F2/F3 dominant performances.

### 1.3 Expected Performance — $E_A$ (Likelihood function)

Given two teammates A and B with ELOs $R_A$ and $R_B$:

$$\boxed{E_A = \frac{1}{1 + 10^{\dfrac{R_B - R_A}{400}}}}$$

This is the continuous ELO expected performance formula. It returns the probability (0–1) that driver A outperforms driver B in FP3.

**Example: Monaco FP3 — ANT vs. RUS (Mercedes)**

| Parameter | Value |
|---|---|
| ANT Prior ELO ($R_A$) | 1860 |
| RUS Prior ELO ($R_B$) | 1840 |
| $E_{ANT} = \frac{1}{1 + 10^{(1840-1860)/400}}$ | $= \frac{1}{1 + 10^{-0.05}} = \mathbf{0.529}$ |

→ Prior says **Antonelli has 52.9% chance** of beating Russell in FP3.

### 1.4 Likelihood — $P(B|A)$: FP3 Observed Outcome

| | ANT | RUS |
|---|---|---|
| FP3 lap time | **1:12.430** | 1:12.520 |
| **Outcome ($S$)** | **1.0 (Won)** | 0.0 (Lost) |

### 1.5 Posterior Update — $P(A|B)$

Using the **Bayesian Elo Update Formula** with K-factor = 32:

$$\boxed{R'_A = R_A + K \cdot (S_A - E_A)}$$

**ANT (Won):**
$$R'_{ANT} = 1860 + 32 \times (1.0 - 0.529) = 1860 + 15 = \mathbf{1875}$$

**RUS (Lost):**
$$R'_{RUS} = 1840 + 32 \times (0.0 - 0.471) = 1840 - 15 = \mathbf{1825}$$

| Driver | Prior ELO | FP3 Result | Posterior ELO | Delta |
|---|---|---|---|---|
| ANT | 1860 | Won (faster) | **1875** | +15 |
| RUS | 1840 | Lost (slower) | **1825** | -15 |

> The 20-point raw gap between ANT and RUS widens to **50 points** after this Bayesian update, signaling ANT's dominance in current form at Monaco.

---

## 🟡 LAYER 2 — LightGBM Qualifying Grid Predictor

### 2.1 Architecture

The model combines two algorithms:

| Sub-model | Algorithm | Purpose |
|---|---|---|
| **Grid Ranker** | LightGBM LambdaMART (LTR) | Predict relative P1–P22 starting order |
| **Time Estimator** | LightGBM Quantile Regression (×3) | Predict lap time credible intervals |

### 2.2 Hybrid Pacing Score

Before feeding data to LightGBM, each driver receives a **Hybrid Qualifying Score** that combines three signals:

$$\boxed{S_{total} = 0.50 \cdot S_{ELO} + 0.35 \cdot S_{car} + 0.15 \cdot S_{FP3} + \varepsilon}$$

Where:

#### A. ELO Score $S_{ELO}$

Linear normalization of Updated Posterior ELO (range 1300–2000):

$$S_{ELO} = \text{clip}\left(\frac{ELO_{posterior} - 1300}{700}, 0, 1\right)$$

**ANT example:**
$$S_{ELO}^{ANT} = \frac{1875 - 1300}{700} = \frac{575}{700} = \mathbf{0.821}$$

#### B. Constructor Car Score $S_{car}$

Normalized from constructor pace offsets (range −0.65 → +0.50, lower is faster):

$$S_{car} = \text{clip}\left(\frac{-\text{pace\_offset} + 0.50}{1.15}, 0, 1\right)$$

| Team | Pace Offset | $S_{car}$ |
|---|---|---|
| Mercedes | −0.65 | $\frac{0.65 + 0.50}{1.15} = \mathbf{1.00}$ |
| McLaren | −0.55 | $\mathbf{0.913}$ |
| Ferrari | −0.50 | $\mathbf{0.870}$ |
| Red Bull | −0.15 | $\mathbf{0.565}$ |
| Aston Martin | +0.35 | $\mathbf{0.130}$ |

#### C. Practice Pace Score $S_{FP3}$

FP3 times are first **tyre-normalized** (removing compound advantage):

$$t_{norm} = t_{raw} - \Delta_{tyre}$$

Where $\Delta_{tyre}$: Soft = 0.0s, Medium = +0.6s, Hard = +1.2s

Then the score is calculated relative to the best normalized time in the session:

$$S_{FP3} = \text{clip}\left(1 - \frac{t_{norm,driver} - t_{norm,best}}{3.0}, 0, 1\right)$$

**ANT example** (Monaco FP3, Soft tyres, 1:12.430 = 72.43s):
- Best normalized FP3 = ANT himself at 72.43s
- $S_{FP3}^{ANT} = 1 - \frac{72.43 - 72.43}{3.0} = \mathbf{1.000}$

#### D. Stochastic Session Variance $\varepsilon$

To simulate organic qualifying unpredictability (track evolution, lock-ups, yellow flags):

$$\varepsilon \sim \mathcal{N}(0, \sigma^2), \quad \sigma = 0.025$$

#### E. Combined Score — Monaco Example

| Driver | $S_{ELO}$ | $S_{car}$ | $S_{FP3}$ | $\varepsilon$ | **$S_{total}$** | Grid |
|---|---|---|---|---|---|---|
| ANT | 0.821 | 1.000 | 1.000 | +0.008 | **0.868** | **P1** |
| RUS | 0.750 | 1.000 | 0.956 | −0.003 | **0.840** | **P2** |
| NOR | 0.814 | 0.913 | 0.920 | +0.012 | **0.840** | **P3** |

### 2.3 LambdaMART Ranker (LTR)

After scoring, the LightGBM Ranker fine-tunes the relative ordering using the following feature vector:

$$\mathbf{x} = [\text{fp3\_avg}, \text{speed\_trap}, \text{track\_temp}, \text{driver\_elo}, \text{tyre\_code}, \text{rain\_intensity}]$$

**Objective:** Maximize NDCG (Normalized Discounted Cumulative Gain):

$$NDCG@k = \frac{DCG@k}{IDCG@k}$$

$$DCG@k = \sum_{i=1}^{k} \frac{2^{rel_i} - 1}{\log_2(i+1)}$$

Where $rel_i$ is the relevance score (rank in session) — fastest driver gets relevance = 22, slowest gets 1.

### 2.4 Quantile Regression — Credible Intervals

Three separate LightGBM Regressors with `objective="quantile"` estimate the **90% Bayesian Credible Interval** for each driver's Q3 lap time:

| Quantile ($\alpha$) | Meaning | Formula in Code |
|---|---|---|
| 0.10 | **Best-case** (fastest 10% of scenarios) | `y_time_q10` |
| 0.50 | **Median** (expected central estimate) | `y_time_q50` |
| 0.90 | **Worst-case** (slowest 10% of scenarios) | `y_time_q90` |

**Loss function for quantile $\alpha$:**

$$\mathcal{L}(y, \hat{y}; \alpha) = \begin{cases} \alpha \cdot (y - \hat{y}) & \text{if } y \geq \hat{y} \\ (1-\alpha) \cdot (\hat{y} - y) & \text{if } y < \hat{y} \end{cases}$$

**Base lap time anchor** for Monaco (circuit-specific):

$$t_{pole} = L_{km} \times s_{type}$$

| Circuit Type | $s_{type}$ (sec/km) |
|---|---|
| Speed-Drag | 13.5 |
| Balanced | 14.2 |
| Traction-Braking | 14.5 |
| Downforce-High | 15.2 |

**Monaco** = 3.337 km × 15.2 (downforce-high) = **50.72s** base pole time

**Gap model between positions:**

$$\Delta t_{rank} = \frac{0.12}{1 + rank \times 0.04} \times U[0.7, 1.3] \times g_{weather}$$

Where $g_{weather} = 1.0 + 1.5 \times rain\_intensity$

**Credible Interval width:**

$$CI_{driver} = \begin{cases} 1.0 \times g_{weather} & \text{if veteran} \\ 1.45 \times g_{weather} & \text{if rookie} \end{cases}$$

**Monaco Q3 Predictions:**

| Pos | Driver | Best Case | Median | Worst Case |
|---|---|---|---|---|
| P1 | ANT | 1:10.320 | 1:10.720 | 1:11.320 |
| P2 | RUS | 1:10.426 | 1:10.834 | 1:11.434 |
| P3 | NOR | 1:10.510 | 1:10.930 | 1:11.530 |

---

## 🔴 LAYER 3 — Vectorized Monte Carlo Race Simulator

### 3.1 State Initialization

The simulator creates matrices of shape $(N_{sims}, N_{drivers})$:

| Matrix | Shape | Meaning |
|---|---|---|
| `base_paces` | (10000, 22) | Fundamental lap time per driver per simulation |
| `total_race_times` | (10000, 22) | Cumulative race time each driver accumulates |
| `tyre_ages` | (10000, 22) | Current age of tyre (laps) per driver |
| `active_dnf` | (10000, 22) | Boolean: has driver retired? |

### 3.2 Base Pace Formula

For each driver $d$ in each simulation:

$$\boxed{T_{base,d} = 72.5 - (ELO_d - 1500) \times 0.003 + \text{pace\_offset}_d + \Delta_{weather}}$$

Where $\Delta_{weather}$: Dry = 0.0s, Damp = +8.0s, Wet = +16.0s

**Monaco Example (dry):**
| Driver | ELO | Pace Offset | $T_{base}$ |
|---|---|---|---|
| ANT | 1875 | −0.65 | $72.5 - (375 \times 0.003) + (−0.65) = \mathbf{70.725s}$ |
| RUS | 1825 | −0.65 | $72.5 - (325 \times 0.003) + (−0.65) = \mathbf{70.875s}$ |
| VER | 1900 | −0.15 | $72.5 - (400 \times 0.003) + (−0.15) = \mathbf{71.15s}$ |

### 3.3 Per-Lap Time Calculation

At every lap $l$, the simulator computes the lap time for all drivers across all simulations simultaneously:

$$\boxed{t_{l,d} = T_{base,d} + \Delta_{fuel}(l) + \Delta_{tyre}(l, d) + \varepsilon_d}$$

#### A. Fuel Drop Effect $\Delta_{fuel}$

As fuel burns, the car becomes lighter (approx. 0.04s faster per lap):

$$\Delta_{fuel}(l) = (L_{total} - l) \times (-0.04)$$

At Monaco (78 laps), Lap 1: $\Delta_{fuel} = 77 \times (−0.04) = −3.08s$, Lap 78: $0.0s$

#### B. Tyre Degradation Penalty $\Delta_{tyre}$

Non-linear wear model using a power law:

$$\Delta_{tyre}(l, d) = k_{compound} \times \text{age}^{1.8} \times 0.1$$

| Compound | $k_{wear}$ (Monaco) |
|---|---|
| Soft | 0.07 |
| Medium | 0.035 |
| Hard | 0.015 |

At Soft age = 10 laps: $\Delta_{tyre} = 0.07 \times 10^{1.8} \times 0.1 = 0.07 \times 63.1 \times 0.1 = \mathbf{0.442s}$

#### C. Pit Stop Threshold

Driver pits when tyre age exceeds compound-specific threshold:

| Compound | Stop Threshold |
|---|---|
| Soft | 16 laps |
| Medium | 26 laps |
| Hard | 38 laps |
| Intermediate | 24 laps |

Pit stop adds **+22.0 seconds** (includes pit lane speed limit, stop, and release time).

#### E. Driver Lap Time Noise $\varepsilon_d$

Per-lap random noise models human inconsistency:

$$\varepsilon_d \sim \mathcal{N}(0, \sigma_d^2)$$

| Driver Type | $\sigma_d$ (dry) |
|---|---|
| Veteran | 0.09s |
| Rookie | 0.18s |

### 3.4 Overtaking Model (DRS Simulation)

At each lap, positions are sorted and adjacent pairs are checked for overtaking eligibility:

$$\text{DRS Window} = \text{gap} < 0.8\text{s}$$

$$P(\text{overtake}) = \text{overtake\_index}_{circuit} \times 0.15$$

If both conditions are met:
$$t_{chase} \leftarrow t_{lead} - 0.3, \quad t_{lead} \leftarrow t_{chase} + 0.3$$

Monaco has `overtaking_index = 0.30` (lowest), so:
$$P(\text{overtake}) = 0.30 \times 0.15 = \mathbf{4.5\%}$$

### 3.5 Safety Car Model

Each lap, a probabilistic SC trigger based on circuit history:

$$P_{SC/lap} = \frac{p_{circuit} \times \mu_{weather}}{L_{total}}$$

Where $\mu_{weather}$: Dry = 1.0×, Damp = 1.5×, Wet = 2.5×

**Monaco (dry):** $p_{circuit} = 0.65$, $L = 78$  
$$P_{SC/lap} = \frac{0.65}{78} = \mathbf{0.83\%}$$

When triggered, all gaps are **compressed** to 1.0–1.5s intervals:
$$t_{i,\text{after SC}} = t_{leader} + \sum_{k=1}^{i} U[1.0, 1.5]$$

### 3.6 DNF Model

Each lap, every driver independently draws a random retirement:

$$P_{DNF/lap} = \frac{p_{dnf,circuit} \times \mu_{weather}}{L_{total}}$$

**Monaco (dry):** $p_{dnf} = 0.10$, $L = 78$  
$$P_{DNF/lap} = \frac{0.10}{78} \approx \mathbf{0.128\%/lap}$$

If DNF drawn: `total_race_times[sim, driver] = 999,999.0` (effectively last place)

### 3.7 Final Probability Compilation

After all $N_{sims} = 10,000$ simulations run to completion, final positions are extracted:

```
final_rankings = argsort(argsort(total_race_times, axis=1), axis=1) + 1
```

Then probabilities are computed as:

$$P_{win}(d) = \frac{\sum_{i=1}^{N} \mathbb{1}[rank_{i,d} = 1]}{N} \times 100$$

$$P_{podium}(d) = \frac{\sum_{i=1}^{N} \mathbb{1}[rank_{i,d} \leq 3]}{N} \times 100$$

$$P_{top10}(d) = \frac{\sum_{i=1}^{N} \mathbb{1}[rank_{i,d} \leq 10]}{N} \times 100$$

$$P_{DNF}(d) = \frac{\sum_{i=1}^{N} \mathbb{1}[\text{dnf}_{i,d}]}{N} \times 100$$

### 3.8 Monaco 2026 — Predicted Final Probabilities

| Driver | Team | Win % | Podium % | Top 10 % | DNF % |
|---|---|---|---|---|---|
| Kimi Antonelli | Mercedes | **54.2%** | 78.1% | 96.3% | 3.7% |
| George Russell | Mercedes | 18.7% | 68.4% | 95.1% | 4.9% |
| Lando Norris | McLaren | 12.3% | 55.2% | 94.8% | 5.2% |
| Charles Leclerc | Ferrari | 8.1% | 42.0% | 93.7% | 6.3% |
| Max Verstappen | Red Bull | 3.9% | 22.4% | 87.2% | 12.8% |

> Note: All probabilities above are illustrative outputs of the simulation model, not actual race results.

---

## 🔄 Complete Use Case Flow by Session State

The system adapts its **entire data pipeline** depending on the current Grand Prix state. Below are three concrete use cases using the same race: **Monaco Grand Prix 2026**, with **ANT vs. RUS** (Mercedes) as the anchor example.

---

### 📅 Scenario A: UPCOMING — Pre-Weekend (No Real Data Yet)

> **Trigger**: Current UTC time is before the first practice session start time.  
> **Example**: Monaco GP weekend hasn't started yet. FP1/FP2/FP3 have not run.

```
MONACO GP 2026 — Status: 📅 UPCOMING
══════════════════════════════════════════════

1. FP3 DATA SOURCE
   FastF1 API returns no session → generate_calibrated_fp3_data() triggers.
   Synthetic FP3 times generated from:
       t_generated = L_km × 14.5 + ε,  ε ~ N(0, σ_team²)
   Where σ_team reflects constructor pace offset variance.
   
   Example synthetic Monaco FP3 (3.337 km × 14.5 = 48.39s base):
   ANT: 72.430s (generated)  ← Based on Mercedes pace tier
   RUS: 72.520s (generated)

2. BAYESIAN ELO (Layer 1) — No Real Update
   UPCOMING mode: FP3 results are synthetic → ELO stays at PRIOR values.
   ANT Prior = 1860 (unchanged)
   RUS Prior = 1840 (unchanged)
   
   No Bayesian update fires. Posterior = Prior.

3. HYBRID QUALIFYING SCORE (Layer 2)
   ELO component uses raw base_elo:
   ANT: 0.50(0.814) + 0.35(1.00) + 0.15(1.00) + ε = 0.857 → P1 (predicted)
   RUS: 0.50(0.771) + 0.35(1.00) + 0.15(0.960) + ε = 0.829 → P2 (predicted)

4. QUALIFYING TIME PREDICTION
   Uses synthetic FP3 → predictions carry higher uncertainty bands.
   ANT: [70.32s, 70.72s, 71.70s]  ← Wider worst-case (less data confidence)
   RUS: [70.43s, 70.83s, 71.81s]

5. MONTE CARLO (Layer 3) — Pure Model Prediction
   Starting Grid: Fully predicted (no actual Saturday qualifying data)
   Base Paces derived entirely from base_elo + constructor pace_offset
   All 10,000 sims run from Lap 0 → Lap 78

6. FINAL OUTPUT
   ⚠️ These are pre-weekend probability estimates.
   ANT: Win ~48% | Podium ~72% | DNF 3.7%
   RUS: Win ~21% | Podium ~65% | DNF 4.9%
   
   [System note: Displayed as "Model Estimate — No Real Session Data"]
```

**Key Behavioral Differences in UPCOMING:**
| Component | Behavior |
|---|---|
| FP3 Data | ❌ FastF1 not called → `generate_calibrated_fp3_data()` synthetic fallback |
| ELO Update | ❌ No real update → Posterior = Prior |
| Qualifying Grid | ❌ Fully model-predicted, not actual Q3 results |
| Race Simulator | ✅ Runs 10,000 sims from Lap 0 |
| Uncertainty | 🔺 Higher — all signals are model-generated |

---

### 🔴 Scenario B: ONGOING — Mid-Race (Live Simulation)

> **Trigger**: Current UTC time is after race start but before race finish.  
> **Example**: Monaco GP is happening right now. It's Lap 38 of 78. ANT leads, RUS P3.

```
MONACO GP 2026 — Status: 🔴 ONGOING / LIVE
══════════════════════════════════════════════

1. LIVE DATA INGESTION
   FastF1 live API polls current session data every N seconds.
   Retrieved from Ergast + FastF1 combined endpoint:
   
   Lap 38 Snapshot:
   ┌──────┬──────┬──────────────┬────────────┬──────────┐
   │ Pos  │ Code │ Gap to Lead  │ Tyre       │ Tyre Age │
   ├──────┼──────┼──────────────┼────────────┼──────────┤
   │  P1  │ ANT  │ 0.000s       │ Hard       │ 12 laps  │
   │  P2  │ NOR  │ +4.821s      │ Hard       │ 10 laps  │
   │  P3  │ RUS  │ +7.340s      │ Hard       │ 14 laps  │
   │  P4  │ LEC  │ +12.100s     │ Medium     │ 8 laps   │
   └──────┴──────┴──────────────┴────────────┴──────────┘
   
   Known DNFs: VER (mechanical failure, Lap 22)

2. ACTIVE STATE INITIALIZATION
   Monte Carlo simulator receives live snapshot as `active_state`:
   
   active_state = {
     "gaps":      [0.0, 4.821, 7.340, 12.100, ...],  # cumulative gaps
     "tyre_ages": [12, 10, 14, 8, ...],               # laps on current tyre
     "pit_stops": [1, 1, 1, 0, ...],                  # stops taken
     "dnfs":      ["VER"]                              # confirmed retirements
   }
   
   Matrices initialized with REAL state (not zeros):
   total_race_times[:, ANT_idx] = 0.000s   (leader baseline)
   total_race_times[:, NOR_idx] = 4.821s
   total_race_times[:, RUS_idx] = 7.340s
   active_dnf[:, VER_idx]       = True     (forced DNF)

3. MID-RACE MONTE CARLO (Layer 3)
   current_lap = 38  → Simulator runs Lap 38 → Lap 78 only (40 remaining laps)
   
   Per remaining lap formula:
   t_l,d = T_base,d + Δ_fuel(l) + Δ_tyre(l,d) + ε_d
   
   Remaining fuel effect (40 laps left):
   Δ_fuel(38) = (78 - 38) × (-0.04) = -1.60s  [still significant fuel load]
   
   RUS tyre: Hard age 14 laps → Δ_tyre = 0.015 × 14^1.8 × 0.1 = 0.015 × 107.5 × 0.1 = 0.161s
   RUS tyre: If reaching Hard threshold (38 laps) at Lap 38+24=62 → pit at Lap 62

4. DNF ENFORCEMENT
   VER's DNF is locked in: active_dnf[:, VER_idx] = True across ALL 10,000 simulations.
   Unlike UPCOMING mode where VER has a chance, here VER cannot score.

5. SAFETY CAR CONTEXT
   If SC occurred at Lap 35 (3 laps ago), all gaps compressed to 1.0–1.5s.
   The active_state would already reflect this compressed field snapshot.

6. FINAL OUTPUT (Live Mid-Race)
   Remaining 40 laps simulated 10,000 times from Lap 38 state.
   ANT: Win 71.3% | Podium 88.4% | DNF 1.8%   ← Higher win % (already leading!)
   RUS: Win 11.2% | Podium 58.7% | DNF 2.1%
   NOR: Win 14.6% | Podium 62.9% | DNF 2.3%
   VER:  Win 0.0% | Podium  0.0% | DNF 100%   ← Confirmed DNF enforced
```

**Key Behavioral Differences in ONGOING:**
| Component | Behavior |
|---|---|
| FP3 Data | ✅ Already used at start of weekend → Posterior ELO already updated |
| Qualifying Grid | ✅ Real Saturday Q3 grid used as `starting_grid` |
| Race Simulator | ✅ Runs from `current_lap` → finish (not Lap 0) |
| Live Gaps | ✅ `active_state["gaps"]` initializes total_race_times with real intervals |
| Tyre Ages | ✅ Real laps-on-compound per driver |
| Known DNFs | ✅ `active_dnf` locked to True, win% = 0% for retired drivers |
| Uncertainty | 🔻 Lower — real track data anchors the simulation |

---

### ✅ Scenario C: DONE / RACE COMPLETED — Post-Race Analysis

> **Trigger**: Current UTC time is after the scheduled race end time.  
> **Example**: Monaco GP finished 2 hours ago. Full 78-lap race data is available.

```
MONACO GP 2026 — Status: ✅ RACE COMPLETED
══════════════════════════════════════════════

1. FULL RACE DATA RETRIEVAL (FastF1 API)
   Session.load() fetches complete race data:
   - All 78 laps for all 22 drivers
   - Sector times, gap evolution, safety car periods
   - Tyre stints, pit stop timing, compound sequences
   
   Final Race Result (actual):
   ┌──────┬──────┬──────────────────┬──────────────────────┐
   │ Pos  │ Code │ Finish Gap       │ Tyre Strategy        │
   ├──────┼──────┼──────────────────┼──────────────────────┤
   │  P1  │ ANT  │ WINNER           │ Soft (16) → Hard (62)│
   │  P2  │ NOR  │ +5.221s          │ Soft (18) → Hard (60)│
   │  P3  │ RUS  │ +9.870s          │ Soft (14) → Hard (64)│
   │  P18 │ VER  │ DNF (Lap 22)     │ Soft → DNF           │
   └──────┴──────┴──────────────────┴──────────────────────┘

2. TELEMETRY REPLAY (Tab 1: Live Practice & Telemetry)
   All 22 drivers' lap-by-lap telemetry is available:
   - Speed traces (km/h by distance)
   - Throttle & brake application curves
   - Mini-sector gap delta overlays
   
   Example post-race telemetry (ANT vs. RUS comparison):
   Speed at Sector 1 exit: ANT 264 km/h vs RUS 261 km/h  [ANT +3 km/h]
   Brake application Mirabeau: ANT 145m vs RUS 148m  [ANT brakes 3m later]

3. BAYESIAN ELO — POST-RACE RETROSPECTIVE
   System can optionally run a retrospective ELO update using race result:
   
   Race outcome (ANT P1 vs RUS P3):
   actual_ANT = 1.0, actual_RUS = 0.0
   E_ANT (pre-race) = 0.529
   
   R'_ANT = 1875 + 32 × (1.0 - 0.529) = 1875 + 15 = 1890
   R'_RUS = 1825 + 32 × (0.0 - 0.471) = 1825 - 15 = 1810
   
   These UPDATED posterior ELOs carry forward as the PRIOR for the NEXT race weekend.

4. QUALIFYING ACCURACY VALIDATION
   System compares pre-race qualifying prediction vs. actual Saturday Q3 result:
   
   | Predicted Grid | Actual Q3 Grid | Match |
   |---|---|---|
   | P1: ANT | P1: ANT | ✅ |
   | P2: RUS | P3: RUS | ⚠️ Off by 1 |
   | P3: NOR | P2: NOR | ⚠️ Off by 1 |
   | P4: LEC | P4: LEC | ✅ |
   
   Top-4 prediction accuracy: 2/4 exact, 4/4 within ±1 position.

5. RACE SIMULATOR RETROSPECTIVE
   Completed race serves as ground truth comparison vs. pre-race Monte Carlo:
   
   Pre-Race Prediction:     Actual Result:
   ANT Win 54.2%       →   ANT Won ✅   (probability materialized)
   VER DNF 12.8%       →   VER DNF ✅   (probability materialized)
   NOR Podium 55.2%    →   NOR P2  ✅   (probability materialized)

6. DISPLAY MODE (App UI Changes)
   - Tab 1 (Telemetry): Shows FULL race lap comparison, not just FP3.
   - Tab 2 (Qualifying): Shows actual Q3 times vs. predicted times side by side.
   - Tab 3 (Race Sim): Simulator re-runs with ACTUAL starting grid + tyre strategies.
     Shows how probability distribution would have played out if run from real data.
   - Sidebar: "RACE COMPLETED" badge shown with full lap count (e.g., "Finished 53/53 laps")
```

**Key Behavioral Differences in DONE:**
| Component | Behavior |
|---|---|
| FP3 / Practice Data | ✅ Full session data available and displayed |
| Qualifying Grid | ✅ Actual Q3 result used — shown alongside prediction accuracy |
| Race Simulator | ✅ Runs with real starting grid + real tyre strategies |
| Telemetry | ✅ Full 78-lap race telemetry available for all drivers |
| ELO Carry-forward | ✅ Post-race ELO becomes Prior for NEXT Grand Prix |
| Uncertainty | ✅ Lowest — all ground truth data used |

---

## 📊 Cross-Scenario Comparison

| Dimension | 📅 UPCOMING | 🔴 ONGOING | ✅ DONE |
|---|---|---|---|
| **FP3 Source** | Synthetic (generated) | Real FP3 (already loaded) | Real FP3 + Race |
| **ELO Posterior** | = Prior (no update) | Updated from FP3 result | Updated from Race result |
| **Qualifying Grid** | Model-predicted | Actual Q3 result | Actual Q3 result |
| **Sim Start Lap** | Lap 0 | Current live lap | Lap 0 (retrospective) |
| **Live Gaps** | None | Real telemetry gaps | N/A (race finished) |
| **Known DNFs** | None | Enforced (real) | All enforced (real) |
| **Output Type** | Pre-weekend forecast | Live win probability | Historical validation |
| **Confidence** | 🔴 Lowest | 🟡 Medium | 🟢 Highest |

---

## 📦 Dependencies Summary

| Library | Usage |
|---|---|
| `lightgbm` | LambdaMART Ranker + Quantile Regressors |
| `numpy` | Vectorized lap-by-lap race simulation matrices |
| `pandas` | Feature engineering and result tabulation |
| `fastf1` | Real FP3/SQ/Race data ingestion from official F1 API |
| `streamlit` | Real-time interactive web dashboard |
| `plotly` | Interactive telemetry, qualifying, and probability charts |

---

*Report generated for the F1 Bayesian Predictor & Live Tracker 2026 | fbudimannn/F1_PREDICTION*
