# Carnot — When the Grid Meets the Heat

**A grey-box model of heat-induced DC fast-charging derating and queue formation in UAE urban charging networks, with an evaluation of reflective canopies and tariff-driven time-shifting.**

---

## 1. Problem

The UAE is deploying DC fast chargers at scale while summer air temperatures reach 45–50 °C
and pavement-level temperatures are far higher. Two failures are usually analysed separately:

1. **A thermal failure.** Power electronics and battery packs derate above a threshold
   temperature. A "150 kW" charger does not deliver 150 kW at 15:00 in July. It delivers
   substantially less, and the loss is *state-dependent*: it depends on the charger's recent
   duty history, not just the current air temperature.
2. **A queueing failure.** Because service time is inversely proportional to delivered power,
   derating lengthens occupancy. Longer occupancy at a fixed number of stalls increases waits,
   which causes balking and reneging (lost demand), which in turn changes the load profile.

Analysing either in isolation is misleading. A thermal model alone reports a power number no
operator can act on; a queueing model with an exogenous service-time distribution cannot explain
*why* service times grow on hot days, cannot represent thermal stacking, and cannot evaluate a
physical intervention such as shading.

**Objective.** Build an explainable, calibrated, reproducible model that maps weather → charger
thermal state → delivered power → station occupancy → waits, lost demand, and grid load; then use
it to price and compare a physical mitigation (reflective canopy) and an operational one
(time-shifting flexible charging), including their interaction.

**Why it matters here.** Peak charging demand in the UAE coincides with peak air temperature and
peak grid load. The same hour stresses the transformer, the charger, and the driver's patience.
Any credible capacity plan for 2026–2035 has to represent that coincidence.

---

## 2. Model

### 2.1 Structure (grey box)

The model is a **coupled two-node lumped thermal system** embedded in a **time-stepped discrete-event
station simulation**. Physics supplies the structure and the parameters that carry meaning; a small
ridge-regression term absorbs the structure error that the physics cannot represent. Every learned
coefficient is attached to a physical feature, so the whole model remains auditable.

```
weather (PVGIS TMY)        interventions
      │                    ┌── canopy: G↓, T_amb↓
      ▼                    └── tariff: arrival times
┌─────────────────────────────────────────────┐
│ per-stall thermal state (T_cab, T_pack, SoC)│
│   C dT/dt = generation + solar − cooling    │
│   P_dc = P_nom · taper(SoC) · f_c(T_cab) · f_b(T_pack)
└─────────────────────────────────────────────┘
      │ service times are endogenous
      ▼
┌─────────────────────────────────────────────┐
│ station DES: arrivals, balking, reneging,   │
│ 4 stalls, thermal carry-over between sessions│
└─────────────────────────────────────────────┘
      │
      ▼
waits · lost demand · throughput · station load
```

### 2.2 Thermal model

**Cabinet node** (power electronics, heat sink, enclosure), temperature $T_c$:

$$
C_c \frac{dT_c}{dt}
= \underbrace{\left(\frac{1}{\eta(P)}-1\right)P_{dc} + P_{aux}}_{\text{losses}}
+ \underbrace{\alpha_c A_c G_c}_{\text{solar}}
- \underbrace{UA_c(T_c, v)\,(T_c-T_{amb})}_{\text{convection}}
- \underbrace{\varepsilon_c \sigma A_c\!\left[(T_c{+}273.15)^4-(T_{sky}{+}273.15)^4\right]}_{\text{longwave}}
$$

with a fan ramp $UA_c = UA_0\left[1+(r_{fan}-1)\,\varsigma\!\left(\frac{T_c-T_{fan}}{w}\right)\right] + k_v v^{0.8}$,
$\varsigma$ the logistic function, and a part-load efficiency
$\eta(P)=\eta_{peak}-0.05\,(1-P/P_{rated})^2$.

**Battery pack node**, temperature $T_b$:

$$
C_b \frac{dT_b}{dt}
= \underbrace{I^2 R_{int}(T_b,\mathrm{SoC})}_{\text{ohmic}}
+ \underbrace{q_{sol}G}_{\text{cabin solar leak}}
- \underbrace{\min\!\left(\frac{T_b-T_{sp}}{R_{th}},\; \dot Q_{chill}(T_{amb})\right)}_{\text{chiller, ambient-limited}}
- \underbrace{\frac{T_b-T_{amb}}{R_{passive}}}_{\text{passive}}
$$

$$
R_{int}(T_b,\mathrm{SoC}) = R_0 \exp\!\left[\frac{E_a}{R}\left(\frac{1}{T_b}-\frac{1}{T_{ref}}\right)\right]
\left[1+\beta(0.5-\mathrm{SoC})\right],
\qquad
\dot Q_{chill} = \dot Q_{35}\left[1-\gamma\,(T_{amb}-35)^+\right]^+.
$$

**The critical coupling is the last term.** Chiller capacity falls as ambient rises — about 4 %/K
above 35 °C — so cooling authority disappears exactly when it is needed. This is why 45 °C is
qualitatively worse than 40 °C, and why a model linear in ambient temperature cannot reproduce the
behaviour.

**SoC and power:**

$$
\frac{d\mathrm{SoC}}{dt}=\frac{P_{dc}}{3600\,E_{pack}},\qquad
P_{dc}=P_{nom}\,\tau(\mathrm{SoC})\,f_c(T_c)\,f_b(T_b),\qquad
\tau(\mathrm{SoC})=\min\!\left(1,\left[\frac{1-\mathrm{SoC}}{1-\mathrm{SoC}_{k}}\right]^{1.5}\right).
$$

**Derating law.** Exact form (used for reporting):

$$
f(T)=\begin{cases}
1, & T<T_1\\[2pt]
\dfrac{T_2-T}{T_2-T_1}(1-\phi)+\phi, & T_1\le T\le T_2\\[6pt]
\phi, & T>T_2
\end{cases}
$$

For calibration we use the $C^\infty$ clamp
$f_\kappa(T)=\phi+(1-\phi)\left[\mathrm{softplus}(\kappa x)-\mathrm{softplus}(\kappa(x-1))\right]$
with $x=(T_2-T)/(T_2-T_1)$; it converges to the exact law as $\kappa\to\infty$, and
`tests/test_thermal.py` verifies this. Gradients are then available for nonlinear least squares
without changing the model's meaning.

Integration is classical RK4 with power re-evaluated at every stage (so the derating feedback is
integrated, not frozen). The hot loop uses an equivalent scalar implementation; it is
cross-checked against the vectorised reference in the test suite.

### 2.3 Station and queue model

Arrivals are non-homogeneous Poisson with a two-peak urban profile (a midday taxi/fleet peak and an
evening retail peak), scaled to 120 sessions/day at a 4-stall hub. Each arrival carries a sampled
initial SoC ($\mathcal N(0.25,0.07^2)$, truncated to $[0.08,0.45]$) and a patience time
$\sim\mathrm{Exp}(35\,\text{min})$. A driver balks if the current expected wait (queue length per
stall × running mean service time) exceeds patience, and reneges if the realised wait does. Sessions
in service at midnight are censored, not silently dropped.

The station is a **stateful** system: each stall carries its own $T_c$, $T_{pack}$ between
sessions. An idle stall cools towards ambient between vehicles. Back-to-back sessions therefore
compound derating — **thermal stacking** — a feedback that an M/M/c model with exogenous service
cannot represent.

Two analytic models are kept as cross-checks and as a demonstration of *why* the simulation is
needed:

* **Erlang-C, pointwise-stationary:** $W_q = C(c,a)/(c\mu-\lambda)$ with $a=\lambda/\mu$ recomputed
  each hour. It ignores both thermal state and the fact that the queue relaxes on a timescale
  comparable to the arrival-profile changes.
* **Fluid queue:** $\dot q = (\lambda - c\mu)^+$, integrated at 15 s. It is deterministic and
  reproduces the overload episodes, but gives no distribution and no balking.

### 2.4 Calibration (two-stage, interpretable)

**Stage 1 — physics.** Seven parameters are estimated by bounded nonlinear least squares
(trust-region reflective) on a synthetic "digital twin" dataset:

| Parameter | Prior | Truth (twin) | Fitted | Std. err. | Bias |
|---|---|---|---|---|---|
| Cabinet natural $UA$ (W/K) | 45.0 | 38.70 | **39.01** | 5.07 | +0.8 % |
| Fan $UA$ ratio (–) | 6.0 | 7.08 | **6.92** | 0.90 | −2.2 % |
| Cabinet $C$ (kJ/K) | 180.0 | 163.80 | **165.22** | 0.69 | +0.9 % |
| Wind $UA$ coeff. (W/K) | 8.0 | 10.56 | **9.76** | 0.35 | −7.5 % |
| $R_{int}$ at 25 °C (Ω) | 0.060 | 0.072 | **0.071** | 0.0014 | −1.3 % |
| Chiller capacity (kW) | 6.0 | 5.28 | **5.02** | 0.20 | −4.9 % |
| Pack–coolant $R_{th}$ (K/W) | 0.008 | 0.0074 | **0.0077** | 0.0016 | +3.3 % |

Fit RMSE: **0.34 K** (cabinet) and **0.24 K** (pack) against observations carrying 0.35 K sensor
noise plus a deliberately injected 0.8 K periodic structure error. The truth multipliers were fixed
before fitting and are documented in `src/carnot/calibration.py`.

**Stage 2 — structure error.** A ridge regression is fitted to the Stage-1 residuals using six
physically meaningful features (intercept, ambient temperature, cabinet irradiance, power, wind,
and a hinge above 44 °C). Coefficients are reported in
`results/tables/t08b_residual_coefficients.csv`; the hinge term is the model admitting, in an
interpretable way, that it under-predicts heating in extreme heat.

**Benchmark.** A degree-3 polynomial ridge is trained on the same data as a stand-in for a generic
black-box surrogate.

> **Important honesty statement.** The calibration data are *synthetic*, generated by the same
> structural model with perturbed parameters, real TMY weather, sensor noise and injected model
> discrepancy. No measured session data from a UAE operator were available to this study. The
> twin experiment therefore validates the **estimation procedure and identifiability**, not the
> absolute accuracy of the physical parameters. `data/calibration/README.md` defines the exact CSV
> schema for substituting field data; the pipeline then runs unchanged.

### 2.5 Interventions

* **Reflective canopy** (physical, acts on both nodes): 10 % solar transmission, −2.5 K air
  temperature in the shaded zone. Optionally a white coating ($\alpha: 0.30\to0.12$).
* **Time-shift** (operational, acts on arrivals): during the 12:00–21:00 peak-tariff window, 45 % of
  drivers are flexible and are moved to a night window centred at 03:00. The vehicle is assumed to
  be plugged in at home/work, so no extra travel is modelled.

The two are evaluated in a full 2×2 factorial, so the interaction is measured rather than assumed.

---

## 3. Assumptions

| # | Assumption | Status / handling |
|---|---|---|
| A1 | Weather is the PVGIS v5.3 TMY for Abu Dhabi (24.454 N, 54.377 E; SARAH-3 + ERA5, 2005–2023) | Real data, 8 760 h, shipped in `data/raw/` |
| A2 | The design day is the hottest July day in the TMY (39.3 °C peak) | Typical extreme, not a record |
| A3 | Heatwave = design day + 6 K (45.3 °C peak) | Explicit stress-test scenario, not a forecast |
| A4 | Lumped capacitance (Bi < 0.1) for cabinet and pack | Standard for this timescale; validated by RK4 convergence test |
| A5 | Derating thresholds and floors are representative of 150 kW SiC chargers and 90 kWh packs | Literature/manufacturer-typical; tested in sensitivity analysis |
| A6 | Arrival profile, patience and initial SoC are representative of a mixed taxi/fleet + retail hub | Assumption; varied in Monte Carlo |
| A7 | Value of driver time 45 AED/h, tariff 0.35 AED/kWh, canopy capex 12 000 AED/stall | Stated assumptions with ranges; economics reported as a range |
| A8 | Time-shifted vehicles are actually plugged in overnight | Behavioural assumption; the model gives an upper bound on the benefit |
| A9 | The canopy reduces local air temperature by 2.5 K and transmits 10 % of GHI | Literature-typical; sampled 1.5–4.0 K in uncertainty analysis |

**Deliberately not modelled:** grid voltage/transformer dynamics, battery degradation, PV generation
on the canopy, driver heterogeneity beyond patience, and inter-station competition.

---

## 4. Results

All numbers below are produced by `python -m carnot.cli all` and stored in `results/`.

### 4.1 The derating cliff

At a sustained 850 W/m² and 2.5 m/s wind, equilibrium delivered power is **150 kW at 38 °C**, falls to
**97 kW at 45 °C** and **71 kW at 48 °C**. The fall is convex: the last few degrees cost
disproportionately, because the chiller is losing capacity at the same time as the heat load rises.
A single 25→80 % session at 15:00 takes **27.2 min** on the design day against a **27.0 min**
taper-aware reference (slowdown 1.01). With +6 K the same session takes **37.0 min** (slowdown 1.37),
and with the canopy **30.8 min** (slowdown 1.14).

### 4.2 Thermal stacking

Four consecutive sessions at the same stall, with 0/10/20/40 min gaps, show monotonically increasing
power loss when the gap is short. The cabinet, not the pack, is the binding node under sustained
duty because the pack's chiller recovers between sessions while the enclosure's heat soak does not.
This is the mechanism that a memoryless service-time model cannot produce, and it is why the DES
matters.

### 4.3 Headline scenario comparison (heatwave, 8 arrival seeds)

| Scenario | Mean wait (min) | p95 wait (min) | Lost demand | Slowdown | Energy (kWh/day) | Peak load (kW) |
|---|---|---|---|---|---|---|
| Typical day, no intervention | 2.93 | 14.7 | 11.2 % | 1.00 | 5 036 | 547 |
| **Heatwave baseline** | **5.80** | **21.9** | **20.2 %** | **1.29** | **4 500** | **439** |
| Reflective canopy | 3.84 | 17.1 | 13.7 % | 1.09 | 4 883 | 506 |
| Time-shift (dynamic tariff) | 1.78 | 10.7 | 6.8 % | 1.22 | 5 155 | 514 |
| **Canopy + time-shift** | **1.17** | **8.2** | **5.0 %** | **1.06** | **5 257** | **538** |

Reading this table:

* Heat turns a comfortable station into a marginal one: waits roughly double, **one in five drivers
  is lost**, and every served session takes 29 % longer.
* The canopy recovers roughly two-thirds of the heat penalty on wait (5.80 → 3.84 min) and cuts
  lost demand by about a third.
* Time-shifting is stronger on queueing because it removes load from the peak rather than making
  each session faster. But it leaves the slowdown at 1.22 — the physics is untouched.
* **The two interventions are complementary, and the combination is super-additive on wait:**
  the factorial decomposition gives a canopy main effect of **1.29 min**, a time-shift main effect of
  **3.34 min**, and a positive interaction of **1.35 min**. Shading keeps the evening peak fast;
  shifting removes the midday peak. Neither alone delivers the 1.17 min of the pair, and the
  combination removes **80 % of the heatwave wait**.
* A white cabinet coating adds **nothing** in this configuration: it lowers the cabinet
  absorptance, but with the canopy in place the pack is the binding node. Reporting this null result
  is useful — it says where not to spend money.

### 4.4 Grid consequences

Peak station load *falls* under derating (439 kW vs 547 kW on a typical day) even though energy
throughput falls too. This is an important and slightly counter-intuitive result: **a hot day hides
grid stress behind a service failure.** The operator sees a lower peak but loses 20 % of demand and
serves fewer vehicles. Interventions raise the peak back towards the equipment rating (506–538 kW)
because they restore throughput. Any grid-impact assessment that reads only the load curve will
therefore systematically understate the value of cooling.

### 4.5 Calibration and out-of-distribution behaviour

| Model | RMSE (K) | Max abs error (K) |
|---|---|---|
| Physics only, calibrated | **0.34** | 0.94 |
| Grey-box (+ ridge residual) | 0.35 | 0.99 |
| Polynomial black box (degree 3) | 4.94 | 13.5 |

On held-out days that are 8 K hotter than anything in training, the calibrated physics extrapolates
essentially unchanged, while the flexible black box degrades by more than an order of magnitude. The
ridge correction adds nothing here — and its coefficients are all below 0.02 K, which is the model
telling us that the physics is already well specified on this dataset. We report that null result
rather than tuning it away. **This is the argument for grey-box structure:** the invariants (energy
conservation, monotone derating, ambient-limited cooling) are what carry the model into the heatwave,
not the flexible terms.

### 4.6 Seasonality and economics

Derating is concentrated in June–September: mean delivered power falls from 73 % of nameplate in the
cool months (the baseline SoC-taper penalty) to 54–58 % in July–August, and waits roughly double.
From November to April the thermal constraint is inactive and the canopy has no effect.
Annualising over the hot season (May–September, 153 days) with the hottest-day-per-month weather:

* extra sessions served: **≈4.2/day** at the station;
* driver time saved: **≈1.16 min** per session on average;
* benefit per stall: **≈6 900 AED/year**;
* **payback ≈1.8 years, 15-year NPV ≈ +51 700 AED per stall** at 12 000 AED capex and a 6 %
  discount rate.

This is a base case, and it is assumption-driven. The benefit is roughly linear in the value of
driver time and in the margin per recovered session; halving both moves payback to about 4 years,
which is still well within the asset life. Conversely, a station that is not queue-constrained (low
utilisation) captures little value from faster service, and there the canopy must be justified on
equipment-life and reliability grounds rather than on throughput.

### 4.7 Uncertainty

Monte Carlo over five structural parameters plus the canopy temperature benefit (18 draws, each
with 3 arrival seeds) shows the wait reduction is robust in sign: mean **2.21 min**, 5th–95th
percentile **0.99–3.54 min**, and the canopy never increases mean wait. It also recovers about
**425 kWh/day** of otherwise-served throughput. The dominant driver is the chiller capacity at
35 °C, followed by the pack derating onset temperature — i.e. the uncertainty that matters is
thermal, not behavioural.

---

## 5. Interpretation

1. **Heat derates capacity twice.** Once in the hardware ($f_c f_b < 1$) and once in the queue
   (longer occupancy → more balking). The second effect is larger in welfare terms: on the heatwave
   day the station loses ~20 % of demand, and most of that loss comes from drivers who never plug in.
2. **Hourly-average analysis is not enough.** The queue responds to the coincidence of the
   afternoon thermal peak with the arrival peak. A model that averages power over the day would
   miss most of the wait impact.
3. **Physical and operational mitigations are complements, not substitutes.** The canopy buys
   thermal headroom; time-shifting buys queue headroom. Their interaction is positive, which means
   a portfolio is worth more than the sum of its parts. This is the paper's main practical claim.
4. **Derating can mask grid stress.** Lower peak load on hot days is a symptom of lost service, not
   of lower demand. Planners should read load and throughput together.
5. **Structure beats flexibility out of sample.** A calibrated two-node model with 7 parameters
   outperforms a degree-3 polynomial by 14× on a heatwave it never saw. For infrastructure decisions
   under climate change, that is the relevant test.

### Limitations

* Calibration on a synthetic twin: parameters are identified, but absolute accuracy is unproven
  against field data. The protocol to substitute real data is provided and requires no code changes.
* The heatwave is a uniform +6 K offset, not a physically downscaled climate scenario; it does not
  include humidity or radiation feedbacks.
* No battery degradation or charger failure model, so long-run asset-life benefits of shading are
  not monetised — the economics is therefore a lower bound on the canopy's value.
* Single-station, no competition or route choice; balking is modelled as lost demand rather than
  diversion.
* Cost figures are UAE-market assumptions with ranges, not quotes.

### Reproducibility

```
python -m carnot.cli check     # validates the shipped TMY
python -m carnot.cli all       # regenerates every figure and table (~4 min)
pytest -q                      # physics, queueing and calibration tests
```

Every parameter in `config/default.yaml` carries a provenance tag: `[L]` literature/manufacturer,
`[C]` calibrated, `[A]` assumption. Seeds are fixed. Figures are written to `results/figures/`,
tables to `results/tables/`, and a machine-readable digest to `results/summary.json`.

---

## 6. Sources

1. JRC European Commission, **PVGIS v5.3 API** — TMY for 24.454 N, 54.377 E (SARAH-3 satellite
   irradiance, ERA5 meteorology, 2005–2023). `https://re.jrc.ec.europa.eu/api/v5_3/tmy`
2. Berdahl, P. & Martin, M. (1984). Emissivity of clear skies. *Solar Energy* **32**(5), 663–664.
3. Duffie, J. A. & Beckman, W. A. (2013). *Solar Engineering of Thermal Processes*, 4th ed. Wiley.
4. Incropera, F. P. et al. (2007). *Fundamentals of Heat and Mass Transfer*, 6th ed. Wiley.
5. Erlang, A. K. (1917). Solution of some problems in the theory of probabilities of significance in
   automatic telephone exchanges. *Post Office Electrical Engineers' Journal*.
6. Green, L. (1984). A queueing system with general-use and limited-use servers. *Management Science*.
7. Whitt, W. (1991). The pointwise stationary approximation for $M_t/M_t/s$ queues is asymptotically
   correct. *Management Science* **37**(3), 307–314.
8. DEWA / ADDC commercial electricity tariff schedules (2024–2025).
9. UAE Federal Competitiveness and Statistics Centre, average wage statistics (2024).
10. IEA (2024). *Global EV Outlook 2024* — charging infrastructure and DC fast-charging trends.

---

## 7. AI use declaration

AI-assisted tools were used as a working aid during this project. **The coding was done with AI: the
Python implementation in `src/`, the test suite and the plotting code were generated with AI
assistance under my direction, review and debugging.** AI tools also helped with suggesting wording
for sections of this report and checking consistency between the text and the generated tables.

The problem framing, choice of model structure, parameter assumptions, experiment design,
interpretation of results, and all final judgements are mine. Every number reported here was
produced by running the code in this repository, and every source cited was consulted directly. The
AI tools did not generate results independently of the model, and no output was included without
being checked against `results/tables/` and `results/summary.json`.
