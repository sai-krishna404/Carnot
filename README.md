# Carnot

**Heat-induced DC fast-charging derating and queueing in UAE urban charging networks — a grey-box model, with canopy and time-shift interventions.**

This repository contains everything needed to regenerate every number, figure and table in
`report/report.md`.

## Quick start

```bash
pip install -r requirements.txt
export PYTHONPATH=src          # Windows: $env:PYTHONPATH="...\Carnot\src"

python -m carnot.cli check     # validate the shipped TMY data
python -m carnot.cli all       # regenerate all results (~2.5 min)
pytest -q                      # physics, queueing and calibration tests
```

Outputs:

| Path | Contents |
|---|---|
| `results/figures/` | 15 figures (PNG, 220 dpi) |
| `results/tables/` | 12 CSV tables, quoted directly in the report |
| `results/summary.json` | machine-readable digest of the whole run |
| `data/processed/` | cleaned 8 760-hour TMY with solar geometry and sky temperature |

## What the model does

Two coupled lumped thermal nodes (charger cabinet and battery pack) with an **ambient-limited
chiller** are integrated with RK4 inside a **time-stepped discrete-event simulation** of a 4-stall
station. Service time is therefore endogenous: it is produced by the thermal state of each stall,
which carries over between sessions. The station model includes balking, reneging, thermal stacking,
non-homogeneous arrivals, and a full 2×2 factorial of a reflective canopy and tariff-driven
time-shifting.

Formal equations, assumptions and results are in `report/report.md`.

## Layout

```
config/default.yaml        every parameter, tagged [L] literature, [C] calibrated, [A] assumption
data/raw/                  PVGIS v5.3 TMY for Abu Dhabi (unmodified download)
data/processed/            cleaned TMY + solar geometry
src/carnot/thermal.py      heat balances, derating laws, session simulation, steady state
src/carnot/queueing.py     station DES, Erlang-C and fluid-queue cross-checks
src/carnot/calibration.py  synthetic twin, NLS fit, ridge residual learner, black-box benchmark
src/carnot/experiments.py  experiments E1-E7 (all figures and tables)
src/carnot/plotting.py     figure code
src/carnot/climate.py      TMY parsing, solar position, plane-of-array, day selection
src/carnot/metrics.py      operational and grid metrics
tests/                     pytest suite
report/report.md           the report
```

## Calibration data

The calibration is run on a **documented synthetic digital twin**, not on measured field sessions.
`data/calibration/README.md` specifies the CSV schema for substituting real measurements; no code
changes are required, only pointing `calibration.data_path` at the file.

## Reproducibility

* Fixed seeds (`meta.seed`), fixed arrival seeds per scenario.
* Runtime ~145 s on a laptop CPU.
* All assumptions that are not literature values are tagged `[A]` and varied in the uncertainty
  experiment.
