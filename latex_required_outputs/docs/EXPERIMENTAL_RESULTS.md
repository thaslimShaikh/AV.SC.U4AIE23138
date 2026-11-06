# Experimental Results (Synthetic Smoke-Test Execution)

**IMPORTANT:** Real dataset files were **not present** in the execution environment.
All numeric results in this document and in `outputs/` / `results/` were produced from
synthetic time-series that match the **schemas, target names and temporal structure**
described in the manuscript. They are **not** the experimental numbers reported in the paper
(e.g. RMSE 1.5211, R² 0.9524, 13.01 % improvement).

## Execution environment
- Date: 2026-11-06
- Seed: 42
- Packages available: numpy, pandas, matplotlib, seaborn, PyTorch, PyYAML
- Packages unavailable (network blocked): scikit-learn, xgboost, lightgbm
- Pure-NumPy / PyTorch substitutes were used for ML models.

## Datasets actually processed
Nine synthetic series matching the paper’s extended-experiment targets:
Cooling_Return_Average, Cooling_Supply_Average, Inlet_Temp_Avg, Outlet_Avg_Temp,
Overall_Coolant_Supply_Temp, Overall_average_Coolant_Return_Temp,
SubLoop1/2/3_Coolant_Return_Temp (n=2500 each).

Four original-family stubs also generated (NLR, Danish, NREL RSF, Thai).

## Models ranked by macro NRMSE (synthetic)

| Rank | Model | Category | Macro NRMSE | Macro R² |
|------|-------|----------|-------------|----------|
| 1 | persistence | baseline | 0.080 | 0.826 |
| 2 | hybrid_stack | hybrid | 0.089 | 0.786 |
| 3 | ridge | ml | 0.100 | 0.762 |
| 4 | hist_gradient_boosting | ml | 0.103 | 0.746 |
| 5 | gradient_boosting | ml | 0.103 | 0.745 |

On this synthetic data the lag-1 persistence baseline is already strong (high autocorrelation).
The hybrid stack is the best *learned* model. Deep models under-trained in the short epoch budget
and are not competitive here; longer training and real data are required for a fair DL comparison.

## Agentic AI
Six operational scenarios evaluated (high PUE, high inlet temperature, cooling drift,
IT power spike, low ΔT, high uncertainty). Recommendations generated with explicit evidence
references; all require human review.

## What remains for a full research claim
1. Obtain the real measurement files listed in the manuscript.
2. Install the full `requirements.txt` stack.
3. Re-run `python -m src.pipeline_complete` (or the sklearn-based `src.pipeline`).
4. Replace all synthetic metrics with real ones before citing numbers in the paper.
