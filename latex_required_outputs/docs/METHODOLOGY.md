# Experimental Methodology

This document restates the methodology of the research paper in executable form and records design decisions that prevent data leakage.

## 1. Problem Type

Continuous **regression / time-series forecasting**.
No classification metrics (accuracy, ROC-AUC, confusion matrices) are used.

## 2. Data Handling

- Schemas and expected dimensions are taken directly from the paper (see `src/data/schema.py`).
- Real files are **not** redistributed; a synthetic generator produces series that match the target names, temporal ordering and approximate statistical behaviour of the nine aligned series used in the extended-data experiment.
- When real data are supplied they must be loaded with the same chronological discipline.

## 3. Chronological Split

Default ratios (configurable):

- Train 60 %
- Validation 20 %
- Test 20 %

Splits are performed after sorting by timestamp. Future observations never appear in earlier partitions.

## 4. Feature Engineering (Leakage-Free)

- Lag features: `t-1 … t-L` (default L ∈ {1,2,3,6,12})
- Rolling statistics: mean / std / min / max over windows that are shifted by one step (only past values)
- Calendar features with optional cyclic encoding (hour, day-of-week, month)
- StandardScaler and SimpleImputer are **fit exclusively on the training partition** and then applied to validation and test.

## 5. Models

### Baseline
Lag-1 persistence: \(\hat{y}_{t} = y_{t-1}\)

### Machine Learning
- HistGradientBoostingRegressor (final model family selected in the paper’s extended experiment)
- XGBRegressor
- RandomForestRegressor
- ExtraTreesRegressor
- GradientBoostingRegressor
- LGBMRegressor (optional)

### Deep Learning
- MLP, Residual MLP, LSTM, GRU, 1D-CNN (PyTorch)
- Early stopping on validation loss

### Hybrid
Time-ordered stacking:

1. Base models trained on the training partition
2. Predictions generated on the validation partition (OOF)
3. Meta-learner (Ridge) fitted on the OOF matrix + validation targets
4. Bases re-fit on train+val for final test inference

Test data are never seen by the meta-learner during training.

## 6. Metrics

Reported for every series and every model:

- RMSE
- MAE
- \(R^2\)
- Normalised RMSE (by range of the series)

Relative RMSE improvement versus persistence is also computed.

## 7. Agentic AI Decision Support

- Explicit rule base covering PUE, inlet temperature, cooling-supply drift, IT-power spikes, low ΔT and high model uncertainty.
- Recommendations are prioritised and always flagged for human review.
- The system never claims autonomous control of physical equipment.

## 8. Ablation & Comparison Tables

The pipeline produces:

1. Per-series model comparison tables
2. Grand mean comparison across all series
3. Best-model-per-series table
4. Feature-importance and residual plots
5. Agentic recommendation reports

All tables and figures are written under `outputs/`.
