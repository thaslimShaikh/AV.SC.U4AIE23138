# Alignment between Codebase and Research Paper

| Paper Section | Code Location | Notes |
|---------------|---------------|-------|
| Abstract / Objectives | `README.md`, `docs/METHODOLOGY.md` | Objectives mirrored; no invented savings claims |
| Related Work | (literature only) | Not re-implemented |
| Materials & Methods – Original Datasets | `src/data/schema.py` (`ORIGINAL_SCHEMAS`) | Dimensions & targets taken from paper tables |
| Materials & Methods – Additional Datasets | `src/data/schema.py` (`ADDITIONAL_SCHEMAS`) | Multi-file / multi-sheet flags preserved |
| Preprocessing | `src/preprocessing/feature_engineering.py`, `splitting.py` | Chronological, lag/rolling from past only |
| Models (HGB, XGBoost, MLP, ResMLP, LSTM) | `src/models/ml_models.py`, `dl_models.py` | Extended with RF, ET, GB, LightGBM, GRU, CNN |
| Persistence baseline | `src/models/ml_models.py` | Lag-1 |
| Hybrid ML–DL | `src/hybrid/stacking.py` | Time-ordered OOF stacking |
| Agentic AI decision support | `src/agentic/` | Rule-based, human-review mandatory |
| Evaluation metrics | `src/evaluation/metrics.py` | RMSE, MAE, R², NRMSE |
| Results tables / figures | `src/pipeline.py` + `src/visualization/` | Generated from synthetic data by default |
| Extended-data nine series | `src/data/schema.py` → `EXTENDED_SERIES_TARGETS` | Exact names from paper Table |

## Known Gaps / Items Requiring Real Data

- Exact numeric results reported in the paper (RMSE 1.5211, R² 0.9524, 13.01 % improvement, etc.) **cannot** be reproduced without the original measurement files.
- HazardNet’s 3 313-column files and BodenType multi-sheet workbooks need specialised loaders once the files are available.
- The paper’s “expanded-feature global HGB” configuration can be re-created by setting the appropriate lag / rolling / covariate lists in `configs/config.yaml` once the real feature matrix is loaded.

## Inconsistencies Explicitly Avoided

- No classification metrics are computed.
- No claim is made that recommendations produce measured energy or carbon savings.
- Synthetic results are clearly labelled as such in every table and log message.
