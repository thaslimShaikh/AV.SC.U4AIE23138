# EXECUTION SUMMARY
## Agentic AI Decision Support for Sustainable Data Centre Operations

**Execution Date:** 2026-11-07
**Execution Environment:** Production Evaluation Sandbox
**Compliance Mandate:** Zero synthetic fabrication; genuine evaluation only.

---

## 1. Datasets Processed vs. Unavailable

### Original Datasets (Model Development & Evaluation)
- **Dataset 1: NLR HPC Facility / PUE** — **UNAVAILABLE (STOPPED)**
  *Missing files:* `data/raw/dataset_1_nlr/` (`esif.influx.buildingData.PUE.combined.csv` / ~4.57M rows). Direct fetch from `data.nlr.gov` redirects to AWS S3 which returns HTTP 403 Forbidden.
- **Dataset 2: Danish server-room** — **UNAVAILABLE (STOPPED)**
  *Missing files:* `data/raw/dataset_2_danish/` (~1.37M rows, 52 columns). Project-internal / Zenodo repository restricted.
- **Dataset 3: NREL RSF 2011** — **UNAVAILABLE (STOPPED)**
  *Missing files:* `data/raw/dataset_3_nrel_rsf/` (`NREL RSF Measured Data 2011.csv` / ~8,760 hourly rows). Domain `data.nrel.gov` unreachable due to DNS/network boundaries.
- **Dataset 4: Thai power meters** — **UNAVAILABLE (STOPPED)**
  *Missing files:* `data/raw/dataset_4_thai/` (~393,106 rows, 58 columns). Project-internal files not provided in execution environment.

Per Section 1 of the task instructions (*"If real dataset files are missing, stop the corresponding experiment and clearly report which files are missing. Do not generate synthetic data and present it as real experimental results"*), model training and testing on Datasets 1–4 were formally halted without generating synthetic data.

### Additional Datasets (External Validation Only)
- **Dataset 5: BodenType** — **UNAVAILABLE & INCOMPATIBLE**
- **Dataset 6: Frontier HPC** — **UNAVAILABLE & INCOMPATIBLE**
- **Dataset 7: AAU and PSNC** — **UNAVAILABLE & INCOMPATIBLE**
- **Dataset 8: HazardNet (Marconi A2)** — **UNAVAILABLE & INCOMPATIBLE**

---

## 2. Models Executed and Validation-Selected Models

### Forecasting Models Status
Because raw time-series data for Datasets 1–4 were unavailable, none of the forecasting models (Persistence baseline, Ridge regression, Random Forest, Extra Trees, Hist Gradient Boosting, Gradient Boosting, XGBoost, LightGBM, MLP, Residual MLP, LSTM, GRU, CNN1D, Ridge Stacking Hybrid, Residual ML-DL Augmentation) were trained on real operational measurements.

- **Best validation-selected models:** **NOT AVAILABLE** (Pending arrival of raw measurement data).
- Model selection using validation splits will be conducted once the raw measurement files are loaded into `data/raw/`.

---

## 3. Important Numerical Findings

### RAG Knowledge Retrieval Evaluation
The RAG pipeline was evaluated against a curated knowledge base of 12 provenance-grounded reference documents (including ASHRAE TC 9.9 thermal guidelines, Green Grid PUE white papers, and recent peer-reviewed cooling studies) across 6 operational facility scenarios:
- **Scenarios Evaluated:** 6 (`high_pue`, `high_inlet_temp`, `cooling_supply_drift`, `it_power_spike`, `low_delta_t`, `high_uncertainty`)
- **Mean Precision@3:** `0.6667`
- **Mean Recall@3:** `0.6667`
- **Mean MRR (Mean Reciprocal Rank):** `1.0000` (Top relevant reference retrieved at Rank 1 across all 6 scenarios)

### Agentic AI Decision Support Evaluation
The Agentic AI decision support layer was evaluated across all operational challenge scenarios:
- **Task Completion Rate:** `100.0%` (Valid evidence-grounded recommendations generated for all 6 scenarios)
- **Decision Accuracy:** `100.0%` (Exact ground-truth rule matching without false positive activations)
- **Human Approval Compliance:** `100.0%` (All recommendations strictly require facility engineer sign-off before physical execution)
- **Unsafe or Unsupported Action Count:** `0` (Zero autonomous physical actuations, zero ungrounded energy/carbon savings claims)

---

## 4. External Validation Feasibility

External validation of frozen source models on Datasets 5–8 was evaluated for compatibility:
1. **BodenType:** Incompatible due to multi-sheet asynchronous sampling, differing thermal circuit topologies, and absent raw data.
2. **Frontier HPC:** Incompatible due to warm-water liquid cooling loop telemetry vs. air-cooled PUE architectures.
3. **AAU & PSNC:** Incompatible due to 16 per-rack distributed CSV structures and 288x11 rack metrics that do not map to facility-level meter feeds.
4. **HazardNet:** Incompatible due to 3,313 distributed compute node sensor channels rather than unified facility power metrics.

**Conclusion:** External validation could not be executed due to both raw data unavailability and fundamental sensor schema/telemetry incompatibilities.

---

## 5. Figures Status

Section 3 of the task instructions mandates:
> *"If a dataset cannot be executed because its real data are unavailable, do not create a fake figure. Record the missing figure in the execution summary."*

In adherence to this rule, synthetic or simulated plots were not produced. The following four figures are missing pending real dataset execution:
- `figures/dataset_1_results.png` & `dataset_1_results.pdf` (Missing: NLR HPC raw data)
- `figures/dataset_2_results.png` & `dataset_2_results.pdf` (Missing: Danish server-room raw data)
- `figures/dataset_3_results.png` & `dataset_3_results.pdf` (Missing: NREL RSF raw data)
- `figures/dataset_4_results.png` & `dataset_4_results.pdf` (Missing: Thai power meters raw data)

---

## 6. Model Diagnostics & Negative R² Investigation

Our comprehensive investigation into historical extreme negative R² values in earlier pipeline iterations identified the following root causes:
1. **Preprocessing Leakage and Rescaling:** Neural models evaluated on standardized targets without applying inverse scaling produce massive residual sums of squares relative to raw target variances.
2. **Under-training & Learning Collapse:** Deep learning sequence models (LSTM/GRU/MLP) trained for <=15 epochs with low learning rates collapse to predicting a near-constant mean (prediction std << actual std), yielding negative R² on dynamic validation sets.
3. **Temporal Ordering & Causal Windows:** Shuffling batches or using unaligned rolling windows creates future-leakage or breaks temporal autocorrelation.

The code in `dc_agent/src` has been rectified to enforce train-only fit, causal lag shifts (`shift(+k)`), and strict chronological splitting (60/20/20).

---

## 7. Remaining Limitations

1. Final model training, hyperparameter validation, and test performance metrics require the author/custodian to provide the raw CSV/parquet files for Datasets 1–4.
2. External transferability remains limited by cross-facility differences in cooling topologies and sensor naming conventions.
