# OUTPUTS_INDEX: Results and Discussion Artifacts for Academic LaTeX Manuscript

This document indexes all generated result files, tables, figures, configurations, and logs prepared for authoring the **Results and Discussion** section of the academic LaTeX manuscript on *Agentic AI Decision Support for Sustainable Data Centre Operations*.

---

## 1. Executive Summary & Critical Data Provenance Disclaimers

> **CRITICAL SCIENTIFIC INTEGRITY NOTICE (PRESERVATION OF ORIGINAL DATA):**
> - **Real-World Measurement Data**: As documented in the pipeline logs (`logs/pipeline.log`) and project documentation (`docs/EXPERIMENTAL_RESULTS.md`, `docs/LIMITATIONS.md`), original production telemetry datasets (e.g., NLR HPC Facility, Danish server-room, NREL RSF 2011, Thai power meters, HazardNet, BodenType, Frontier, AAU, PSNC) were **NOT PRESENT** in the execution environment.
> - **Synthetic Smoke-Test Data**: All numeric metrics in these tables and figures were computed using **synthetic time-series** generated strictly according to the paper schemas, target definitions, and chronological structures (seed = 42).
> - **Non-Fabrication**: Values are preserved exactly as produced by the pipeline. No metrics have been altered, smoothed, or fabricated.
> - **Operational Claims Disclaimer**: Simulated agentic recommendations and decision-support outputs are heuristic rule evaluations on smoke-test data. They **must not** be described as measured real-world operational savings (e.g. energy, PUE reduction, cost, or carbon abatement) without physical facility deployment.

---

## 2. Coverage of Specific Manuscript Requirements

| Item | Requested Topic | Status in Artifacts | Relevant Files | Key Notes |
|:---|:---|:---|:---|:---|
| **1** | **Dataset metadata & sample summaries** | **AVAILABLE (Synthetic stubs + Schemas)** | `tables/table1_dataset_inventory.csv`, `tables/dataset_inventory.csv`, `docs/all_schemas.json`, `docs/DATASET_DOCUMENTATION.md` | Documents all 9 dataset families (4 original, 5 additional), expected dimensions (up to 4.5M rows, 3,313 cols), target columns, and availability status. Real data is not present; synthetic stubs match target schema. |
| **2** | **Preprocessing & chronological split statistics** | **AVAILABLE (Exact Splits)** | `tables/table2_preprocessing_summary.csv`, `tables/table3_split_distribution.csv` | Chronological split ratios strictly applied: 60.0% Train ($n=1,496$), 20.0% Validation ($n=498$), 20.0% Test ($n=500$) for each of the 9 aligned extended series ($N=2,500$). Leakage-free lag ($1, 2, 3, 6, 12$) and rolling window features. |
| **3** | **Complete model comparison results** | **AVAILABLE (Complete Metrics)** | `tables/table6_hybrid_comparison.csv`, `tables/table7_overall_comparison.csv`, `tables/model_comparison.csv`, `tables/all_metrics_long.csv`, `latex_tables/table7_overall.tex` | Full metric suite across all models: RMSE, MAE, R², and NRMSE, plus training and inference latencies. |
| **4** | **Per-dataset and per-target forecasting results** | **AVAILABLE (Per-Target Breakdown)** | `tables/table4_ml_performance.csv`, `tables/table5_dl_performance.csv`, `tables/table6_hybrid_comparison.csv`, `tables/table8_final_test_performance.csv`, `tables/final_test_results.csv`, `latex_tables/table8_final_test.tex`, `metrics/raw_results.json` | Detailed metrics broken down across each of the 9 extended targets: `Cooling_Return_Average`, `Cooling_Supply_Average`, `Inlet_Temp_Avg`, `Outlet_Avg_Temp`, `Overall_Coolant_Supply_Temp`, `Overall_average_Coolant_Return_Temp`, `SubLoop1/2/3_Coolant_Return_Temp`. |
| **5** | **ML, DL, and Hybrid model comparisons (Fusion / Meta-learner)** | **AVAILABLE (Comparative Analysis)** | `tables/table6_hybrid_comparison.csv`, `tables/table7_overall_comparison.csv`, `figures/png/11_ml_dl_hybrid_comparison.png`, `figures/pdf/11_ml_dl_hybrid_comparison.pdf` | Compares pure ML regressors (HistGB, Ridge, GradientBoosting, RF, ET), DL architectures (MLP, ResMLP, LSTM, GRU, 1D-CNN), and time-ordered out-of-fold Stacking Hybrid ensemble with a Ridge meta-learner. Hybrid stack ranks #1 among learned models (Macro NRMSE 0.0886, R² 0.7858). |
| **6** | **Persistence baseline comparisons** | **AVAILABLE (Benchmark Baseline)** | `tables/table6_hybrid_comparison.csv`, `tables/table7_overall_comparison.csv`, `tables/table8_final_test_performance.csv` | Lag-1 persistence baseline ($\hat{y}_t = y_{t-1}$) included on every target. On high-autocorrelation synthetic series, persistence achieved Macro NRMSE 0.0798, R² 0.8257. |
| **7** | **Agentic AI evaluation results** | **AVAILABLE (Full Scenario Reports)** | `tables/table10_agentic_evaluation.csv`, `metrics/agentic_reports.txt`, `figures/png/20_agentic_recommendation_summary.png`, `figures/pdf/20_agentic_recommendation_summary.pdf` | Evaluates 6 operational scenarios: High PUE, High Inlet Temp, Cooling Supply Drift, IT Power Spike, Low Delta-T, and High Model Uncertainty. Includes rule IDs, priority levels, confidence ratings, human review flags (`human_review_required=True` across all), and explicit evidence dictionaries. |
| **8** | **Sustainability metrics (Energy, PUE, Cooling, Carbon)** | **PARTIALLY AVAILABLE (Indicators only; Real savings NOT AVAILABLE)** | `tables/table9_sustainability_indicators.csv`, `docs/LIMITATIONS.md` | Specific thermal and cooling indicators are tracked (Supply/Return coolant and air temperatures). PUE (1.65) and IT power (950 kW) are evaluated only as illustrative rule triggers. Measured real-world kilowatt-hour energy savings, actual PUE reductions, and carbon footprint emissions are **NOT CALCULATED / NOT AVAILABLE**. |
| **9** | **Ablation study results** | **NOT AVAILABLE (as separate quantitative run)** | Documented in `docs/latex_methodology_update.tex`, `docs/METHODOLOGY.md` | The methodology describes an ablation protocol isolating feature engineering, augmentation, and hybrid stacking. However, a separate multi-stage quantitative ablation table was **NOT GENERATED / NOT AVAILABLE** in the execution artifacts beyond the model-class comparisons in Table 6. |
| **10** | **Statistical significance tests & confidence intervals** | **NOT AVAILABLE / NOT CONDUCTED** | Documented in `OUTPUTS_INDEX.md` | Formal inferential tests (e.g. Diebold-Mariano test, Wilcoxon signed-rank test, bootstrap confidence intervals) were **NOT CONDUCTED / NOT AVAILABLE** in this pipeline run. |

---

## 3. Directory Structure and Complete File Inventory

```
latex_required_outputs/
├── OUTPUTS_INDEX.md                           # This index file
├── tables/                                    # CSV tables of results and splits
│   ├── table1_dataset_inventory.csv           # Table 1: Inventory of 9 dataset families, schema & provenance
│   ├── dataset_inventory.csv                  # Duplicate/root copy of Table 1 from results/
│   ├── table2_preprocessing_summary.csv       # Table 2: Feature sets, row counts, filtering stats
│   ├── table3_split_distribution.csv          # Table 3: Chronological 60/20/20 train/val/test sample splits
│   ├── table4_ml_performance.csv              # Table 4: Machine learning models performance per target
│   ├── table5_dl_performance.csv              # Table 5: Deep learning models performance per target
│   ├── table6_hybrid_comparison.csv           # Table 6: Per-target comparison of baseline, ML, DL, and Hybrid
│   ├── table7_overall_comparison.csv          # Table 7: Macro-average performance summary and model rankings
│   ├── model_comparison.csv                   # Duplicate/root copy of Table 7 from results/
│   ├── table8_final_test_performance.csv      # Table 8: Best model selection and final test partition performance
│   ├── final_test_results.csv                 # Duplicate/root copy of Table 8 from results/
│   ├── table9_sustainability_indicators.csv   # Table 9: Tracked thermal & sustainability indicators
│   ├── table10_agentic_evaluation.csv         # Table 10: 6 Agentic AI scenario evaluations & recommendations
│   └── all_metrics_long.csv                   # Full long-format metrics dataset across all runs
├── metrics/                                   # Detailed execution metrics and agentic reports
│   ├── raw_results.json                       # Complete JSON serialization of all predictions, features & metrics
│   └── agentic_reports.txt                    # Formatted text reports of the 6 agentic decision-support scenarios
├── figures/                                   # High-resolution figures for LaTeX manuscript
│   ├── png/                                   # Raster format (300 DPI) for preview / web rendering
│   │   ├── 01_architecture.png                # System end-to-end modular architecture diagram
│   │   ├── 04_cross_series_r2.png             # Cross-series R² comparison across models
│   │   ├── 04_target_distributions.png        # Target variable value distributions across series
│   │   ├── 05_actual_vs_predicted.png         # Actual vs. predicted time-series trajectories
│   │   ├── 07_rmse_comparison.png             # Root Mean Squared Error comparison across models
│   │   ├── 08_mae_comparison.png              # Mean Absolute Error comparison across models
│   │   ├── 09_r2_comparison.png               # Coefficient of determination (R²) comparison
│   │   ├── 10_nrmse_comparison.png            # Normalized RMSE comparison across models
│   │   ├── 11_ml_dl_hybrid_comparison.png     # Macro comparison: ML vs. DL vs. Hybrid ensemble
│   │   ├── 13_training_time.png               # Computational training latency comparison
│   │   ├── 15_residual_analysis.png           # Residual distributions and error analysis
│   │   └── 20_agentic_recommendation_summary.png # Summary of agentic rule activations and priorities
│   └── pdf/                                   # Vector PDF format for publication-quality LaTeX embedding
│       ├── 01_architecture.pdf
│       ├── 04_cross_series_r2.pdf
│       ├── 04_target_distributions.pdf
│       ├── 05_actual_vs_predicted.pdf
│       ├── 07_rmse_comparison.pdf
│       ├── 08_mae_comparison.pdf
│       ├── 09_r2_comparison.pdf
│       ├── 10_nrmse_comparison.pdf
│       ├── 11_ml_dl_hybrid_comparison.pdf
│       ├── 13_training_time.pdf
│       ├── 15_residual_analysis.pdf
│       └── 20_agentic_recommendation_summary.pdf
├── latex_tables/                              # Ready-to-input LaTeX table fragments
│   ├── table7_overall.tex                     # LaTeX tabular fragment for Table 7 (Overall comparison)
│   ├── table8_final_test.tex                  # LaTeX tabular fragment for Table 8 (Final test performance)
│   └── latex_methodology_update.tex           # LaTeX section snippet updating methodology & ablation protocol
├── configs/                                   # Experiment configuration & hyperparameters
│   └── config.yaml                            # Full pipeline configuration (lags, windows, models, splits)
├── logs/                                      # Pipeline execution traces
│   └── pipeline.log                           # Complete timestamped execution log of pipeline run
└── docs/                                      # Research documentation & dataset schemas
    ├── all_schemas.json                       # Formal JSON schemas for all 9 data centre datasets
    ├── DATASET_DOCUMENTATION.md               # Detailed documentation of dataset schemas and target variables
    ├── EXPERIMENTAL_RESULTS.md                # Summary notes on synthetic experimental results
    ├── LIMITATIONS.md                         # Detailed list of execution constraints and limitations
    ├── METHODOLOGY.md                         # Detailed experimental methodology, split discipline & models
    └── PAPER_ALIGNMENT.md                     # Alignment matrix mapping paper sections to codebase artifacts
```

---

## 4. Detailed File-by-File Description & Data Provenance

### Tables (`tables/`)

| File Name | Original Source | Content Description | Data Nature |
|:---|:---|:---|:---|
| `table1_dataset_inventory.csv` | `outputs/tables/table1_dataset_inventory.csv` | Comprehensive inventory of 9 data centre dataset families (NLR HPC, Danish server room, NREL RSF, Thai power meters, BodenType, Frontier HPC, AAU, PSNC, HazardNet). Contains expected rows, cols, targets, and availability status. | Metadata / Schemas of Real datasets; Pipeline status: Synthetic stubs. |
| `dataset_inventory.csv` | `results/dataset_inventory.csv` | Top-level summary duplicate of `table1_dataset_inventory.csv`. | Metadata / Schemas. |
| `table2_preprocessing_summary.csv` | `outputs/tables/table2_preprocessing_summary.csv` | Preprocessing audit across 9 targets, including retained sample count ($n=2,500$), verification of zero missing values and zero duplicate timestamps, and feature lists. | Synthetic smoke-test series. |
| `table3_split_distribution.csv` | `outputs/tables/table3_split_distribution.csv` | Exact sample counts and percentages for chronological splitting: 1,496 train (60%), 498 val (20%), 500 test (20%) across each target. | Synthetic smoke-test series. |
| `table4_ml_performance.csv` | `outputs/tables/table4_ml_performance.csv` | Validation performance (RMSE, MAE, R², NRMSE, train/inference time) for individual Machine Learning regressors: HistGradientBoosting, RandomForest, ExtraTrees, Ridge, and GradientBoosting across all 9 series. | Synthetic smoke-test series. |
| `table5_dl_performance.csv` | `outputs/tables/table5_dl_performance.csv` | Validation performance for Deep Learning architectures: MLP, Residual MLP, LSTM, GRU, and 1D-CNN across all 9 series. (Under-trained under 15-epoch budget). | Synthetic smoke-test series. |
| `table6_hybrid_comparison.csv` | `outputs/tables/table6_hybrid_comparison.csv` | Unified comparative results on all 9 series across all 12 evaluated models (Persistence baseline, 5 ML models, 5 DL models, and Stacking Hybrid). | Synthetic smoke-test series. |
| `table7_overall_comparison.csv` | `outputs/tables/table7_overall_comparison.csv` | Overall macro-averaged metrics (macro NRMSE, RMSE, MAE, R², latency) across all series with model rankings. Persistence ranks #1, Hybrid Stacking ranks #2 (best learned model), Ridge ranks #3. | Synthetic smoke-test series. |
| `model_comparison.csv` | `results/model_comparison.csv` | Top-level summary duplicate of `table7_overall_comparison.csv`. | Synthetic smoke-test series. |
| `table8_final_test_performance.csv` | `outputs/tables/table8_final_test_performance.csv` | Out-of-sample test partition performance ($n=500$) for the optimal model selected per series during validation. | Synthetic smoke-test series. |
| `final_test_results.csv` | `results/final_test_results.csv` | Top-level summary duplicate of `table8_final_test_performance.csv`. | Synthetic smoke-test series. |
| `table9_sustainability_indicators.csv` | `outputs/tables/table9_sustainability_indicators.csv` | Audit of tracked sustainability indicators (Cooling Return/Supply, Inlet/Outlet temperatures, Coolant temps, PUE, IT power). | Documentation of tracked indicators. |
| `table10_agentic_evaluation.csv` | `outputs/tables/table10_agentic_evaluation.csv` | Evaluation table for 6 operational decision scenarios detailing scenario name, rule ID, priority, confidence score, human review flag, evidence, and message preview. | Synthetic evaluation scenarios. |
| `all_metrics_long.csv` | `outputs/tables/all_metrics_long.csv` | Full unaggregated metric records across all series, models, categories, and metrics. | Synthetic smoke-test series. |

### Metrics & Diagnostic Logs (`metrics/` & `logs/`)

| File Name | Original Source | Content Description | Data Nature |
|:---|:---|:---|:---|
| `raw_results.json` | `outputs/metrics/raw_results.json` | Complete machine-readable JSON structure (1.6 MB) containing all individual predictions, ground truth series, model hyperparameters, feature arrays, and validation scores for all 9 series. | Synthetic smoke-test execution. |
| `agentic_reports.txt` | `outputs/metrics/agentic_reports.txt` | Formatted ASCII operational reports for the 6 Agentic AI scenarios showing context, forecast inputs, sensor observations, model uncertainty, prioritised recommendations, evidence, and safety human-in-the-loop disclaimers. | Synthetic evaluation scenarios. |
| `pipeline.log` | `outputs/logs/pipeline.log` | Full pipeline execution trace recording random seed initialization (42), data generation, model training durations, per-target validation metrics, and agentic rule evaluations. | Execution log trace. |

### Figures (`figures/png/` & `figures/pdf/`)

Both 300 DPI raster PNG (for immediate inspection) and publication-quality vector PDF formats (for LaTeX manuscript embedding via `\includegraphics`) are provided for all 12 generated figures:

| File Name | Description | Key Manuscript Takeaway |
|:---|:---|:---|
| `01_architecture.*` | High-level system architecture illustrating data ingestion, leakage-free feature engineering, model training, hybrid fusion, and agentic decision support. | Conceptual architectural diagram. |
| `04_cross_series_r2.*` | Heatmap/bar chart showing cross-series $R^2$ performance across models. | Shows high predictive power on return coolant temps ($R^2 > 0.90$) and moderate on supply temps. |
| `04_target_distributions.*` | Empirical distribution (violin/histogram) for each of the 9 target temperature variables. | Illustrates operational distributions and variance across sensor locations. |
| `05_actual_vs_predicted.*` | Time-series traces comparing ground truth actual measurements against predicted values from top models. | Demonstrates tracking capability and lag response of persistence vs. hybrid models. |
| `07_rmse_comparison.*` | Grouped bar chart comparing Root Mean Squared Error across models and targets. | Visualises absolute error differences across series. |
| `08_mae_comparison.*` | Grouped bar chart comparing Mean Absolute Error across models and targets. | Highlights robust L1 error comparisons. |
| `09_r2_comparison.*` | Comparative plot of $R^2$ scores across all models. | Highlights positive explanatory power for ML/Hybrid vs. negative scores for under-trained DL. |
| `10_nrmse_comparison.*` | Normalised RMSE comparison across all series, enabling fair cross-target comparison despite differing temperature scales. | Primary metric for ranking models across heterogeneous targets. |
| `11_ml_dl_hybrid_comparison.*` | Macro performance comparison grouping models by class: Baseline vs. ML vs. DL vs. Hybrid Stacking. | Demonstrates that the Hybrid Stack outperforms individual learned ML regressors. |
| `13_training_time.*` | Wall-clock execution time for training and inference across all model families. | Ridge and Tree ensembles train in $<0.1$ s, while neural models require orders of magnitude more compute. |
| `15_residual_analysis.*` | Residual error distributions, QQ plots, and autocorrelation of prediction residuals. | Evaluates whether model residuals approximate zero-mean Gaussian noise. |
| `20_agentic_recommendation_summary.*` | Overview of activated agentic recommendations across the 6 test scenarios, categorized by priority and human-in-the-loop requirement. | Summarises rule coverage, action priority distribution, and safety governance. |

### LaTeX Tables and Fragments (`latex_tables/`)

| File Name | Original Source | Description |
|:---|:---|:---|
| `table7_overall.tex` | `paper/tables/table7_overall.tex` | Pre-formatted LaTeX `tabular` fragment of Table 7 (Overall macro-averaged model comparison and rankings). Ready to include in the manuscript with `\input{tables/table7_overall.tex}`. |
| `table8_final_test.tex` | `paper/tables/table8_final_test.tex` | Pre-formatted LaTeX `tabular` fragment of Table 8 (Final out-of-sample test performance for selected best models). Ready to include via `\input{tables/table8_final_test.tex}`. |
| `latex_methodology_update.tex` | `docs/latex_methodology_update.tex` | LaTeX section fragment describing the empirical ablation and comparison protocol, model classes, and ethical safety boundaries. |

### Configurations & Documentation (`configs/` & `docs/`)

| File Name | Original Source | Description |
|:---|:---|:---|
| `config.yaml` | `configs/config.yaml` | Experiment parameters: 60/20/20 splits, lag orders $[1, 2, 3, 6, 12]$, rolling windows $[3, 6, 12, 24]$, model configurations, agentic uncertainty threshold ($0.15$), and random seed ($42$). |
| `all_schemas.json` | `data/schemas/all_schemas.json` | Exact JSON schema definitions (columns, data types, physical units, sampling rates) for all 9 data centre datasets described in the paper. |
| `DATASET_DOCUMENTATION.md` | `docs/DATASET_DOCUMENTATION.md` | Detailed documentation of original and additional dataset families and their physical parameters. |
| `EXPERIMENTAL_RESULTS.md` | `docs/EXPERIMENTAL_RESULTS.md` | Experimental results summary, macro rankings, and execution environment specifications. |
| `LIMITATIONS.md` | `docs/LIMITATIONS.md` | Critical transparency document explaining limitations (synthetic data, package fallback, short DL epochs, non-operational savings). |
| `METHODOLOGY.md` | `docs/METHODOLOGY.md` | Comprehensive methodology specification (problem framing, chronological splitting, feature engineering, stacking setup). |
| `PAPER_ALIGNMENT.md` | `docs/PAPER_ALIGNMENT.md` | Mapping between the paper's manuscript sections and codebase implementations. |

---

## 5. Guidance for Authoring the Results and Discussion LaTeX Section

When drafting the LaTeX manuscript, authors should structure the narrative using the following data points and caveats:

1. **Section Structure Suggestion**:
   - **Data Characteristics and Partitioning**: Cite `table1_dataset_inventory.csv` and `table3_split_distribution.csv` to explain dataset dimensionalities and the strict 60/20/20 chronological split discipline.
   - **Predictive Performance & Model Comparison**: Present `table7_overall.tex` (Macro comparison) and Figure 11 (`11_ml_dl_hybrid_comparison.pdf`). Highlight that among learned models, the **Hybrid Stacking Ensemble** achieved the best predictive accuracy (Macro NRMSE: 0.0886, Macro $R^2$: 0.7858), surpassing individual ML models (Ridge: 0.0997, HistGB: 0.1030).
   - **Per-Target Evaluation & Baseline Benchmark**: Present `table8_final_test.tex` and Figure 10 (`10_nrmse_comparison.pdf`). Note that the lag-1 persistence baseline (Macro NRMSE 0.0798) is highly competitive due to strong one-step temporal autocorrelation, establishing a rigorous benchmark.
   - **Computational Latency & Efficiency**: Reference `table7_overall_comparison.csv` and Figure 13 (`13_training_time.pdf`), demonstrating the ultra-low training latency of linear and tree models compared to deep architectures.
   - **Agentic Decision Support & Safety Governance**: Cite `table10_agentic_evaluation.csv`, `agentic_reports.txt`, and Figure 20 (`20_agentic_recommendation_summary.pdf`). Discuss the 6 simulated operational scenarios, highlighting that each recommendation pairs predictive evidence with explicit human review requirements (`human_review_required: True`), preventing uncontrolled autonomous actions.
   - **Limitations & Future Work**: Explicitly articulate the synthetic nature of the current pipeline execution, referencing `LIMITATIONS.md`, noting that real telemetry is required to substantiate physical energy and PUE reductions.
