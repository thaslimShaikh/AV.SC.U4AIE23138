# Limitations

1. **No real measurement files** were available during execution. All reported metrics are synthetic.
2. **scikit-learn / XGBoost / LightGBM** could not be installed (network proxy failures). Pure-NumPy approximations were substituted; they are not identical to the paper’s HistGradientBoosting / XGBoost configurations.
3. **Deep models** were trained for only ~15 epochs for runtime feasibility; they are under-trained on the synthetic data.
4. **Thresholds** in the Agentic AI rule base are configurable illustrative values, not calibrated facility set-points.
5. **No energy, cost or carbon savings** are claimed. Forecasting accuracy alone does not establish operational savings.
6. **Heterogeneous units** across real series mean raw RMSE values must never be compared directly; NRMSE / per-series reporting is required.
7. Results from this smoke test **must not** be copied into the manuscript as experimental findings.
