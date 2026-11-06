# Dataset Documentation

Schemas are defined in `src/data/schema.py` and serialised to `data/schemas/all_schemas.json`.

## Original families (paper Section Materials and Methods)
1. NLR HPC Facility / PUE – 4 569 543 rows, 11 cols; targets IT_power, PUE
2. Danish server-room – 1 371 363 rows, 52 cols; targets Ch1/2/3_power
3. NREL RSF 2011 – 8 760 rows, 12 cols; target Total_Data_Center_kW
4. Thai power meters – 393 106 rows, 58 cols; targets ADB1_CRAC3/4, UDB1_ULC_5, UDB2_ULC_6

## Additional groups
5. BodenType (multi-sheet xlsx)
6. Frontier HPC (Frontier2023 sheet, 49 874 rows)
7. AAU Field Validation (16 CSV files, 720 rows each)
8. PSNC (288 rows)
9. HazardNet Marconi A2 (3 files × 50 573 rows × 3 313 cols)

## Availability in this execution
**None of the real files were present.** Synthetic generators produce series that
respect the extended-experiment target names and approximate temporal dynamics.
