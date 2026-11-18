# CSP-CAD — Safety-Aware Predictive Maintenance

Corrected research archive for Conformal Selective Prediction with Cost-Aware Deferral (CSP-CAD).

## Methodology

Three models: Random Forest, XGBoost, RUSBoost. Repeated stratified 5×10 CV, with 50% training, 10% probability calibration, 10% conformal calibration, 10% policy tuning and 20% independent test in each outer fold.

Cost scenarios `(C_FN,C_FP,C_D)`: `(10,1,1)`, `(20,1,2)`, `(30,1,5)`.

## Layout

- `repeated_cv/`: corrected runner, model builders and helpers
- `configs/`: nine archived YAML configurations
- `Data_file/`: UCI AI4I2020 data (UCI DOI: https://doi.org/10.24432/C5HS5C)
- `results/`: corrected evaluation CSVs
- `docs/PROVENANCE.md`: provenance limitations and interpretation cautions
- `SHA256SUMS.txt`: retrospective checksums of published files

## Important caveats

The archived configurations are not immutable execution-time evidence. Full-run code/configuration hash provenance has not been verified. Candidate risk/cost coverage curves come from policy-tuning data, not held-out testing. Repeated test folds overlap across repeats. The code has not been rerun as part of this publication preparation.

