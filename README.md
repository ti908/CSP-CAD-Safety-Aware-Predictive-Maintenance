# CSP-CAD: Safety-Aware Predictive Maintenance

**Conformal Selective Prediction with Cost-Aware Deferral (CSP-CAD)**

Research implementation for calibrated failure prediction, split conformal prediction sets, and cost-aware selective decisions using the UCI AI4I 2020 predictive maintenance dataset.

## Methodology
- Models: Random Forest, XGBoost, RUSBoost.
- Evaluation: repeated stratified cross-validation, 5 folds × 10 repeats.
- Within each outer fold: 50% training, 10% probability calibration, 10% conformal calibration, 10% policy tuning, 20% held-out test.
- Experimental cost scenarios (false negative, false positive, deferral): (10,1,1), (20,1,2), (30,1,5).
- Tuning data select classification and deferral thresholds; held-out test labels do not select policies.

## Publication status
This public repository is **under construction**. Do not treat it as a complete independently reproducible snapshot until the corrected source code, configurations, outputs, and checksums are published and inspected.

The original full 50-fold experiment does not have a verified immutable execution-time manifest linking its exact code, configuration, environment and output hashes.

## Dataset
UCI AI4I 2020 Predictive Maintenance Dataset, DOI https://doi.org/10.24432/C5HS5C.

## Important interpretation limits
- Cost-versus-coverage and risk-versus-coverage candidate curves are policy-tuning diagnostics, not held-out-test curves.
- Marginal conformal coverage is not failure-class conditional coverage.
- Repeated test folds overlap across repeats and are not independent test observations.
- Some archived YAML model settings differ from the corrected builder implementation.

See [Provenance and limitations](docs/PROVENANCE.md).

## Citation
Cite the associated MSc manuscript by its full title and author. A code archival DOI has not been verified. The dataset DOI is not a DOI for this code.
