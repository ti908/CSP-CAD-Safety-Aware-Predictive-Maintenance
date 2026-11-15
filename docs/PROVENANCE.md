# Provenance and limitations

Corrected experiment identifier: `20261114T203635883859Z_53561323`.

The present files have retrospective SHA-256 hashes recorded in `SHA256SUMS.txt`. This does not prove that these exact files were used in the original execution; no immutable run-specific full 50-fold manifest was verified.

The 45,000 tuning-curve rows are policy-tuning diagnostics, not independent-test curves.

The archived RUSBoost YAML `model.estimator.max_depth` differs from the builder lookup `model.base_max_depth`; the XGBoost `reg_lambda` YAML setting is not explicitly passed in the inspected builder. Scaling flags in YAML do not establish that the corrected runner scaled features.

The dataset DOI https://doi.org/10.24432/C5HS5C is not a code archival DOI.
