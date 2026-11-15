# Experiment provenance and limitations

## Confirmed corrected evaluation identifier
`20261114T203635883859Z_53561323`

Previously inspected corrected outputs include:
- `all_models_50fold_summary.csv`: 450 primary independent-test summary records (3 models × 50 folds × 3 costs).
- `all_models_50fold_tuning_curves.csv`: 45,000 policy-tuning curve records.

These filenames and counts refer to the corrected local research archive. They are **not assertions that those files have already been published here**.

## What is verified
- The new repository is publicly visible under `ti908`.
- The corrected archive includes model-building, CV runner, helper, and YAML files.
- Existing SHA-256 registers are retrospective fingerprints of archived files.

## What remains unverified
- An immutable, execution-time manifest for the complete 50-fold run linking the exact dataset, source revision, configuration revision, package versions, and output hashes.
- An archival DOI for the research code.
- Whether every file in this repository is byte-identical to the file used in the original execution until individual hashes are compared.

## Model/configuration discrepancies
- XGBoost: the archived YAML `reg_lambda` entry was not explicitly passed by the inspected builder.
- RUSBoost: YAML `model.estimator.max_depth` differs from the builder's `model.base_max_depth` lookup; the builder's default may therefore apply.
- The corrected repeated-CV runner does not perform the scaling implied by some archived YAML flags.

## Research integrity
Do not infer execution provenance from present-day hashes, Git commits created after the run, or manifests from separate one-fold sanity tests. Clearly label policy-tuning diagnostics and independent-test measurements.
