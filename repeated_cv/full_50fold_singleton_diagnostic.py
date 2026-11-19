"""Reproduce and audit 50 independent test folds for each model.

Place in repeated_cv/ next to singleton_agreement_diagnostic.py.
No historical outputs are modified. This diagnostic may take substantial time.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from repeated_cv.singleton_agreement_diagnostic import diagnose_one_fold


def run_full_singleton_diagnostic(experiments, baseline_csv, *,
                                  output_dir='Step4_Validation_Results/singleton_50fold',
                                  cost_scenario='20_1_2', n_folds=50):
    """experiments: iterable of (baseline model name, config path, model builder).

    Runs strict baseline reproduction on every fold. Writes fresh files in a
    newly created directory; fails if the directory already exists.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=False)
    baseline_csv = Path(baseline_csv)
    baseline = pd.read_csv(baseline_csv)
    all_results = []
    for model_name, cfg_path, builder in experiments:
        expected = baseline.loc[(baseline.model == model_name) &
                                (baseline.cost_scenario == cost_scenario)]
        if expected.fold_number.nunique() < n_folds:
            raise ValueError(f'{model_name}: insufficient baseline folds')
        for fold in range(1, n_folds + 1):
            summary, rows = diagnose_one_fold(
                model_name, cfg_path, builder, baseline_csv,
                fold_number=fold, cost_scenario=cost_scenario)
            # Exact test-only counterfactual: force retained predictions to
            # the conformal singleton label; do NOT tune on test labels.
            retained = rows['retained'].to_numpy(dtype=bool)
            y = rows['y_true'].to_numpy(dtype=int)
            pred = rows['cost_sensitive_prediction'].to_numpy(dtype=int)
            singleton = rows['singleton_label'].to_numpy(dtype=int)
            alt_pred = np.where(retained, singleton, pred)
            fn_base = int(np.sum(retained & (pred == 0) & (y == 1)))
            fp_base = int(np.sum(retained & (pred == 1) & (y == 0)))
            fn_alt = int(np.sum(retained & (alt_pred == 0) & (y == 1)))
            fp_alt = int(np.sum(retained & (alt_pred == 1) & (y == 0)))
            n_deferred = int((~retained).sum())
            cost = expected.loc[expected.fold_number == fold].iloc[0]
            c_fn, c_fp, c_def = (int(cost[k]) for k in ('C_FN', 'C_FP', 'C_DEF'))
            n = len(rows)
            base_cost = (c_fn * fn_base + c_fp * fp_base + c_def * n_deferred) / n
            alt_cost = (c_fn * fn_alt + c_fp * fp_alt + c_def * n_deferred) / n
            if not np.isclose(base_cost, float(cost.best_cost_per_sample), atol=1e-9):
                raise AssertionError(f'{model_name} fold {fold}: cost mismatch')
            summary.update({
                'n_deferred': n_deferred,
                'baseline_fn': fn_base, 'baseline_fp': fp_base,
                'singleton_policy_fn': fn_alt, 'singleton_policy_fp': fp_alt,
                'baseline_cost_per_sample': base_cost,
                'singleton_policy_cost_per_sample': alt_cost,
                'singleton_minus_baseline_cost': alt_cost - base_cost,
                'disagreement_true_failures': int(np.sum(rows.singleton_disagreement & (rows.y_true == 1))),
                'disagreement_true_nonfailures': int(np.sum(rows.singleton_disagreement & (rows.y_true == 0))),
            })
            all_results.append(summary)
            # Incremental checkpointing; a partial file is deliberately named PARTIAL.
            pd.DataFrame(all_results).to_csv(out / 'singleton_50fold_PARTIAL.csv', index=False)
            print(f'{model_name}: fold {fold}/{n_folds} reproduced; disagreements={summary["n_disagreements"]}', flush=True)
    results = pd.DataFrame(all_results)
    results.to_csv(out / 'singleton_50fold_fold_results.csv', index=False)
    grouped = results.groupby('model').agg(
        folds=('fold_number', 'count'),
        mean_disagreement_rate=('disagreement_fraction_retained', 'mean'),
        std_disagreement_rate=('disagreement_fraction_retained', 'std'),
        total_disagreements=('n_disagreements', 'sum'),
        disagreement_false_positives=('disagreement_false_positives', 'sum'),
        disagreement_true_failures=('disagreement_true_failures', 'sum'),
        mean_baseline_cost=('baseline_cost_per_sample', 'mean'),
        mean_singleton_policy_cost=('singleton_policy_cost_per_sample', 'mean'),
        mean_cost_difference=('singleton_minus_baseline_cost', 'mean'),
    ).reset_index()
    grouped.to_csv(out / 'singleton_50fold_model_summary.csv', index=False)
    (out / 'manifest.json').write_text(json.dumps({
        'scenario': cost_scenario, 'folds_per_model': n_folds,
        'baseline_csv': str(baseline_csv),
        'counterfactual': 'fixed singleton label on retained rows; no test-set tuning',
        'note': 'Overlapping repeated-CV test folds are not independent replicates'
    }, indent=2))
    return results, grouped
