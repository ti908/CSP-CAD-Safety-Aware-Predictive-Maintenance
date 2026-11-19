"""Read-only one-fold reproduction and singleton-label diagnostic.

Run from the project root in the corrected notebook. Requires the existing
runner, its configuration files, and the original AI4I data. Does not modify
baseline outputs or runner code.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import LabelEncoder

from repeated_cv.cv_helpers import (load_config, compute_conformal_quantile,
    build_conformal_sets, evaluate_deferral_policy)
from repeated_cv.repeated_cv_runner import (_split_development, _fit_calibrator,
    _choose_threshold, _tune_tau)


def diagnose_one_fold(model_name, cfg_path, build_model_fn, baseline_csv,
                      *, fold_number=1, cost_scenario='20_1_2'):
    """Return row-level diagnostics after strict baseline reproduction checks.

    Raises on any mismatch; outputs are returned in memory, not saved.
    """
    baseline = pd.read_csv(baseline_csv)
    match = baseline.loc[(baseline['model'] == model_name) &
                         (baseline['fold_number'] == fold_number) &
                         (baseline['cost_scenario'] == cost_scenario)]
    if len(match) != 1:
        raise ValueError(f'Expected exactly one baseline row; found {len(match)}')
    reference = match.iloc[0]
    cfg = load_config(cfg_path)
    df = pd.read_csv(cfg['data']['path'])
    if cfg['preprocessing']['encode_product_id']:
        encoder = LabelEncoder()
        df['Product_ID'] = encoder.fit_transform(df['Product ID'].astype(str).str[0])
    X = df[cfg['features']['input']].copy()
    X.columns = (X.columns.str.replace('[', '(', regex=False)
        .str.replace(']', ')', regex=False).str.replace('<', 'lt', regex=False)
        .str.replace(' ', '_', regex=False))
    y = df[cfg['features']['target']].astype(int)
    cv = cfg['repeated_cv']
    seed = int(cv['random_state'])
    splitter = RepeatedStratifiedKFold(n_splits=int(cv['n_splits']),
        n_repeats=int(cv['n_repeats']), random_state=seed)
    if not 1 <= fold_number <= int(cv['n_splits']) * int(cv['n_repeats']):
        raise ValueError('Invalid fold_number')
    dev_idx, test_idx = next(pair for i, pair in enumerate(splitter.split(X, y), 1)
                                 if i == fold_number)
    partitions = _split_development(dev_idx, y, seed + 1000 * fold_number)
    train = partitions['train']
    probcal = partitions['probability_calibration']
    confcal = partitions['conformal_calibration']
    tune = partitions['policy_tuning']
    model = build_model_fn(cfg, y.iloc[train])
    model.fit(X.iloc[train], y.iloc[train])
    calibrated = _fit_calibrator(model, X.iloc[probcal], y.iloc[probcal],
                                 cfg['calibration']['method'])
    q = compute_conformal_quantile(calibrated.predict_proba(X.iloc[confcal]),
                                    y.iloc[confcal], alpha=cfg['conformal']['alpha'])
    p_tune = calibrated.predict_proba(X.iloc[tune])
    threshold = _choose_threshold(p_tune, y.iloc[tune].to_numpy(), cfg)
    cp_tune = build_conformal_sets(p_tune, q)
    costs = tuple(int(v) for v in cost_scenario.split('_'))
    taus = np.linspace(cfg['deferral']['tau_min'], cfg['deferral']['tau_max'],
                       int(cfg['deferral']['steps']))
    winner, _ = _tune_tau(p_tune, y.iloc[tune].to_numpy(), threshold, cp_tune,
                          taus, costs, cfg['conformal']['defer_on_ambiguous'])
    p_test = calibrated.predict_proba(X.iloc[test_idx])
    y_test = y.iloc[test_idx].to_numpy()
    cp_test = build_conformal_sets(p_test, q)
    frozen = evaluate_deferral_policy(p_test, y_test, threshold, winner['tau'],
                                      *costs, cp_sets=cp_test,
                                      defer_on_ambiguous=cfg['conformal']['defer_on_ambiguous'])
    checks = {
        'threshold': threshold, 'q_alpha': q, 'best_tau': winner['tau'],
        'tuning_cost_per_sample': winner['cost_per_sample'],
        'best_cost_per_sample': frozen['cost_per_sample'],
        'best_deferred_fraction': frozen['deferred_fraction'],
        'best_retained_recall': frozen['retained_recall'],
        'best_retained_precision': frozen['retained_precision'],
        'fn': frozen['fn'], 'fp': frozen['fp'],
        'deferred_count': frozen['deferred_count'],
    }
    mismatches = {k: (v, reference[k]) for k, v in checks.items()
                  if not np.isclose(v, reference[k], rtol=1e-7, atol=1e-9,
                                    equal_nan=True)}
    if mismatches:
        raise AssertionError(f'Baseline reproduction mismatch for {model_name}: {mismatches}')

    sizes = np.array([len(s) for s in cp_test])
    singleton_labels = np.array([s[0] if len(s) == 1 else -1 for s in cp_test])
    pred = (p_test[:, 1] >= threshold).astype(int)
    certainty = p_test.max(axis=1)
    defer = certainty < winner['tau']
    if cfg['conformal']['defer_on_ambiguous']:
        defer |= sizes != 1
    retained = ~defer
    disagree = retained & (sizes == 1) & (pred != singleton_labels)
    rows = pd.DataFrame({
        'model': model_name, 'fold_number': fold_number,
        'cost_scenario': cost_scenario, 'dataset_row_index': test_idx,
        'y_true': y_test, 'p_failure': p_test[:, 1],
        'cost_sensitive_prediction': pred,
        'conformal_set_size': sizes, 'singleton_label': singleton_labels,
        'deferred': defer, 'retained': retained,
        'singleton_disagreement': disagree,
        'disagreement_false_positive': disagree & (pred == 1) & (y_test == 0),
        'disagreement_false_negative': disagree & (pred == 0) & (y_test == 1),
    })
    n_retained = int(retained.sum())
    summary = {
        'model': model_name, 'fold_number': fold_number,
        'cost_scenario': cost_scenario, 'baseline_reproduced': True,
        'n_test': len(rows), 'n_retained': n_retained,
        'n_disagreements': int(disagree.sum()),
        'disagreement_fraction_retained': float(disagree.sum() / n_retained)
            if n_retained else np.nan,
        'predicted_failure_cp_zero': int((disagree & (pred == 1) &
                                          (singleton_labels == 0)).sum()),
        'predicted_normal_cp_one': int((disagree & (pred == 0) &
                                        (singleton_labels == 1)).sum()),
        'disagreement_false_positives': int(rows['disagreement_false_positive'].sum()),
        'disagreement_false_negatives': int(rows['disagreement_false_negative'].sum()),
    }
    return summary, rows
