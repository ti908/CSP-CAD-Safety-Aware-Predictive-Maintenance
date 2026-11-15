"""Leakage-free repeated stratified CV; historical runner retained separately.

Each outer 80% development fold is split 50/10/10/10 percentage points of
all observations. The outer 20% test labels never select a threshold or tau.
"""
from pathlib import Path
from datetime import datetime, timezone
import json
import uuid

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (recall_score, precision_score, f1_score, roc_auc_score,
                             average_precision_score, brier_score_loss, log_loss)
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder

from repeated_cv.cv_helpers import (load_config, expected_calibration_error,
    compute_conformal_quantile, build_conformal_sets, summarize_conformal,
    evaluate_deferral_policy)

COST_SCENARIOS = ((10, 1, 1), (20, 1, 2), (30, 1, 5))


def _split_development(dev_idx, y, seed):
    """From 80% development, return train=50%, each other partition=10%."""
    dev_idx = np.asarray(dev_idx)
    remaining, policy = train_test_split(dev_idx, test_size=1/8,
        stratify=y.iloc[dev_idx], random_state=seed)
    remaining, conformal = train_test_split(remaining, test_size=1/7,
        stratify=y.iloc[remaining], random_state=seed + 1)
    train, probability = train_test_split(remaining, test_size=1/6,
        stratify=y.iloc[remaining], random_state=seed + 2)
    return {'train': np.asarray(train), 'probability_calibration': np.asarray(probability),
            'conformal_calibration': np.asarray(conformal), 'policy_tuning': np.asarray(policy)}


def _fit_calibrator(fitted_model, X_cal, y_cal, method):
    try:
        from sklearn.frozen import FrozenEstimator
        calibrated = CalibratedClassifierCV(FrozenEstimator(fitted_model), method=method)
    except ImportError:
        calibrated = CalibratedClassifierCV(fitted_model, method=method, cv='prefit')
    calibrated.fit(X_cal, y_cal)
    return calibrated


def _choose_threshold(probs, y, cfg):
    """Classification threshold is selected exclusively on policy tuning."""
    thresholds = np.linspace(cfg['threshold'].get('search_min', .01),
        cfg['threshold'].get('search_max', .5), cfg['threshold'].get('steps', 100))
    target = cfg['threshold'].get('target_recall', .90)
    scores = []
    for t in thresholds:
        pred = (probs[:, 1] >= t).astype(int)
        scores.append((float(t), float(recall_score(y, pred, zero_division=0)),
                       float(precision_score(y, pred, zero_division=0))))
    valid = [x for x in scores if x[1] >= target]
    if valid:
        return max(valid, key=lambda x: (x[2], x[0]))[0]
    return max(scores, key=lambda x: (x[1], x[2], x[0]))[0]


def _set_diagnostics(cp_sets, y):
    y = np.asarray(y, dtype=int)
    sizes = np.asarray([len(s) for s in cp_sets])
    failures = y == 1
    return {'empty_set_fraction': float(np.mean(sizes == 0)),
            'singleton_set_fraction': float(np.mean(sizes == 1)),
            'two_label_set_fraction': float(np.mean(sizes == 2)),
            'failure_class_coverage': float(np.mean([1 in cp_sets[i] for i in np.flatnonzero(failures)]))
                if failures.any() else float('nan')}


def _tune_tau(probs, y, threshold, cp_sets, taus, costs, defer_on_ambiguous):
    """The only input labels here are from the policy-tuning partition."""
    candidates = [evaluate_deferral_policy(probs, y, threshold, float(tau),
        *costs, cp_sets=cp_sets, defer_on_ambiguous=defer_on_ambiguous)
        for tau in taus]
    return min(candidates, key=lambda p: (p['cost_per_sample'],
                                         p['deferred_fraction'], p['tau'])), candidates


def repeated_cv_safety_eval(model_name, cfg_path, build_model_fn, *,
                            cost_scenarios=COST_SCENARIOS, max_folds=None,
                            output_root='corrected_repeated_cv_results',
                            experiment_id=None):
    """Return (frozen test summaries, tuning-only curves).

    max_folds is for sanity testing only; omit it for the complete 5x10 study.
    All outputs are written to a unique directory, never historical paths.
    """
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
    cv_cfg = cfg['repeated_cv']
    n_splits = int(cv_cfg['n_splits'])
    n_repeats = int(cv_cfg['n_repeats'])
    random_seed = int(cv_cfg['random_state'])
    splitter = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats,
                                       random_state=random_seed)
    taus = np.linspace(cfg['deferral']['tau_min'], cfg['deferral']['tau_max'],
                       int(cfg['deferral']['steps']))
    experiment_id = experiment_id or (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
                                     + '_' + uuid.uuid4().hex[:8])
    output = Path(output_root) / experiment_id / model_name
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite previous run: {output}')
    output.mkdir(parents=True, exist_ok=False)
    summary_rows, tuning_rows = [], []
    for fold_number, (dev_idx, test_idx) in enumerate(splitter.split(X, y), start=1):
        if max_folds is not None and fold_number > max_folds:
            break
        fold_seed = random_seed + 1000 * fold_number
        indices = _split_development(dev_idx, y, fold_seed)
        indices['test'] = np.asarray(test_idx)
        sizes = {f'n_{key}': len(idx) for key, idx in indices.items()}
        positives = {f'positive_{key}': int(y.iloc[idx].sum()) for key, idx in indices.items()}
        assert sum(sizes.values()) == len(df)
        assert len(np.unique(np.concatenate(list(indices.values())))) == len(df), 'Partition overlap!'
        assert all(positives.values()), 'Stratification yielded an empty failure class!'
        model = build_model_fn(cfg, y.iloc[indices['train']])
        model.fit(X.iloc[indices['train']], y.iloc[indices['train']])
        calibrated = _fit_calibrator(model, X.iloc[indices['probability_calibration']],
                                     y.iloc[indices['probability_calibration']],
                                     cfg['calibration']['method'])
        p_conformal = calibrated.predict_proba(X.iloc[indices['conformal_calibration']])
        q_alpha = compute_conformal_quantile(p_conformal,
            y.iloc[indices['conformal_calibration']], alpha=cfg['conformal']['alpha'])
        p_tune = calibrated.predict_proba(X.iloc[indices['policy_tuning']])
        y_tune = y.iloc[indices['policy_tuning']].to_numpy()
        cp_tune = build_conformal_sets(p_tune, q_alpha)
        threshold = _choose_threshold(p_tune, y_tune, cfg)
        # Only now load outer test probabilities/labels for frozen-policy evaluation.
        p_test = calibrated.predict_proba(X.iloc[indices['test']])
        y_test = y.iloc[indices['test']].to_numpy()
        cp_test = build_conformal_sets(p_test, q_alpha)
        cp_summary = summarize_conformal(cp_test, y_test)
        set_metrics = _set_diagnostics(cp_test, y_test)
        assert np.isclose(sum(set_metrics[k] for k in
            ('empty_set_fraction', 'singleton_set_fraction', 'two_label_set_fraction')), 1.0)
        pred_test = (p_test[:, 1] >= threshold).astype(int)
        common = {'model': model_name, 'run_id': experiment_id,
                  'random_seed': random_seed, 'fold_seed': fold_seed,
                  'fold_number': fold_number,
                  'repeat': (fold_number - 1) // n_splits + 1,
                  'fold': (fold_number - 1) % n_splits + 1,
                  'threshold': threshold, 'q_alpha': q_alpha,
                  'alpha': float(cfg['conformal']['alpha']),
                  **sizes, **positives}
        metrics = {'roc_auc': roc_auc_score(y_test, p_test[:, 1]),
                   'pr_auc': average_precision_score(y_test, p_test[:, 1]),
                   'recall': recall_score(y_test, pred_test, zero_division=0),
                   'precision': precision_score(y_test, pred_test, zero_division=0),
                   'f1': f1_score(y_test, pred_test, zero_division=0),
                   'brier_score': brier_score_loss(y_test, p_test[:, 1]),
                   'log_loss': log_loss(y_test, p_test[:, 1], labels=[0, 1]),
                   'ece': expected_calibration_error(y_test, p_test[:, 1],
                                                    n_bins=cfg['evaluation']['ece_bins']),
                   'conformal_coverage': cp_summary['conformal_coverage'],
                   'avg_conformal_set_size': cp_summary['avg_set_size'],
                   'ambiguous_rate': cp_summary['ambiguous_rate'],
                   **set_metrics}
        for costs in cost_scenarios:
            costs = tuple(map(int, costs))
            scenario = '_'.join(map(str, costs))
            cost_meta = {'cost_scenario': scenario, 'C_FN': costs[0],
                         'C_FP': costs[1], 'C_DEF': costs[2]}
            winner, curve = _tune_tau(p_tune, y_tune, threshold, cp_tune, taus,
                                     costs, cfg['conformal']['defer_on_ambiguous'])
            for policy in curve:
                tuning_rows.append({**common, **cost_meta, 'partition': 'policy_tuning',
                    'tau': policy['tau'], 'coverage': policy['coverage'],
                    'cost_per_sample': policy['cost_per_sample'],
                    'retained_error': policy['retained_error'],
                    'deferred_fraction': policy['deferred_fraction'],
                    'retained_recall': policy['retained_recall'],
                    'retained_precision': policy['retained_precision']})
            frozen = evaluate_deferral_policy(p_test, y_test, threshold, winner['tau'],
                *costs, cp_sets=cp_test,
                defer_on_ambiguous=cfg['conformal']['defer_on_ambiguous'])
            summary_rows.append({**common, **cost_meta, 'partition': 'test',
                'best_tau': winner['tau'], 'tuning_cost_per_sample': winner['cost_per_sample'],
                'best_cost_per_sample': frozen['cost_per_sample'],
                'best_coverage': frozen['coverage'],
                'best_deferred_fraction': frozen['deferred_fraction'],
                'best_retained_error': frozen['retained_error'],
                'best_retained_recall': frozen['retained_recall'],
                'best_retained_precision': frozen['retained_precision'],
                'fn': frozen['fn'], 'fp': frozen['fp'],
                'deferred_count': frozen['deferred_count'], **metrics})
        print(f'{model_name}: outer fold {fold_number}/{n_splits * n_repeats} '
              f'completed; partition counts={sizes}; failures={positives}')
    fold_df, curve_df = pd.DataFrame(summary_rows), pd.DataFrame(tuning_rows)
    fold_df.to_csv(output / 'repeated_cv_fold_summary.csv', index=False)
    curve_df.to_csv(output / 'repeated_cv_TUNING_ONLY_curve_records.csv', index=False)
    manifest = {'run_id': experiment_id, 'model': model_name, 'random_seed': random_seed,
        'folds_executed': int(fold_df['fold_number'].nunique()),
        'expected_full_folds': n_splits * n_repeats,
        'sanity_only': max_folds is not None,
        'cost_scenarios': [list(c) for c in cost_scenarios],
        'curve_partition': 'policy_tuning', 'summary_partition': 'test',
        'configuration': str(cfg_path), 'output': str(output)}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    return fold_df, curve_df
