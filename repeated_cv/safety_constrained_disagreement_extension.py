"""Step 4.4.2: safety-constrained, read-only disagreement-resolution experiment.

For each fold, reproduce the original baseline and tune a single global action
on disagreement cases using POLICY-TUNING labels only. Evaluate the frozen
choice on the independent TEST partition. Never writes baseline files.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import LabelEncoder
from repeated_cv.cv_helpers import (load_config, compute_conformal_quantile,
    build_conformal_sets, evaluate_deferral_policy)
from repeated_cv.repeated_cv_runner import (_split_development, _fit_calibrator,
    _choose_threshold, _tune_tau)

ACTIONS = ('keep_classifier', 'use_singleton', 'defer_disagreement')


def _policy_rows(probs, y, cp_sets, threshold, tau, costs, action, conformal_deferral=True):
    """Return aggregate cost and decisions for one frozen action."""
    y = np.asarray(y, dtype=int)
    pred = (probs[:, 1] >= threshold).astype(int)
    sizes = np.asarray([len(s) for s in cp_sets])
    labels = np.asarray([s[0] if len(s) == 1 else -1 for s in cp_sets])
    defer = probs.max(axis=1) < tau
    if conformal_deferral:
        defer |= sizes != 1
    mismatch = (~defer) & (sizes == 1) & (pred != labels)
    pred = pred.copy()
    if action == 'use_singleton':
        pred[mismatch] = labels[mismatch]
    elif action == 'defer_disagreement':
        defer |= mismatch
    elif action != 'keep_classifier':
        raise ValueError(f'Unknown action: {action}')
    keep = ~defer
    fn = int(np.sum(keep & (pred == 0) & (y == 1)))
    fp = int(np.sum(keep & (pred == 1) & (y == 0)))
    tp = int(np.sum(keep & (pred == 1) & (y == 1)))
    deferred = int(defer.sum())
    c_fn, c_fp, c_def = costs
    return dict(action=action, n=len(y), disagreements=int(mismatch.sum()),
        fn=fn, fp=fp, tp=tp, deferred_count=deferred,
        retained_count=int(keep.sum()),
        retained_positive_count=int(tp + fn),
        failure_count=int(np.sum(y == 1)),
        deferred_failure_count=int(np.sum(defer & (y == 1))),
        detected_failure_fraction=tp / np.sum(y == 1) if np.sum(y == 1) else np.nan,
        retained_recall=tp / (tp + fn) if (tp + fn) else np.nan,
        cost_per_sample=(c_fn * fn + c_fp * fp + c_def * deferred) / len(y))


def run_safety_constrained_extension(experiments, baseline_csv, *,
                               output_dir='Step4_Validation_Results/disagreement_safety_smoke',
                               fold_numbers=(1,), cost_scenarios=('10_1_1','20_1_2','30_1_5'),
                               recall_targets=(0.80, 0.90)):
    """Run diagnostic; creates a NEW directory and refuses overwriting it.

    experiments: [(model_name, cfg_path, build_model_fn), ...].
    Results: (test_results_df, tuning_actions_df). Test results contain all
    three fixed-action counterfactuals plus the tuning-selected action.
    """
    if not recall_targets or any(not 0 <= float(r) <= 1 for r in recall_targets):
        raise ValueError("recall_targets must be nonempty values between 0 and 1")
    out = Path(output_dir)
    if out.exists():
        raise FileExistsError(f'Refusing to overwrite existing output: {out}')
    baseline = pd.read_csv(baseline_csv)
    test_rows, tuning_rows = [], []
    # Check inputs before creating output directory.
    for name, _, _ in experiments:
        for fold in fold_numbers:
            for scenario in cost_scenarios:
                n = ((baseline.model == name) & (baseline.fold_number == fold) &
                     (baseline.cost_scenario == scenario)).sum()
                if n != 1:
                    raise ValueError(f'Expected one baseline row: {name}/{fold}/{scenario}, found {n}')
    out.mkdir(parents=True, exist_ok=False)
    for model_name, cfg_path, build_model_fn in experiments:
        cfg = load_config(cfg_path)
        df = pd.read_csv(cfg['data']['path'])
        if cfg['preprocessing']['encode_product_id']:
            encoder = LabelEncoder()
            df['Product_ID'] = encoder.fit_transform(df['Product ID'].astype(str).str[0])
        X = df[cfg['features']['input']].copy()
        X.columns = (X.columns.str.replace('[','(',regex=False)
                     .str.replace(']',')',regex=False)
                     .str.replace('<','lt',regex=False)
                     .str.replace(' ','_',regex=False))
        y = df[cfg['features']['target']].astype(int)
        cv = cfg['repeated_cv']
        seed = int(cv['random_state'])
        splitter = RepeatedStratifiedKFold(n_splits=int(cv['n_splits']),
                   n_repeats=int(cv['n_repeats']), random_state=seed)
        wanted = set(fold_numbers)
        if not wanted or min(wanted) < 1 or max(wanted) > int(cv['n_splits']) * int(cv['n_repeats']):
            raise ValueError('Invalid fold_numbers')
        for fold, (dev_idx, test_idx) in enumerate(splitter.split(X,y), start=1):
            if fold not in wanted:
                continue
            partitions = _split_development(dev_idx, y, seed + 1000 * fold)
            train, probcal, confcal, tune = (partitions[k] for k in
                ('train','probability_calibration','conformal_calibration','policy_tuning'))
            model = build_model_fn(cfg, y.iloc[train])
            model.fit(X.iloc[train], y.iloc[train])
            calibrated = _fit_calibrator(model, X.iloc[probcal], y.iloc[probcal],
                                         cfg['calibration']['method'])
            q = compute_conformal_quantile(calibrated.predict_proba(X.iloc[confcal]),
                   y.iloc[confcal], alpha=cfg['conformal']['alpha'])
            p_tune = calibrated.predict_proba(X.iloc[tune])
            y_tune = y.iloc[tune].to_numpy()
            cp_tune = build_conformal_sets(p_tune, q)
            threshold = _choose_threshold(p_tune, y_tune, cfg)
            p_test = calibrated.predict_proba(X.iloc[test_idx])
            y_test = y.iloc[test_idx].to_numpy()
            cp_test = build_conformal_sets(p_test, q)
            taus = np.linspace(cfg['deferral']['tau_min'],cfg['deferral']['tau_max'],
                               int(cfg['deferral']['steps']))
            for scenario in cost_scenarios:
                costs = tuple(map(int, scenario.split('_')))
                reference = baseline.loc[(baseline.model == model_name) &
                    (baseline.fold_number == fold) & (baseline.cost_scenario == scenario)].iloc[0]
                winner, _ = _tune_tau(p_tune, y_tune, threshold, cp_tune, taus,
                    costs, cfg['conformal']['defer_on_ambiguous'])
                frozen = evaluate_deferral_policy(p_test, y_test, threshold,
                    winner['tau'], *costs, cp_sets=cp_test,
                    defer_on_ambiguous=cfg['conformal']['defer_on_ambiguous'])
                checks = dict(threshold=threshold, q_alpha=q, best_tau=winner['tau'],
                    tuning_cost_per_sample=winner['cost_per_sample'],
                    best_cost_per_sample=frozen['cost_per_sample'],
                    best_deferred_fraction=frozen['deferred_fraction'],
                    best_retained_recall=frozen['retained_recall'],
                    best_retained_precision=frozen['retained_precision'],
                    fn=frozen['fn'], fp=frozen['fp'], deferred_count=frozen['deferred_count'])
                mismatches = {k:(v,reference[k]) for k,v in checks.items()
                    if not np.isclose(v, reference[k], rtol=1e-7, atol=1e-9, equal_nan=True)}
                if mismatches:
                    raise AssertionError(f'Baseline mismatch {model_name}/{fold}/{scenario}: {mismatches}')
                tuning = [_policy_rows(p_tune,y_tune,cp_tune,threshold,winner['tau'],
                    costs,action,cfg['conformal']['defer_on_ambiguous']) for action in ACTIONS]
                for target in recall_targets:
                    # A candidate with no retained positives is NOT treated as
                    # satisfying a recall constraint (undefined recall).
                    feasible = [r for r in tuning if
                        np.isfinite(r['retained_recall']) and
                        r['retained_recall'] >= target]
                    # Explicit fallback: original policy, even if infeasible.
                    # Report constraint failure; do not mislabel it as feasible.
                    if feasible:
                        selected = min(feasible, key=lambda r:
                            (r['cost_per_sample'], ACTIONS.index(r['action'])))['action']
                        selection_status = 'feasible'
                    else:
                        selected = 'keep_classifier'
                        selection_status = 'fallback_no_feasible_action'
                    for row in tuning:
                        tuning_rows.append(dict(model=model_name, fold_number=fold,
                            cost_scenario=scenario, recall_target=float(target),
                            selected_action=selected, selection_status=selection_status,
                            candidate_feasible=bool(np.isfinite(row['retained_recall']) and
                                row['retained_recall'] >= target), **row))
                    for action in ACTIONS:
                        result = _policy_rows(p_test,y_test,cp_test,threshold,winner['tau'],
                            costs,action,cfg['conformal']['defer_on_ambiguous'])
                        test_rows.append(dict(model=model_name, fold_number=fold,
                            cost_scenario=scenario, recall_target=float(target),
                            selected_action=selected, selection_status=selection_status,
                            is_tuning_selected=(action==selected), baseline_reproduced=True,
                            test_recall_target_met=bool(np.isfinite(result['retained_recall']) and
                                result['retained_recall'] >= target),
                            baseline_cost_per_sample=frozen['cost_per_sample'],
                            delta_cost_vs_baseline=result['cost_per_sample']-frozen['cost_per_sample'],
                            threshold=threshold, q_alpha=q, tau=winner['tau'], **result))
            pd.DataFrame(test_rows).to_csv(out/'PARTIAL_test_results.csv',index=False)
            pd.DataFrame(tuning_rows).to_csv(out/'PARTIAL_tuning_actions.csv',index=False)
            print(f'{model_name}: fold {fold} reproduced; disagreement actions evaluated', flush=True)
    test_df, tune_df = pd.DataFrame(test_rows), pd.DataFrame(tuning_rows)
    test_df.to_csv(out/'safety_constrained_test_results.csv',index=False)
    tune_df.to_csv(out/'safety_constrained_tuning_actions.csv',index=False)
    (out/'manifest.json').write_text(json.dumps(dict(folds=list(fold_numbers),
        scenarios=list(cost_scenarios), recall_targets=list(recall_targets),
        action_options=list(ACTIONS),
        selection='minimum tuning cost among candidates meeting retained recall target; tie -> original; no feasible -> original fallback and flag',
        note='Retained recall is conditional on nondeferred actual positives; not a test guarantee. Test labels only evaluate frozen choices; repeated folds overlap.'),indent=2))
    return test_df, tune_df
