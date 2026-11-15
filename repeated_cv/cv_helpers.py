# -----------------------------------------
# Importation of necessary dependencies
# -----------------------------------------
import yaml
import numpy as np

from sklearn.metrics import (
    recall_score,
    precision_score
)

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def expected_calibration_error(y_true, y_prob, n_bins=10):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        lower, upper = bins[i], bins[i + 1]
        mask = (y_prob > lower) & (y_prob <= upper)

        if mask.sum() == 0:
            continue

        bin_confidence = y_prob[mask].mean()
        bin_accuracy = y_true[mask].mean()
        bin_weight = mask.mean()

        ece += bin_weight * abs(bin_accuracy - bin_confidence)

    return float(ece)

def compute_conformal_quantile(probs_cal, y_cal, alpha=0.10):
    y_cal = np.asarray(y_cal).astype(int)
    true_probs = probs_cal[np.arange(len(y_cal)), y_cal]
    scores = 1.0 - true_probs

    m = len(scores)
    q_level = np.ceil((m + 1) * (1 - alpha)) / m
    q_level = min(q_level, 1.0)

    try:
        q_alpha = np.quantile(scores, q_level, method="higher")
    except TypeError:
        q_alpha = np.quantile(scores, q_level, interpolation="higher")

    return float(q_alpha)


def build_conformal_sets(probs, q_alpha):
    sets = []

    for row in probs:
        label_set = []
        for y in [0, 1]:
            if 1.0 - row[y] <= q_alpha:
                label_set.append(y)
        sets.append(label_set)

    return sets

def summarize_conformal(cp_sets, y_true):
    y_true = np.asarray(y_true).astype(int)

    covered = np.array([int(y in s) for y, s in zip(y_true, cp_sets)])
    sizes = np.array([len(s) for s in cp_sets])

    return {
        "conformal_coverage": covered.mean(),
        "avg_set_size": sizes.mean(),
        "singleton_rate": (sizes == 1).mean(),
        "ambiguous_rate": (sizes > 1).mean(),
        "empty_set_rate": (sizes == 0).mean()
    }

def evaluate_deferral_policy(
    probs,
    y_true,
    base_threshold,
    tau,
    c_fn,
    c_fp,
    c_def,
    cp_sets=None,
    defer_on_ambiguous=True
):
    y_true = np.asarray(y_true).astype(int)

    p1 = probs[:, 1]
    pred = (p1 >= base_threshold).astype(int)
    certainty = probs.max(axis=1)

    defer_mask = certainty < tau

    if cp_sets is not None and defer_on_ambiguous:
        cp_sizes = np.array([len(s) for s in cp_sets])
        defer_mask = defer_mask | (cp_sizes != 1)

    retained_mask = ~defer_mask

    fn = int(((pred == 0) & (y_true == 1) & retained_mask).sum())
    fp = int(((pred == 1) & (y_true == 0) & retained_mask).sum())
    tp = int(((pred == 1) & (y_true == 1) & retained_mask).sum())
    tn = int(((pred == 0) & (y_true == 0) & retained_mask).sum())

    deferred_count = int(defer_mask.sum())

    total_cost = c_fn * fn + c_fp * fp + c_def * deferred_count
    cost_per_sample = total_cost / len(y_true)

    if retained_mask.sum() > 0:
        retained_error = (pred[retained_mask] != y_true[retained_mask]).mean()
        retained_recall = recall_score(
            y_true[retained_mask],
            pred[retained_mask],
            zero_division=0
        )
        retained_precision = precision_score(
            y_true[retained_mask],
            pred[retained_mask],
            zero_division=0
        )
    else:
        retained_error = np.nan
        retained_recall = np.nan
        retained_precision = np.nan

    return {
        "tau": tau,
        "coverage": retained_mask.mean(),
        "deferred_fraction": defer_mask.mean(),
        "retained_error": retained_error,
        "retained_recall": retained_recall,
        "retained_precision": retained_precision,
        "fn": fn,
        "fp": fp,
        "tp": tp,
        "tn": tn,
        "deferred_count": deferred_count,
        "total_cost": total_cost,
        "cost_per_sample": cost_per_sample
    }
