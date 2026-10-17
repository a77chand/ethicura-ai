"""
Detection metrics, with confidence intervals from a person-level bootstrap.

Photos of the same person aren't independent samples, so resampling images
would understate the uncertainty. The bootstrap resamples *people* (groups)
instead.
"""

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def eer(labels, scores) -> float:
    """Equal error rate: where false-positive rate equals false-negative rate."""
    fpr, tpr, _ = roc_curve(labels, scores)
    fnr = 1 - tpr
    i = np.nanargmin(np.abs(fnr - fpr))
    return float((fpr[i] + fnr[i]) / 2)


def tpr_at_fpr(labels, scores, target_fpr: float) -> float:
    """Detection rate when at most target_fpr of real faces are flagged."""
    fpr, tpr, _ = roc_curve(labels, scores)
    return float(np.interp(target_fpr, fpr, tpr))


def ece(labels, probs, n_bins: int = 15) -> float:
    """Expected calibration error over equal-width probability bins."""
    labels, probs = np.asarray(labels, float), np.asarray(probs, float)
    bins = np.minimum((probs * n_bins).astype(int), n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        mask = bins == b
        if mask.any():
            total += mask.mean() * abs(labels[mask].mean() - probs[mask].mean())
    return float(total)


def reliability(labels, probs, n_bins: int = 10):
    """(mean predicted, observed fake rate, count) per bin, for a diagram."""
    labels, probs = np.asarray(labels, float), np.asarray(probs, float)
    bins = np.minimum((probs * n_bins).astype(int), n_bins - 1)
    out = []
    for b in range(n_bins):
        mask = bins == b
        if mask.any():
            out.append((float(probs[mask].mean()), float(labels[mask].mean()), int(mask.sum())))
    return out


def balanced_indices(labels, seed: int = 0) -> np.ndarray:
    """All samples of the minority class plus an equal-size random draw from
    the majority — calibration is defined against a 50/50 prior, matching
    how the model is trained."""
    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(labels == 1), np.flatnonzero(labels == 0)
    small, large = (pos, neg) if len(pos) < len(neg) else (neg, pos)
    return np.sort(np.concatenate([small, rng.choice(large, len(small), replace=False)]))


def summary(labels, probs, threshold: float = 0.5) -> dict:
    labels = np.asarray(labels, int)
    probs = np.asarray(probs, float)
    pred = probs >= threshold
    tp = int((pred & (labels == 1)).sum())
    tn = int((~pred & (labels == 0)).sum())
    fp = int((pred & (labels == 0)).sum())
    fn = int((~pred & (labels == 1)).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {
        "n": int(len(labels)),
        "n_real": int((labels == 0).sum()),
        "n_fake": int((labels == 1).sum()),
        "auc": float(roc_auc_score(labels, probs)),
        "eer": eer(labels, probs),
        "tpr_at_fpr_1pct": tpr_at_fpr(labels, probs, 0.01),
        "tpr_at_fpr_5pct": tpr_at_fpr(labels, probs, 0.05),
        "accuracy": (tp + tn) / len(labels),
        "balanced_accuracy": 0.5 * (recall + tn / max(tn + fp, 1)),
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / max(precision + recall, 1e-12),
        "ece": ece(labels, probs),
    }


def bootstrap_auc_ci(labels, probs, groups, n_boot: int = 1000, seed: int = 0, alpha: float = 0.05):
    """95% CI for AUC, resampling groups (people / source videos) with replacement."""
    labels, probs, groups = np.asarray(labels), np.asarray(probs), np.asarray(groups)
    unique, inverse = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inverse == g) for g in range(len(unique))]
    rng = np.random.default_rng(seed)
    aucs = []
    for _ in range(n_boot):
        idx = np.concatenate([members[g] for g in rng.integers(0, len(unique), len(unique))])
        y = labels[idx]
        if y.min() == y.max():
            continue
        aucs.append(roc_auc_score(y, probs[idx]))
    lo, hi = np.quantile(aucs, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)
