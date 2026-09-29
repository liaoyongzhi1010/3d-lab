"""Paired inference and evidence-validity helpers."""

import numpy as np
from scipy import stats


def _paired(candidate, baseline):
    candidate = np.asarray(candidate, dtype=float)
    baseline = np.asarray(baseline, dtype=float)
    if candidate.ndim != 1 or candidate.shape != baseline.shape:
        raise ValueError(
            "paired samples must have the same length and be one-dimensional"
        )
    if (
        len(candidate) == 0
        or not np.isfinite(candidate).all()
        or not np.isfinite(baseline).all()
    ):
        raise ValueError("paired samples must be non-empty and finite")
    return candidate, baseline


def paired_bootstrap_ci(
    candidate, baseline, *, seed=0, n_resamples=10000, confidence=0.95
):
    candidate, baseline = _paired(candidate, baseline)
    differences = candidate - baseline
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(differences), size=(n_resamples, len(differences)))
    samples = differences[indices].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return {
        "estimate": float(differences.mean()),
        "low": float(np.quantile(samples, alpha)),
        "high": float(np.quantile(samples, 1.0 - alpha)),
        "count": len(differences),
    }


def paired_wilcoxon(candidate, baseline):
    candidate, baseline = _paired(candidate, baseline)
    result = stats.wilcoxon(candidate, baseline)
    return {
        "statistic": float(result.statistic),
        "pvalue": float(result.pvalue),
        "count": len(candidate),
    }


def spearman_bootstrap(x, y, *, seed=0, n_resamples=10000, confidence=0.95):
    x, y = _paired(x, y)
    rho = float(stats.spearmanr(x, y).statistic)
    rng = np.random.default_rng(seed)
    bootstrapped = []
    for _ in range(n_resamples):
        indices = rng.integers(0, len(x), size=len(x))
        value = stats.spearmanr(x[indices], y[indices]).statistic
        if np.isfinite(value):
            bootstrapped.append(value)
    if not bootstrapped:
        raise ValueError("no valid Spearman bootstrap resamples")
    alpha = (1.0 - confidence) / 2.0
    return {
        "rho": rho,
        "low": float(np.quantile(bootstrapped, alpha)),
        "high": float(np.quantile(bootstrapped, 1.0 - alpha)),
        "count": len(x),
    }


def aggregate_seeds(values):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError(
            "seed aggregation requires at least two seeds with finite values"
        )
    return {
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)),
        "count": len(values),
    }


def success_rate(successes, *, total):
    successes = tuple(bool(value) for value in successes)
    if total <= 0 or len(successes) > total:
        raise ValueError("total must retain all attempted scenes")
    count = sum(successes)
    return {
        "rate": count / total,
        "successes": count,
        "total": total,
        "missing": total - len(successes),
    }


def fid_rankable(sample_count):
    return int(sample_count) >= 2048
