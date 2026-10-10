"""Exact-checkpoint aggregation for query-budget experiments."""

from __future__ import annotations


def aggregate_query_budgets(rows: list[dict], budgets: list[int]) -> dict:
    """Aggregate observed rows without dropping failures or interpolating checkpoints."""
    buckets = sorted({row["bucket"] for row in rows})
    result = {}
    for bucket in buckets:
        result[bucket] = {}
        for budget in budgets:
            observed = [
                row
                for row in rows
                if row["bucket"] == bucket and row["budget"] == budget
            ]
            successes = [row for row in observed if row.get("success", False)]
            scores = [row["score"] for row in successes if row.get("score") is not None]
            attempted = len(observed)
            succeeded = len(successes)
            result[bucket][budget] = {
                "attempted": attempted,
                "succeeded": succeeded,
                "failures": attempted - succeeded,
                "asr": succeeded / attempted if attempted else None,
                "mean_score": sum(scores) / len(scores) if scores else None,
            }
    return result
