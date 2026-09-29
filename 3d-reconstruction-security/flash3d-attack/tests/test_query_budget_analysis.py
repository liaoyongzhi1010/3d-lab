import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from query_budget_analysis import aggregate_query_budgets


def test_aggregation_retains_failures_in_asr_denominator():
    rows = [
        {
            "scene": "a",
            "bucket": "target_5",
            "budget": 1000,
            "success": True,
            "score": 2.0,
        },
        {
            "scene": "b",
            "bucket": "target_5",
            "budget": 1000,
            "success": False,
            "score": None,
            "failure": "oom",
        },
        {
            "scene": "c",
            "bucket": "target_5",
            "budget": 5000,
            "success": True,
            "score": 3.0,
        },
    ]
    result = aggregate_query_budgets(rows, budgets=[1000, 5000])
    at_1000 = result["target_5"][1000]
    assert at_1000["attempted"] == 2
    assert at_1000["succeeded"] == 1
    assert at_1000["asr"] == 0.5
    assert at_1000["failures"] == 1


def test_aggregation_does_not_interpolate_missing_budgets_and_separates_buckets():
    rows = [
        {
            "scene": "a",
            "bucket": "source",
            "budget": 1000,
            "success": True,
            "score": 1.0,
        },
        {
            "scene": "a",
            "bucket": "target_10",
            "budget": 5000,
            "success": True,
            "score": 2.0,
        },
    ]
    result = aggregate_query_budgets(rows, budgets=[1000, 5000])
    assert set(result) == {"source", "target_10"}
    assert result["source"][5000]["attempted"] == 0
    assert result["source"][5000]["asr"] is None
    assert result["target_10"][1000]["attempted"] == 0
    assert result["target_10"][1000]["mean_score"] is None
