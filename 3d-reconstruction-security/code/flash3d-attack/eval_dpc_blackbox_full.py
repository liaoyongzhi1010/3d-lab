"""Foundations for full-model black-box evaluation without running experiments."""

from __future__ import annotations

from attacks.blackbox_api import CountingRenderAPI
from attacks.dpc_blackbox import BlackBoxDPCConfig, blackbox_dpc_attack


def evaluate_blackbox_scene(
    image, renderer, config: BlackBoxDPCConfig, budget: int
) -> dict:
    """Run one scene under an exact query budget and return a row-level record."""
    api = CountingRenderAPI(renderer, budget=budget)
    try:
        attacked, delta, info = blackbox_dpc_attack(image, api, config)
    except Exception as exc:
        return {
            "schema": "query",
            "success": False,
            "failure": repr(exc),
            "budget": budget,
            "queries": api.queries,
        }
    return {
        "schema": "query",
        "success": True,
        "budget": budget,
        "queries": api.queries,
        "attacked": attacked,
        "delta": delta,
        "attack": info,
    }
