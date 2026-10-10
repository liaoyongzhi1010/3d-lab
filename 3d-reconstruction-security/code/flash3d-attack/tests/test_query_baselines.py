import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from attacks.blackbox_api import CountingRenderAPI
from query_baselines import random_search_attack, square_attack


def _renderer(image, pose):
    return image * (1.0 + pose[0, 3])


def _run(attack, budget):
    image = torch.full((1, 3, 8, 8), 0.5)
    poses = [torch.eye(4)]
    poses[0][0, 3] = 0.2
    api = CountingRenderAPI(_renderer, budget=budget)
    attacked, delta, info = attack(
        image, api, poses, epsilon=0.1, lambda_src=1.0, seed=4
    )
    return image, attacked, delta, info, api


def test_random_search_never_overshoots_and_uses_ledger_count():
    image, attacked, delta, info, api = _run(random_search_attack, budget=7)
    assert api.queries == info["queries"] == 6
    assert api.queries <= api.budget
    assert delta.abs().max().item() <= 0.1 + 1e-6
    assert torch.equal(attacked - image, delta)


def test_square_attack_never_overshoots_and_returns_queried_best():
    _, _, delta, info, api = _run(square_attack, budget=9)
    assert api.queries == info["queries"] == 8
    assert info["candidate_queries"] == [4, 6, 8]
    assert info["best_query"] in info["candidate_queries"]
    assert delta.abs().max().item() <= 0.1 + 1e-6
