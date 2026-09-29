import inspect
import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from attacks.blackbox_api import CountingRenderAPI
from eval_dpc_transfer import craft_transfer_attack, evaluate_transfer_artifact


def test_transfer_crafting_signature_has_no_target():
    parameters = inspect.signature(craft_transfer_attack).parameters
    assert not any("target" in name or name == "api" for name in parameters)


def test_target_is_never_queried_until_artifact_evaluation():
    image = torch.full((1, 3, 4, 4), 0.5)
    surrogate = lambda value, pose: value
    artifact = craft_transfer_attack(
        image, surrogate, method="random", epsilon=0.1, seed=3
    )

    target = CountingRenderAPI(lambda value, pose: value, budget=2)
    assert target.queries == 0
    result = evaluate_transfer_artifact(artifact, target, [torch.eye(4)])
    assert target.queries == 2
    assert result["queries"] == 2
    assert result["method"] == "random"


def test_transfer_artifact_respects_epsilon():
    image = torch.full((1, 3, 4, 4), 0.5)
    artifact = craft_transfer_attack(
        image, lambda value, pose: value, method="dpc", epsilon=0.05, seed=1
    )
    assert (artifact.attacked - image).abs().max().item() <= 0.05 + 1e-6
