import inspect
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from attacks.blackbox_api import BudgetExceeded, CountingRenderAPI


def test_ledger_charges_success_and_failure_with_metadata():
    calls = 0

    def renderer(image, pose):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("renderer failed")
        return image

    api = CountingRenderAPI(renderer, budget=3)
    image = torch.zeros(1, 3, 2, 2)
    pose = torch.eye(4)
    assert torch.equal(
        api.render(image, pose, phase="reference", purpose="source"), image
    )
    with pytest.raises(RuntimeError, match="renderer failed"):
        api.render(image, pose, phase="attack", purpose="candidate")

    assert api.queries == 2
    assert [record.success for record in api.records] == [True, False]
    assert [(record.phase, record.purpose) for record in api.records] == [
        ("reference", "source"),
        ("attack", "candidate"),
    ]


def test_budget_rejection_does_not_charge_or_call_renderer():
    calls = 0

    def renderer(image, pose):
        nonlocal calls
        calls += 1
        return image

    api = CountingRenderAPI(renderer, budget=1)
    image = torch.zeros(1, 3, 2, 2)
    api.render(image, torch.eye(4), phase="reference", purpose="source")
    with pytest.raises(BudgetExceeded):
        api.render(image, torch.eye(4), phase="attack", purpose="candidate")
    assert calls == api.queries == 1
    assert len(api.records) == 1


def test_api_enforces_one_image_and_one_pose_before_charging():
    api = CountingRenderAPI(lambda image, pose: image, budget=5)
    with pytest.raises(ValueError, match="B=1"):
        api.render(torch.zeros(2, 3, 2, 2), torch.eye(4), phase="x", purpose="y")
    with pytest.raises(ValueError, match="4, 4"):
        api.render(
            torch.zeros(1, 3, 2, 2), torch.eye(4).unsqueeze(0), phase="x", purpose="y"
        )
    assert api.queries == 0


def test_clean_references_charge_source_and_each_novel_pose():
    api = CountingRenderAPI(lambda image, pose: image + pose[0, 3], budget=5)
    image = torch.zeros(1, 3, 2, 2)
    poses = [torch.eye(4) for _ in range(4)]
    source, novel = api.clean_references(image, poses)
    assert source.shape == image.shape
    assert len(novel) == 4
    assert api.queries == 5
    assert all(record.phase == "reference" for record in api.records)


def test_public_render_requires_phase_and_purpose():
    signature = inspect.signature(CountingRenderAPI.render)
    assert signature.parameters["phase"].default is inspect.Parameter.empty
    assert signature.parameters["purpose"].default is inspect.Parameter.empty
