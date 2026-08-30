from __future__ import annotations

import torch

from robosynchallenge.managers.events import _projected_obb_separation_2d


def pose(x: float, y: float, yaw_deg: float = 0.0) -> torch.Tensor:
    yaw = torch.tensor(yaw_deg * torch.pi / 180.0)
    c, s = torch.cos(yaw), torch.sin(yaw)
    value = torch.eye(4).unsqueeze(0)
    value[0, :2, :2] = torch.tensor([[c, -s], [s, c]])
    value[0, 0, 3] = x
    value[0, 1, 3] = y
    return value


def test_projected_obb_separation_reports_gap():
    first = pose(0.0, 0.0)
    second = pose(2.25, 0.0)
    half_extents = torch.tensor([1.0, 0.5, 0.25])
    gap = _projected_obb_separation_2d(first, second, half_extents, half_extents)
    assert torch.allclose(gap, torch.tensor([0.25]), atol=1e-6)


def test_projected_obb_separation_rejects_overlap_after_rotation():
    first = pose(0.0, 0.0, 30.0)
    second = pose(0.5, 0.0, -20.0)
    half_extents = torch.tensor([1.0, 0.5, 0.25])
    gap = _projected_obb_separation_2d(first, second, half_extents, half_extents)
    assert gap.item() < 0.0
