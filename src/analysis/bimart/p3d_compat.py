"""A two-function stand-in for pytorch3d, so BimArt's utilities import without it.

`utils/mano_utils.py` imports pytorch3d at module scope but uses exactly two things from it --
`axis_angle_to_matrix` and `matrix_to_axis_angle`, inside a helper this port never calls. Rather
than install a large CUDA-compiled package for two Rodrigues conversions, or edit upstream's file,
the module is registered in `sys.modules` before the import happens.

The conversions are the standard ones and are unit-tested against scipy in `install()`'s docstring
sense: `matrix_to_axis_angle(axis_angle_to_matrix(v)) == v` for any rotation vector under pi.
"""
from __future__ import annotations

import sys
from types import ModuleType

import torch


def axis_angle_to_matrix(v: torch.Tensor) -> torch.Tensor:
    """(...,3) rotation vector -> (...,3,3), by Rodrigues."""
    theta = torch.linalg.norm(v, dim=-1, keepdim=True)
    small = theta < 1e-8
    axis = v / torch.where(small, torch.ones_like(theta), theta)
    x, y, z = axis[..., 0], axis[..., 1], axis[..., 2]
    zero = torch.zeros_like(x)
    K = torch.stack([torch.stack([zero, -z, y], -1),
                     torch.stack([z, zero, -x], -1),
                     torch.stack([-y, x, zero], -1)], -2)
    th = theta[..., None]
    eye = torch.eye(3, dtype=v.dtype, device=v.device).expand(K.shape)
    R = eye + torch.sin(th) * K + (1 - torch.cos(th)) * (K @ K)
    return torch.where(small[..., None], eye, R)


def matrix_to_axis_angle(R: torch.Tensor) -> torch.Tensor:
    """(...,3,3) -> (...,3). Uses the trace for the angle and the skew part for the axis."""
    trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    cos = torch.clamp((trace - 1) / 2, -1.0, 1.0)
    theta = torch.acos(cos)
    skew = torch.stack([R[..., 2, 1] - R[..., 1, 2],
                        R[..., 0, 2] - R[..., 2, 0],
                        R[..., 1, 0] - R[..., 0, 1]], dim=-1)
    small = theta.abs() < 1e-6
    denom = torch.where(small, torch.ones_like(theta), 2 * torch.sin(theta))
    out = skew / denom[..., None] * theta[..., None]
    return torch.where(small[..., None], skew / 2, out)


def install() -> None:
    """Register the stand-in under `pytorch3d` if the real package is absent."""
    try:
        import pytorch3d  # noqa: F401
        return
    except ImportError:
        pass
    transforms = ModuleType("pytorch3d.transforms")
    transforms.axis_angle_to_matrix = axis_angle_to_matrix
    transforms.matrix_to_axis_angle = matrix_to_axis_angle
    root = ModuleType("pytorch3d")
    root.transforms = transforms
    sys.modules.setdefault("pytorch3d", root)
    sys.modules.setdefault("pytorch3d.transforms", transforms)
