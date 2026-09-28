"""The posed MANO hand: FK plus the link geometry that contact and penetration need.

`ManoFK` (src/data/mano.py) answers "where are the link FRAMES". This adds "and what SURFACE do
they carry", which is what penetration, collision and contact recomputation all actually need.

The 16 collision meshes come from the same URDF the FK is parsed from, so a link's mesh and its
frame cannot disagree. Surface samples are drawn ONCE per link, seeded, and then carried by the
link -- resampling per frame would make a metric noisy in a way that looks like a result.

WHY `palm` MATTERS HERE. Genesis folds a fixed-joint child into its parent, so a simulator-side
hand has no `palm` link at all and silently loses a quarter of the hand's contact surface. dexcore
poses the RAW URDF tree, so `palm` stays addressable and the palm's geometry is present.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

from src import paths
from src.data.mano import LINK_NAMES, NUM_LINKS, ManoFK, get_fk


def _parse_link_meshes(urdf_path, tag: str) -> Dict[str, Path]:
    """{link name -> mesh path} for one of the URDF's mesh tags ("visual" or "collision")."""
    d = Path(urdf_path).parent
    root = ET.parse(str(urdf_path)).getroot()
    out = {}
    for ln in root.findall("link"):
        el = ln.find(f"{tag}/geometry/mesh")
        if el is not None and el.get("filename"):
            out[ln.get("name")] = d / el.get("filename")
    return out


@dataclass
class HandModel:
    """One side's MANO hand: FK, per-link collision geometry, and seeded surface samples."""

    side: str
    fk: ManoFK
    mesh_files: Dict[str, Path]            # collision meshes
    visual_mesh_files: Dict[str, Path] = field(default_factory=dict)
    samples_per_link: int = 64
    seed: int = 0
    _local_points: Optional[np.ndarray] = field(default=None, repr=False)   # (16, S, 3)

    @classmethod
    def load(cls, side: str, device="cpu", samples_per_link: int = 64, seed: int = 0,
             dtype=torch.float32) -> "HandModel":
        urdf = paths.mano_urdf(side)
        return cls(side=side, fk=get_fk(side, device=device, dtype=dtype),
                   mesh_files=_parse_link_meshes(urdf, "collision"),
                   visual_mesh_files=_parse_link_meshes(urdf, "visual"),
                   samples_per_link=int(samples_per_link), seed=int(seed))

    # ------------------------------------------------------------------ geometry
    def link_mesh(self, link: str, visual: bool = False):
        """One link's mesh. `visual=True` is the hand SKIN -- what contact is defined against in
        `process_arctic`; the collision meshes are a coarser proxy used for speed."""
        import trimesh
        files = self.visual_mesh_files if visual else self.mesh_files
        if link not in files:
            raise KeyError(f"no {'visual' if visual else 'collision'} mesh for link {link!r} "
                           f"in {self.fk.urdf_path}")
        return trimesh.load(str(files[link]), process=False, force="mesh")

    def skin_points_world(self, q) -> np.ndarray:
        """(F, V, 3) every VISUAL vertex of every posed link -- the hand surface `process_arctic`
        measures contact against. Vertices, not samples: the reference uses the mesh directly."""
        T = self.link_frames(q)
        out = []
        for i, name in enumerate(LINK_NAMES):
            v = np.asarray(self.link_mesh(name, visual=True).vertices, dtype=np.float64)
            out.append(np.einsum("fij,vj->fvi", T[:, i, :3, :3], v) + T[:, i, None, :3, 3])
        return np.concatenate(out, axis=1)

    def local_points(self) -> np.ndarray:
        """(16, S, 3) link-local surface samples, drawn once and cached.

        Seeded per link so the same hand always yields the same points -- a metric that changed
        run to run because its sampling did would be indistinguishable from a real effect.
        """
        if self._local_points is None:
            import trimesh
            pts = []
            for i, name in enumerate(LINK_NAMES):
                m = self.link_mesh(name)
                p, _ = trimesh.sample.sample_surface(
                    m, self.samples_per_link, seed=self.seed * 1000 + i)
                pts.append(np.asarray(p, dtype=np.float64))
            self._local_points = np.stack(pts)
        return self._local_points

    # ------------------------------------------------------------------ posing
    def link_frames(self, q: torch.Tensor) -> np.ndarray:
        """(F,16,4,4) world pose of every contact link at hand pose q."""
        with torch.no_grad():
            return self.fk.forward(q).cpu().numpy().astype(np.float64)

    def link_points_world(self, q: torch.Tensor) -> np.ndarray:
        """(F,16,S,3) world-space surface samples of every link at hand pose q."""
        T = self.link_frames(q)                                     # (F,16,4,4)
        loc = self.local_points()                                   # (16,S,3)
        return np.einsum("flij,lsj->flsi", T[:, :, :3, :3], loc) + T[:, :, None, :3, 3]

    def keypoints(self, q: torch.Tensor, local_offsets: torch.Tensor) -> np.ndarray:
        """(F,21,3) world MANO keypoints at pose q, given their link-local offsets."""
        with torch.no_grad():
            return self.fk.joints_from_dofs(q, local_offsets).cpu().numpy().astype(np.float64)

    def dofs(self, configs: Dict[str, np.ndarray], frames: Optional[int] = None) -> torch.Tensor:
        return self.fk.dofs_from_cfg(configs, frames=frames)


@lru_cache(maxsize=8)
def get_hand(side: str, device: str = "cpu", samples_per_link: int = 64,
             seed: int = 0) -> HandModel:
    """Cached hand model -- loading 16 STLs repeatedly is pure waste."""
    return HandModel.load(side, device=device, samples_per_link=samples_per_link, seed=seed)
