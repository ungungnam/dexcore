"""The articulated object: geometry, the hinge, and part-local <-> world transforms.

An ARCTIC object is two rigid parts joined by ONE revolute joint:

    bottom / base   part id 2, and the object ROOT frame -- its part-local frame IS the root frame
    top / lid       part id 1, rides the hinge at `obj_arti`

WHAT THIS CLASS IS FOR. Two things only, at this milestone:
  * `part_to_world` -- the one definition of "where is this part right now", used by the
    penetration metric and by any part-aware coordinate frame. Having a single entry point is how
    dexcore stays on the correct side of the raw-hinge-vs-calibrated-hinge distinction rather than
    each call site getting it right independently.
  * part meshes, for penetration and collision.

TWO CANONICAL-FRAME RULES, CHOSEN BY OBJECT CATEGORY, NOT BY A THRESHOLD. The +Z axis has to be
"the direction the lid opens away from", and how to read that off the geometry depends on what kind
of thing the object is:

    lid_hint      (default) the base OBB axis, perpendicular to the hinge, best aligned with the
                  base->lid direction. Right for BOXY objects, whose base is a tray with a clear
                  up direction and whose lid sits above it.
    base_normal   the THINNEST base OBB axis perpendicular to the hinge. Right for SLAB objects --
                  a laptop's base is a 2 cm panel, so its normal is unambiguous while the
                  base->lid hint is not: measured on a baked PartNet-Mobility laptop the hint runs
                  almost along the hinge itself, and `lid_hint` then picks the 30 cm axis instead
                  of the 1.7 cm one, leaving the canonical frame rotated 90 degrees.

The rule is DECLARED per object in `configs/canonical_rules.yaml`, not inferred. A threshold on
"how flat is flat" would be a guess about which family an object belongs to, and getting it wrong
silently rotates the frame every downstream stage is expressed in. Declaring it makes the choice
reviewable, and an object nobody declared keeps today's behaviour.

THE CANONICAL FRAME is what makes two different objects comparable at all. It is defined from the
object's own geometry, never from a hand-authored alignment:

    +X   the hinge line -- the one axis both objects genuinely share, since both are things that
         open about a hinge
    +Z   "up when closed" -- the BASE part's principal axis that is perpendicular to the hinge and
         points toward the lid. The base is used rather than the whole object because the lid
         swings, so an axis fitted to both parts would depend on the articulation the demo happened
         to be at
    +Y   completes a right-handed frame
    origin  the centre of the base's footprint at its BOTTOM (mean X, mean Y, min Z in canonical
            axes) -- where the object RESTS, so two objects aligned this way sit on the same
            surface. Aligning centroids instead floats a tall object and buries a flat one.

THE CANONICAL FRAME IS COMPUTED FROM THE VISUAL MESH, not the collision mesh. They are not the same
geometry: for the ARCTIC box the visual mesh has 176k vertices and the collision mesh 252, and their
centroids sit ~1 cm apart. The frame is a statement about the object's SHAPE, so it must come from
the shape; the collision mesh exists to make proximity queries cheap and is used for exactly that.
Computing the frame from the collision hull instead moved the transferred object 7.8 cm.

ARTICULATION IS COMPARED BY PHASE, not by angle. The box opens over [0, 2.6] rad and another object
may open over [-1.57, 1.57]; "half open" is the shared notion, so `q_to_alpha` / `alpha_to_q` map
through the normalised range. Copying the raw angle across would put a lid somewhere its own joint
cannot even reach.

SCALE. A URDF `scale` on the mesh multiplies every root-local coordinate INCLUDING THE HINGE. The
baked assets (box_s090, box_s110, pm100141_calibrated) have their scale already applied to their
vertices, so `calibration` is 1.0 for them. Passing a baked asset a non-unit calibration applies it
twice and produces a plausible-looking, wrong object.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from src import geometry as G
from src import paths
from src.paths import BASE_PART_ID, LID_PART_ID, PART_NAME


CANONICAL_RULES = ("lid_hint", "base_normal")
DEFAULT_CANONICAL_RULE = "lid_hint"


@lru_cache(maxsize=1)
def _canonical_rule_table() -> Dict[str, str]:
    """{object name -> rule} from configs/canonical_rules.yaml. Absent file -> empty table."""
    import yaml
    f = paths.REPO_ROOT / "configs" / "canonical_rules.yaml"
    if not f.exists():
        return {}
    table = yaml.safe_load(f.read_text()) or {}
    bad = {k: v for k, v in table.items() if v not in CANONICAL_RULES}
    if bad:
        raise ValueError(f"{f}: unknown canonical rule(s) {bad}. Known: {list(CANONICAL_RULES)}")
    return dict(table)


def canonical_rule_for(name: str) -> str:
    """The declared rule for `name`, or the default. See the module docstring on why not a threshold."""
    return _canonical_rule_table().get(str(name), DEFAULT_CANONICAL_RULE)


@dataclass
class Hinge:
    """The single revolute joint, in the parent (base/root) frame."""
    name: str
    parent: str
    child: str
    origin_t: np.ndarray            # (3,) joint origin in the parent frame
    origin_R: np.ndarray            # (3,3) joint frame orientation (rpy); identity for ARCTIC
    axis: np.ndarray                # (3,) unit axis in the joint frame
    q_min: float
    q_max: float

    def transform(self, q: float) -> np.ndarray:
        """Child-part-local -> parent frame at articulation `q`."""
        return (G.affine(self.origin_R, self.origin_t)
                @ G.affine(G.rodrigues(self.axis, float(q)), np.zeros(3)))


def _rpy_to_R(rpy) -> np.ndarray:
    r, p, y = (float(v) for v in rpy)
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp,     cp * sr,                cp * cr]])


def _vec(node, attr, default):
    if node is None or node.get(attr) is None:
        return np.asarray(default, dtype=np.float64)
    return np.fromstring(node.get(attr), sep=" ", dtype=np.float64)


@dataclass
class ArticulatedObject:
    """A two-part hinged object loaded from an ARCTIC-layout URDF directory."""

    name: str
    obj_dir: Path
    urdf_path: Path
    hinge: Hinge
    collision_mesh_files: Dict[int, list]     # part id -> convex-decomposition piece paths
    visual_mesh_files: Dict[int, Path]        # part id -> visual mesh path
    calibration: float = 1.0
    canonical_rule: str = DEFAULT_CANONICAL_RULE
    _meshes: dict = field(default_factory=dict, repr=False)
    _canon_R: Optional[np.ndarray] = field(default=None, repr=False)
    _canon_o: Optional[np.ndarray] = field(default=None, repr=False)

    # ------------------------------------------------------------------ construction
    @classmethod
    def load(cls, name: str, obj_dir=None, calibration: float = 1.0,
             canonical_rule: Optional[str] = None) -> "ArticulatedObject":
        """Load `assets/arctic/<name>/<name>.urdf` (or the single URDF in `obj_dir`)."""
        d = Path(obj_dir) if obj_dir is not None else paths.object_dir(name)
        if not d.exists():
            raise FileNotFoundError(
                f"object directory not found: {d}\nKnown objects: {', '.join(paths.list_objects())}")
        urdfs = sorted(d.glob("*.urdf"))
        if not urdfs:
            raise FileNotFoundError(f"no .urdf in {d}")
        urdf = d / f"{name}.urdf" if (d / f"{name}.urdf").exists() else urdfs[0]

        root = ET.parse(str(urdf)).getroot()
        links = {ln.get("name"): ln for ln in root.findall("link")}
        joints = [j for j in root.findall("joint") if j.get("type") in ("revolute", "continuous")]
        if len(joints) != 1:
            raise ValueError(f"{urdf}: expected exactly one revolute joint, found {len(joints)} "
                             f"-- dexcore models ARCTIC's two-part hinged objects only")
        j = joints[0]
        o, a, lim = j.find("origin"), j.find("axis"), j.find("limit")
        lo = float(lim.get("lower")) if lim is not None and lim.get("lower") else -np.pi
        hi = float(lim.get("upper")) if lim is not None and lim.get("upper") else np.pi
        hinge = Hinge(
            name=j.get("name"), parent=j.find("parent").get("link"), child=j.find("child").get("link"),
            origin_t=_vec(o, "xyz", [0, 0, 0]), origin_R=_rpy_to_R(_vec(o, "rpy", [0, 0, 0])),
            axis=G.normalize(_vec(a, "xyz", [1, 0, 0])), q_min=lo, q_max=hi)

        # part id 1 = the CHILD of the hinge (lid/moving), 2 = the parent (base/root)
        part_link = {LID_PART_ID: hinge.child, BASE_PART_ID: hinge.parent}
        coll, vis = {}, {}
        for pid, lname in part_link.items():
            ln = links.get(lname)
            if ln is None:
                raise ValueError(f"{urdf}: joint references missing link {lname!r}")
            # ALL collision meshes, not the first: a baked PartNet-Mobility part carries its
            # CONVEX DECOMPOSITION as many <collision> tags (pm100141's base has 212). Taking
            # `find()` -- the first -- silently substitutes one 8-vertex sliver for the object,
            # and every contact and penetration number computed against it is meaningless.
            coll[pid] = [d / e.get("filename") for e in ln.findall("collision/geometry/mesh")
                         if e.get("filename")]
            el = ln.find("visual/geometry/mesh")
            if el is not None and el.get("filename"):
                vis[pid] = d / el.get("filename")
        missing = [PART_NAME[p] for p in part_link if not coll.get(p)]
        if missing:
            raise ValueError(f"{urdf}: no collision mesh for part(s) {missing}")

        rule = canonical_rule or canonical_rule_for(name)
        if rule not in CANONICAL_RULES:
            raise ValueError(f"unknown canonical rule {rule!r}. Known: {list(CANONICAL_RULES)}")
        return cls(name=name, obj_dir=d, urdf_path=urdf, hinge=hinge,
                   collision_mesh_files=coll, visual_mesh_files=vis,
                   calibration=float(calibration), canonical_rule=rule)

    # ------------------------------------------------------------------ kinematics
    @property
    def q_closed(self) -> float:
        return self.hinge.q_min

    @property
    def q_open(self) -> float:
        return self.hinge.q_max

    def part_pose(self, part_id: int, arti: float) -> np.ndarray:
        """Part-local -> OBJECT ROOT frame at articulation `arti`.

        The base part IS the root frame, so its pose is the identity by construction. The lid rides
        the hinge, scaled by the calibration (a model scale moves the hinge too -- see the module
        docstring on why rotating a calibrated point about a raw hinge is the silent failure here).
        """
        if int(part_id) == BASE_PART_ID:
            return np.eye(4)
        if int(part_id) != LID_PART_ID:
            raise ValueError(f"unknown part id {part_id}; expected {LID_PART_ID} or {BASE_PART_ID}")
        h = self.hinge
        return (G.affine(h.origin_R, h.origin_t * self.calibration)
                @ G.affine(G.rodrigues(h.axis, float(arti)), np.zeros(3)))

    def part_to_world(self, part_id: int, arti: float, obj_pos, obj_quat) -> np.ndarray:
        """Part-local -> WORLD for one frame. The one definition every call site uses."""
        root = G.affine(G.wxyz_to_R(obj_quat), np.asarray(obj_pos, float))
        return root @ self.part_pose(part_id, arti)

    def part_to_world_batch(self, part_id: int, arti, obj_pos, obj_quat) -> np.ndarray:
        """(F,4,4) for a whole trajectory."""
        arti = np.asarray(arti, float).reshape(-1)
        return np.stack([self.part_to_world(part_id, arti[t], obj_pos[t], obj_quat[t])
                         for t in range(len(arti))])

    # ------------------------------------------------------------------ canonical frame
    @property
    def joint_axis(self) -> np.ndarray:
        """The hinge line as a unit vector in the OBJECT ROOT frame."""
        return G.normalize(self.hinge.origin_R @ self.hinge.axis)

    @property
    def hinge_point(self) -> np.ndarray:
        """A point on the hinge line, in the object root frame, at this calibration."""
        return self.hinge.origin_t * self.calibration

    def _part_centroid(self, part_id: int, arti: Optional[float] = None) -> np.ndarray:
        q = self.q_closed if arti is None else float(arti)
        v = self.part_vertices(part_id, collision=False)
        if int(part_id) == LID_PART_ID:
            v = G.transform_points(v, self.part_pose(LID_PART_ID, q))
        return v.mean(axis=0)

    def canonical_rotation(self) -> np.ndarray:
        """(3,3) canonical axes as columns, in the object root frame.

        +Z is the base part's ORIENTED-bounding-box axis that is perpendicular to the hinge and
        points toward the lid. The base->lid centroid vector alone tilts when the lid panel sits
        off-centre, so it is used only as the hint that picks WHICH box axis, and as the fallback
        when no OBB can be built.
        """
        if self._canon_R is None:
            x = self.joint_axis
            base_v = self.part_vertices(BASE_PART_ID, collision=False)
            hint = G.normalize(self._part_centroid(LID_PART_ID, self.q_closed) - base_v.mean(axis=0))
            up = hint
            try:
                obb = self.mesh(BASE_PART_ID, collision=False).bounding_box_oriented
                axes = np.asarray(obb.primitive.transform, dtype=np.float64)[:3, :3].T
                extents = np.asarray(obb.primitive.extents, dtype=np.float64)
                if self.canonical_rule == "base_normal":
                    # SLAB objects: the base's thinnest perpendicular axis IS its normal. The hint
                    # only picks the SIGN here -- on a slab it is unreliable as an axis (it can run
                    # nearly along the hinge) but its half-space is still right.
                    perp = [(e, a) for e, a in zip(extents, axes) if abs(float(a @ x)) < 0.5]
                    if perp:
                        _, axis = min(perp, key=lambda p: p[0])
                        up = G.normalize(axis if float(axis @ hint) >= 0 else -axis)
                else:
                    # BOXY objects: whichever perpendicular axis points most toward the lid.
                    cands = [a for a in np.vstack([axes, -axes]) if abs(float(a @ x)) < 0.5]
                    if cands:
                        up = G.normalize(max(cands, key=lambda a: float(a @ hint)))
            except Exception:
                pass
            self._canon_R = G.orthonormal_frame(x, up)
        return self._canon_R

    def canonical_origin(self) -> np.ndarray:
        """The canonical origin in the object root frame: the centre of the BASE's footprint at its
        BOTTOM -- (mean along +X, mean along +Y, min along +Z) in canonical axes, mapped back.

        NOT the base centroid. The bottom-footprint centre is where the object RESTS, so two objects
        aligned this way sit on the same surface; aligning centroids instead floats a tall object
        above the table and buries a flat one. Measured on box -> pm100141 the two differ by 7.8 cm,
        which is the whole hand.
        """
        if self._canon_o is None:
            R = self.canonical_rotation()
            c = self.part_vertices(BASE_PART_ID, collision=False) @ R   # into canonical axes
            self._canon_o = R @ np.array([c[:, 0].mean(), c[:, 1].mean(), c[:, 2].min()])
        return self._canon_o

    def canonical_pose(self) -> np.ndarray:
        """(4,4) canonical -> object root. Its inverse takes object-root points into canonical."""
        return G.affine(self.canonical_rotation(), self.canonical_origin())

    def to_canonical(self, pts_root) -> np.ndarray:
        return G.transform_points(pts_root, G.invert(self.canonical_pose()))

    def from_canonical(self, pts_canon) -> np.ndarray:
        return G.transform_points(pts_canon, self.canonical_pose())

    def canonical_extent(self) -> np.ndarray:
        """Per-axis extent of the closed object in its own canonical frame (m). The size two
        objects are compared by -- a shared diagonal says nothing if one is twice as tall."""
        v = [self.to_canonical(self.part_vertices(BASE_PART_ID, collision=False)),
             self.to_canonical(G.transform_points(
                 self.part_vertices(LID_PART_ID, collision=False),
                 self.part_pose(LID_PART_ID, self.q_closed)))]
        allv = np.vstack(v)
        return allv.max(0) - allv.min(0)

    # ------------------------------------------------------------------ articulation phase
    def q_to_alpha(self, q: float) -> float:
        """Joint angle -> normalised phase in [0,1], 0 = closed. Clamped, and that is deliberate:
        a source angle slightly outside its own recorded range should not drive the target past
        its limits."""
        span = self.q_open - self.q_closed
        if abs(span) < 1e-12:
            return 0.0
        return float(np.clip((float(q) - self.q_closed) / span, 0.0, 1.0))

    def alpha_to_q(self, alpha: float) -> float:
        """Normalised phase -> this object's own joint angle."""
        a = float(np.clip(alpha, 0.0, 1.0))
        return float(self.q_closed + a * (self.q_open - self.q_closed))

    # ------------------------------------------------------------------ geometry
    def mesh(self, part_id: int, collision: bool = False):
        """The part's surface in PART-LOCAL coordinates, calibration applied.

        `collision=False` (the default) gives the VISUAL mesh -- the object's actual surface, and
        what any question about shape, distance or rendering means. `collision=True` concatenates
        the convex-decomposition pieces, which is a different object: useful for a quick bound,
        but its signed distance is `convex_pieces` territory (see `signed_distance`).
        """
        import trimesh
        key = ("mesh", int(part_id), bool(collision))
        if key not in self._meshes:
            if collision:
                m = trimesh.util.concatenate(
                    [trimesh.load(str(f), process=False, force="mesh")
                     for f in self.collision_mesh_files[int(part_id)]])
            else:
                path = self.visual_mesh_files.get(int(part_id))
                if path is None:
                    raise KeyError(f"{self.name}: no visual mesh for part {part_id}")
                m = trimesh.load(str(path), process=False, force="mesh")
            if self.calibration != 1.0:
                m = m.copy(); m.apply_scale(self.calibration)
            self._meshes[key] = m
        return self._meshes[key]

    def part_vertices(self, part_id: int, collision: bool = False) -> np.ndarray:
        """(V,3) part-local vertices. Visual by default -- see `mesh`."""
        return np.asarray(self.mesh(part_id, collision).vertices, dtype=np.float64)

    def sample_part_surface(self, part_id: int, num_samples: int = 500,
                            seed: int = 0) -> np.ndarray:
        """(N,3) part-local samples of the VISUAL surface. Seeded, so a metric is reproducible."""
        import trimesh
        m = self.mesh(part_id, collision=False)
        pts, _ = trimesh.sample.sample_surface(m, num_samples, seed=int(seed))
        return np.asarray(pts, dtype=np.float64)

    def surface_index(self, part_id: int, num_samples: int = 20000, seed: int = 0):
        """cKDTree over dense VISUAL-surface samples, cached. Answers "how far to the surface"."""
        from scipy.spatial import cKDTree
        key = ("kdtree", int(part_id), int(num_samples), int(seed))
        if key not in self._meshes:
            self._meshes[key] = cKDTree(
                self.sample_part_surface(part_id, num_samples=num_samples, seed=seed))
        return self._meshes[key]

    def nearest_distance(self, part_id: int, pts_local) -> np.ndarray:
        """(N,) UNSIGNED distance to the part's visual surface. Accurate to the sampling density.

        This is the right question for CONTACT -- "does the link reach the surface" -- and the sign
        of a point 8 mm away changes nothing about the answer.
        """
        tree = self.surface_index(part_id)
        pts = np.asarray(pts_local, dtype=np.float64).reshape(-1, 3)
        d, _ = tree.query(pts, k=1, workers=-1)
        return np.asarray(d, dtype=np.float64)

    def signed_distance(self, part_id: int, pts_local,
                        cutoff: Optional[float] = 0.05) -> np.ndarray:
        """(N,) signed distance to the part's VISUAL surface, POSITIVE OUTSIDE.

        Sign convention is stated because trimesh's own `signed_distance` is positive INSIDE; it is
        negated here so "penetration depth" is `max(0, -d)` everywhere in dexcore.

        MEASURED AGAINST THE VISUAL MESH, not the collision decomposition. The decomposition is a
        simulator convenience and does not coincide with the surface: on pm100141 it sits up to
        12.8 mm off it, which is larger than the penetrations being measured. The visual mesh
        reproduces its own sampled surface to 0.00 mm.

        `cutoff` prunes: a point whose nearest sampled surface point is further than this is
        reported as that (positive) distance without an exact evaluation. Correct for anything
        outside, and wrong only for a point buried more than `cutoff` INSIDE a hollow part -- so
        `cutoff` must exceed the deepest penetration worth measuring. Pass None to evaluate all.
        """
        import trimesh
        m = self.mesh(part_id, collision=False)
        pts = np.asarray(pts_local, dtype=np.float64).reshape(-1, 3)
        if cutoff is None:
            return -np.asarray(trimesh.proximity.signed_distance(m, pts), dtype=np.float64)
        approx = self.nearest_distance(part_id, pts)
        near = approx <= float(cutoff)
        out = approx.copy()
        if np.any(near):
            out[near] = -np.asarray(
                trimesh.proximity.signed_distance(m, pts[near]), dtype=np.float64)
        return out

    def bbox_extent(self, arti: Optional[float] = None) -> np.ndarray:
        """Root-frame AABB extent (m) with the lid at `arti` (default: closed). Diagnostics only."""
        q = self.q_closed if arti is None else float(arti)
        v = [self.part_vertices(BASE_PART_ID),
             G.transform_points(self.part_vertices(LID_PART_ID), self.part_pose(LID_PART_ID, q))]
        allv = np.vstack(v)
        return allv.max(0) - allv.min(0)

    def summary(self) -> str:
        h = self.hinge
        ext = np.round(self.canonical_extent() * 100, 1)
        return (f"{self.name}  ({self.urdf_path.name}, calibration {self.calibration:g})\n"
                f"  canon   extent {ext} cm (x=hinge, z=up), "
                f"phase range [{self.q_closed:.3f}, {self.q_open:.3f}] rad\n"
                f"  hinge   {h.name}: {h.parent} -> {h.child}, axis {np.round(h.axis, 3)}, "
                f"range [{h.q_min:.3f}, {h.q_max:.3f}] rad\n"
                f"  extent  {np.round(self.bbox_extent() * 100, 1)} cm (closed)\n"
                f"  meshes  " + ", ".join(f"{PART_NAME[p]}={f.name}"
                                          for p, f in sorted(self.visual_mesh_files.items())))


@lru_cache(maxsize=16)
def get_object(name: str, calibration: float = 1.0,
               canonical_rule: Optional[str] = None) -> ArticulatedObject:
    """Cached loader -- parsing a URDF and reading meshes repeatedly is pure waste.

    `canonical_rule` is part of the cache key: two objects that differ only in their frame rule are
    different objects as far as everything downstream is concerned.
    """
    return ArticulatedObject.load(name, calibration=calibration, canonical_rule=canonical_rule)
