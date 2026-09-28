"""The demonstration container: one dense hand-object demo, in DexMachina's own format.

WHY THIS TYPE EXISTS. The research formulation factorizes a demonstration as

    D  ->  ( tau_O^{1:T} ,  K_HO )

a dense object-state trajectory plus sparse hand-object interaction states. `Demonstration` holds
BOTH, because they are two views of one file, and a "sparse" demo is just a `Demonstration` whose
frames are a subset -- `subset()` returns the same type. Keeping one type is deliberate: every
metric, every writer and every viewer then works unchanged on a source demo, a selected keyframe
set, a transferred keyframe set and a reconstructed dense demo.

THE FILE FORMAT IS DEXMACHINA'S, NOT OURS. `load()` and `save()` round-trip
`assets/arctic/processed/<subject>/<obj>_use_<clip>.npy` exactly, including the fields dexcore
never looks at (per-hand MANO trans/rot/shape). That is what makes the downstream adapter a
`shutil.copy` instead of a converter: a demo dexcore writes is a demo DexMachina trains on.

    params["obj_trans"]                  (F,3)     object ROOT position, metres, world
    params["obj_quat"]                   (F,4)     object ROOT orientation, wxyz
    params["obj_arti"]                   (F,)      the single revolute joint, radians
    world_coord["joints.{side}"]         (F,21,3)  MANO keypoints, world
    world_coord["contact_links_{side}"]  (F,16,4)  per LINK [x,y,z,part_id]; ZERO ROW = no contact
    world_coord["contacts.{side}"]       (F,50,4)  contact points ON THE OBJECT + part id
    world_coord["valid_contacts.{side}"] (F,50)    which slots are filled
    configs_{side}                       {name:(F,)}  the 51-DoF MANO URDF pose

TWO DECODING FACTS handled here so callers never have to:
  * `contact_links` is DENSE -- all 16 rows exist every frame and a non-touching link is an
    all-zero row. Contact is `norm(xyz) > 0`, never "row present".
  * `obj_quat` is a raw per-frame fit whose sign flips freely across the double cover. It is
    sign-unrolled on load, before anything differences or interpolates it.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, replace, asdict
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np

from src import geometry as G
from src import paths
from src.paths import BASE_PART_ID, DEFAULT_FPS, LID_PART_ID, SIDES

DEXCORE_FORMAT_VERSION = 1


# --------------------------------------------------------------------------------- provenance
@dataclass
class Provenance:
    """What produced this demonstration. Written into the .npy and into a sidecar .json.

    Every field is here because a result that cannot say what produced it cannot be compared
    against anything. `selected_frames` are LOCAL indices into the source window; `source_frames`
    are the corresponding absolute indices in the source file, so a keyframe stays actionable.
    """
    source_sequence: str = ""              # e.g. "box_use_01"
    source_object: str = ""                # e.g. "box"
    target_object: str = ""                # e.g. "box_s110" (== source for identity transfer)
    source_path: str = ""

    selector: str = ""
    selection_ratio: Optional[float] = None
    selected_frames: Optional[list] = None      # local indices into the source window
    source_frames: Optional[list] = None        # the same frames, absolute in the source file
    source_num_frames: Optional[int] = None     # dense length the selection came from

    transfer: str = ""
    reconstructor: str = ""

    seed: Optional[int] = None
    config: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)
    format_version: int = DEXCORE_FORMAT_VERSION

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d) -> "Provenance":
        if d is None:
            return cls()
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in dict(d).items() if k in known})

    def describe(self) -> str:
        r = f"{self.selection_ratio:.3g}" if self.selection_ratio is not None else "-"
        n = len(self.selected_frames) if self.selected_frames is not None else "-"
        return (f"{self.source_sequence} [{self.source_object} -> {self.target_object or '(same)'}]  "
                f"select={self.selector or '-'}(ratio {r}, n={n})  "
                f"transfer={self.transfer or '-'}  recon={self.reconstructor or '-'}")


# ------------------------------------------------------------------------------ demonstration
@dataclass
class Demonstration:
    """One demo (dense or sparse), sliced to a frame window, contact arrays already decoded."""

    name: str
    obj_name: str
    use_clip: str
    subject: str
    fps: float

    frame_start: int                    # absolute index of local frame 0 in the source file
    frame_end: int                      # absolute, half-open
    total_frames: int                   # frames in the file before slicing

    obj_pos: np.ndarray                 # (F,3) metres, world
    obj_quat: np.ndarray                # (F,4) wxyz, sign-unrolled
    obj_arti: np.ndarray                # (F,) radians

    joints: Dict[str, np.ndarray]       # side -> (F,21,3) world MANO keypoints
    contact_pos: Dict[str, np.ndarray]  # side -> (F,16,3), zero row == not in contact
    contact_part: Dict[str, np.ndarray] # side -> (F,16) int, 0 none / 1 top / 2 bottom
    contact_mask: Dict[str, np.ndarray] # side -> (F,16) bool

    configs: Dict[str, Dict[str, np.ndarray]] = field(default_factory=dict)  # side -> {joint:(F,)}
    obj_contacts: Dict[str, np.ndarray] = field(default_factory=dict)        # side -> (F,50,4)
    obj_contact_valid: Dict[str, np.ndarray] = field(default_factory=dict)   # side -> (F,50)
    contact_threshold: float = 0.01

    path: str = ""
    provenance: Provenance = field(default_factory=Provenance)
    # every `params` entry not modelled above, sliced alongside. Preserved verbatim on save so a
    # round trip is lossless -- dexcore not reading a field is no reason to destroy it.
    extra_params: Dict[str, np.ndarray] = field(default_factory=dict)
    # absolute source frame of each row. For a dense demo this is just arange(frame_start, ...);
    # for a SPARSE demo it is the selected frames, and it is the only record of which they were.
    frame_index: Optional[np.ndarray] = None

    # ------------------------------------------------------------------ basic properties
    def __len__(self) -> int:
        return int(len(self.obj_arti))

    @property
    def num_frames(self) -> int:
        return len(self)

    @property
    def is_sparse(self) -> bool:
        """True when this demo's rows are a strict subset of the window they were selected from."""
        n = self.provenance.source_num_frames
        return n is not None and self.num_frames < n

    def frames(self) -> np.ndarray:
        """Absolute source frame index of every row."""
        if self.frame_index is not None:
            return np.asarray(self.frame_index, dtype=int)
        return np.arange(self.frame_start, self.frame_start + self.num_frames)

    def local_frames(self) -> np.ndarray:
        """Row index within the source WINDOW (0-based). Equals arange(F) for a dense demo."""
        return self.frames() - self.frame_start

    def times(self) -> np.ndarray:
        """Seconds from the start of the window, using the true frame indices."""
        return self.local_frames() / self.fps

    def to_absolute(self, local) -> int:
        return int(local) + self.frame_start

    # ------------------------------------------------------------------ object frame
    def object_transform(self, t=None) -> np.ndarray:
        """Object ROOT pose T_O(t) as (4,4), or (F,4,4) for `t=None`.

        NOTE, and this is the assumption the linear reconstructor rests on: this is the ROOT
        (base-part) frame. The object is ARTICULATED, so a point riding the lid is not static in
        this frame even when the grasp is unchanged. `obj_arti` is carried and interpolated
        separately; see `ObjectFrame` in src/reconstruction/frames.py for the part-aware variant.
        """
        if t is None:
            return G.affine(G.wxyz_to_R(self.obj_quat), self.obj_pos)
        return G.affine(G.wxyz_to_R(self.obj_quat[t]), self.obj_pos[t])

    def world_to_object(self, pts_world, t=None) -> np.ndarray:
        """World -> object-root-relative. `pts_world` is (F,N,3) with t=None, else (N,3)."""
        return G.transform_points(pts_world, G.invert(self.object_transform(t)))

    def object_to_world(self, pts_obj, t=None) -> np.ndarray:
        """Object-root-relative -> world. Exact inverse of `world_to_object` at the same t."""
        return G.transform_points(pts_obj, self.object_transform(t))

    def object_state(self) -> np.ndarray:
        """(F,8) [pos(3), quat wxyz(4), arti(1)] -- the layout DexMachina's ADD evaluation reads."""
        return np.concatenate(
            [self.obj_pos, self.obj_quat, self.obj_arti[:, None]], axis=1).astype(np.float64)

    # ------------------------------------------------------------------ subsetting
    def subset(self, rows: Sequence[int], selector: str = "", ratio: Optional[float] = None,
               **prov) -> "Demonstration":
        """Keep only `rows` (indices into THIS demo), recording them in the provenance.

        The result is a `Demonstration` of the same shape-contract with fewer frames -- this is
        what `Selector.select` returns. `source_num_frames` is stamped so a reconstructor can tell
        how long the dense output must be without being told separately.
        """
        idx = np.asarray(sorted(set(int(i) for i in rows)), dtype=int)
        if len(idx) == 0:
            raise ValueError("subset with no rows")
        if idx.min() < 0 or idx.max() >= self.num_frames:
            raise IndexError(f"rows out of range [0,{self.num_frames}): "
                             f"min {idx.min()}, max {idx.max()}")
        src_frames = self.frames()[idx]
        p = replace(
            self.provenance,
            selector=selector or self.provenance.selector,
            selection_ratio=ratio if ratio is not None else self.provenance.selection_ratio,
            selected_frames=[int(i) for i in idx],
            source_frames=[int(f) for f in src_frames],
            source_num_frames=int(self.provenance.source_num_frames or self.num_frames),
            **prov,
        )
        return replace(
            self,
            obj_pos=self.obj_pos[idx], obj_quat=self.obj_quat[idx], obj_arti=self.obj_arti[idx],
            joints={s: v[idx] for s, v in self.joints.items()},
            contact_pos={s: v[idx] for s, v in self.contact_pos.items()},
            contact_part={s: v[idx] for s, v in self.contact_part.items()},
            contact_mask={s: v[idx] for s, v in self.contact_mask.items()},
            configs={s: {k: v[idx] for k, v in cfg.items()} for s, cfg in self.configs.items()},
            obj_contacts={s: v[idx] for s, v in self.obj_contacts.items()},
            obj_contact_valid={s: v[idx] for s, v in self.obj_contact_valid.items()},
            extra_params={k: v[idx] for k, v in self.extra_params.items()},
            frame_index=src_frames,
            provenance=p,
        )

    # ------------------------------------------------------------------ IO
    @classmethod
    def load(cls, path=None, obj_name: str = "box", use_clip: str = "01", subject: str = "s01",
             frame_start: Optional[int] = None, frame_end: Optional[int] = None,
             fps: float = DEFAULT_FPS) -> "Demonstration":
        """Load a processed demo and slice it to [frame_start, frame_end).

        `frame_start=None` means "from the beginning of the file"; `frame_end=None` means "to the
        end". The window is applied to every array together, so local index t always means the same
        instant everywhere.

        A file dexcore wrote carries `dexcore_demo` -- its object, clip and the SOURCE FRAMES it
        covers. That is preferred over parsing the filename, so a demo generated from frames 30-230
        still knows it is frames 30-230, and a sparse keyframe set still knows its object. A raw
        ARCTIC file has no such block and falls back to the filename convention.
        """
        p = paths.resolve_demo(path, obj_name, use_clip, subject)
        raw = np.load(p, allow_pickle=True).item()
        for key in ("params", "world_coord"):
            if key not in raw:
                raise ValueError(f"{p} is not a processed demo (missing {key!r}); "
                                 f"got keys {sorted(raw)}")
        params, world = raw["params"], raw["world_coord"]

        n_file = len(np.asarray(params["obj_arti"]))
        req_start = 0 if frame_start is None else int(max(0, frame_start))
        req_end = n_file if frame_end is None else int(min(frame_end, n_file))
        if req_end - req_start < 2:
            raise ValueError(
                f"window [{req_start},{req_end}) has <2 frames; this file holds {n_file}")
        sl = slice(req_start, req_end)

        stem = p.stem
        parsed_obj, parsed_clip = paths.parse_demo_stem(stem)

        # the self-describing block, when dexcore wrote this file
        meta = raw.get("dexcore_demo") or {}
        if meta:
            parsed_obj = meta.get("obj_name", parsed_obj)
            parsed_clip = meta.get("use_clip", parsed_clip)
            subject = meta.get("subject", subject)
            if frame_start is None and frame_end is None:
                fps = float(meta.get("fps", fps))
        idx_all = np.asarray(meta.get("frame_index", np.arange(n_file)), dtype=int)
        frame_index = idx_all[sl] if len(idx_all) == n_file else None
        # absolute frame of local row 0, and the length of the clip this was cut from
        frame_start = int(frame_index[0]) if frame_index is not None else req_start
        total = int(meta.get("total_frames", n_file))
        frame_end = frame_start + (req_end - req_start)

        obj_arti = np.asarray(params["obj_arti"], dtype=np.float64)[sl]
        if obj_arti.ndim > 1:                       # some generated demos carry (F,1)
            obj_arti = obj_arti[:, 0]

        joints, cpos, cpart, cmask = {}, {}, {}, {}
        ocontacts, ovalid = {}, {}
        for side in SIDES:
            joints[side] = np.asarray(world[f"joints.{side}"], dtype=np.float64)[sl]
            links = np.asarray(world[f"contact_links_{side}"], dtype=np.float64)[sl]
            pos, part = links[..., :3], links[..., 3]
            mask = np.linalg.norm(pos, axis=-1) > 0      # zero row == no contact; see docstring
            cpos[side] = pos
            cmask[side] = mask
            cpart[side] = np.where(mask, np.rint(part), 0).astype(np.int8)
            if f"contacts.{side}" in world:
                ocontacts[side] = np.asarray(world[f"contacts.{side}"], dtype=np.float64)[sl]
                ovalid[side] = np.asarray(world[f"valid_contacts.{side}"], dtype=bool)[sl]

        modelled = {"obj_trans", "obj_quat", "obj_arti"}
        extra = {k: np.asarray(v)[sl] for k, v in params.items()
                 if k not in modelled and np.asarray(v).shape[:1] == (total,)}

        prov = Provenance.from_dict(raw.get("dexcore_provenance"))
        if not prov.source_sequence:
            prov.source_sequence = stem
        if not prov.source_object:
            prov.source_object = parsed_obj
        if not prov.source_path:
            prov.source_path = str(p)

        return cls(
            name=stem,
            # taken from the FILENAME, not the arguments: with an explicit path the arguments were
            # never used, and without one the filename was built from them -- right either way.
            obj_name=parsed_obj,
            use_clip=parsed_clip if parsed_clip is not None else use_clip,
            subject=subject,
            fps=float(fps),
            frame_start=frame_start, frame_end=frame_end, total_frames=int(total),
            obj_pos=np.asarray(params["obj_trans"], dtype=np.float64)[sl],
            obj_quat=G.unroll_quat(np.asarray(params["obj_quat"], dtype=np.float64)[sl]),
            obj_arti=obj_arti,
            joints=joints, contact_pos=cpos, contact_part=cpart, contact_mask=cmask,
            configs={s: {k: np.asarray(v, dtype=np.float64)[sl]
                         for k, v in raw.get(f"configs_{s}", {}).items()} for s in SIDES},
            obj_contacts=ocontacts, obj_contact_valid=ovalid,
            contact_threshold=float(np.asarray(world.get("contact_threshold", 0.01))),
            path=str(p), provenance=prov, extra_params=extra,
            frame_index=frame_index,
        )

    def to_raw(self) -> dict:
        """Rebuild the DexMachina on-disk dict. The inverse of what `load` decodes."""
        F = self.num_frames
        world = {"contact_threshold": np.float64(self.contact_threshold)}
        for side in SIDES:
            world[f"joints.{side}"] = self.joints[side].astype(np.float32)
            links = np.zeros((F, 16, 4), dtype=np.float64)
            links[..., :3] = self.contact_pos[side]
            links[..., 3] = self.contact_part[side]
            # re-zero non-contact rows so the "zero row == no contact" invariant survives a save
            links[~self.contact_mask[side]] = 0.0
            world[f"contact_links_{side}"] = links
            if side in self.obj_contacts:
                world[f"contacts.{side}"] = self.obj_contacts[side]
                world[f"valid_contacts.{side}"] = self.obj_contact_valid[side]

        params = {"obj_trans": self.obj_pos.astype(np.float32),
                  "obj_quat": self.obj_quat.astype(np.float32),
                  "obj_arti": self.obj_arti.astype(np.float32)}
        params.update({k: v for k, v in self.extra_params.items()})

        raw = {"world_coord": world, "params": params,
               "dexcore_provenance": self.provenance.to_dict(),
               # WHAT THIS FILE IS, written into the file rather than left to its NAME. Without it
               # a saved demo forgets which source frames it covers (a clip generated from frames
               # 30-230 reloads as 0-199) and a sparse keyframe set cannot say which object it is
               # for, because `selected_keyframes.npy` has no obj_name to parse out of the stem.
               "dexcore_demo": {
                   "obj_name": self.obj_name, "use_clip": self.use_clip,
                   "subject": self.subject, "fps": float(self.fps),
                   "frame_start": int(self.frame_start), "frame_end": int(self.frame_end),
                   "total_frames": int(self.total_frames),
                   "frame_index": [int(f) for f in self.frames()],
                   "format_version": DEXCORE_FORMAT_VERSION,
               }}
        for side in SIDES:
            raw[f"configs_{side}"] = {k: np.asarray(v, dtype=np.float64)
                                      for k, v in self.configs.get(side, {}).items()}
        return raw

    def save(self, path, write_sidecar: bool = True) -> Path:
        """Write a DexMachina-readable .npy (+ a human-readable provenance sidecar)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, self.to_raw(), allow_pickle=True)
        if write_sidecar:
            meta = self.provenance.to_dict()
            meta.update(num_frames=self.num_frames, fps=self.fps, obj_name=self.obj_name,
                        use_clip=self.use_clip, frame_start=self.frame_start,
                        frame_end=self.frame_end, frame_index=[int(f) for f in self.frames()])
            path.with_suffix(".json").write_text(json.dumps(meta, indent=2))
        return path

    # ------------------------------------------------------------------ reporting
    def summary(self) -> str:
        n_contact = {s: int(self.contact_mask[s].any(axis=1).sum()) for s in SIDES}
        kind = "sparse" if self.is_sparse else "dense"
        span = np.round((self.obj_pos.max(0) - self.obj_pos.min(0)) * 100, 1)
        return (
            f"{self.name}  [{kind}]  {self.num_frames} frames "
            f"(window {self.frame_start}-{self.frame_end} of {self.total_frames} @ {self.fps:g} fps)\n"
            f"  object    {self.obj_name}\n"
            f"  arti      {np.degrees(self.obj_arti.min()):7.2f} .. "
            f"{np.degrees(self.obj_arti.max()):7.2f} deg\n"
            f"  obj pos   span {span} cm\n"
            f"  contact   frames with >=1 link: left {n_contact['left']}, right {n_contact['right']}\n"
            f"  prov      {self.provenance.describe()}"
        )
