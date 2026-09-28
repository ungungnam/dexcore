"""The MANO hand: link/joint conventions, and batched differentiable forward kinematics.

Vendored and refactored from dexmachina exp3 (`mano_fk.py`, and the conventions in
`retargeting/process_arctic.py` / `exp4 .../reference.py`). It is vendored rather than imported so
dexcore does not depend on exp3's internals, but the numbers must stay identical -- a demo written
by dexcore is read by DexMachina, so any disagreement here is a silent corruption of the reference.

THREE CONVENTIONS, all of which are file-format facts rather than choices:

  21 KEYPOINTS      the demo's `world_coord["joints.{side}"]` are ARCTIC's MANO joints. Index 0 is
                    the wrist; the rest follow MANO's ordering (see `JOINT_NAMES`).
  16 LINKS          `world_coord["contact_links_{side}"]` has one row per MANO *link* in the slot
                    order `MANO_HAND_LINKS` gives. A link that touches nothing is an ALL-ZERO row,
                    so "in contact" is `norm(xyz) > 0`, not "row present".
  51 DOF            the URDF hand: 3 prismatic wrist + 48 revolute (3 wrist rpy + 15 segments x 3).
                    `configs_{side}` in the demo file maps joint name -> (F,) angles.

THE JOINTS ARE NOT THE LINK ORIGINS. The demo's joints come from ARCTIC's MANO fit; the URDF is a
generic hand, and the two sit ~13mm apart at every joint. Anything that wants to move a keypoint by
moving the hand must therefore carry each joint at its OWN fixed offset in its owner link's frame
(`joint_owner_links`, `joint_local_offsets`), never treat a joint as a link origin. Getting this
wrong shifts every reconstructed keypoint by a constant ~13mm, which looks like a plausible error
rather than a bug.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import Dict, List, Sequence

import numpy as np
import torch

# The 16 MANO links a contact can be assigned to, in the slot order process_arctic.py writes.
# Each entry is (link name, the MANO joint indices that link spans); the joint indices double as
# the skeleton a viewer draws.
MANO_HAND_LINKS = [
    ("palm", [0, 13, 1, 4, 10, 7]),
    ("thumb1", [13, 14]), ("thumb2", [14, 15]), ("thumb3", [15, 16]),
    ("index1", [1, 2]), ("index2", [2, 3]), ("index3", [3, 17]),
    ("middle1", [4, 5]), ("middle2", [5, 6]), ("middle3", [6, 18]),
    ("ring1", [10, 11]), ("ring2", [11, 12]), ("ring3", [12, 19]),
    ("pinky1", [7, 8]), ("pinky2", [8, 9]), ("pinky3", [9, 20]),
]
LINK_NAMES: List[str] = [n for n, _ in MANO_HAND_LINKS]
NUM_LINKS = len(LINK_NAMES)
NUM_JOINTS = 21

# MANO joint ordering, for readable diagnostics only -- indices are what the arrays use.
# The TIP order is read off MANO_HAND_LINKS above (each `*3` link spans [dip, tip]), not guessed:
# thumb3 spans [15,16], index3 [3,17], middle3 [6,18], ring3 [12,19], pinky3 [9,20]. Verified
# geometrically against the demo -- each tip's nearest distal link origin is the one named here.
JOINT_NAMES = [
    "wrist",
    "index_mcp", "index_pip", "index_dip",
    "middle_mcp", "middle_pip", "middle_dip",
    "pinky_mcp", "pinky_pip", "pinky_dip",
    "ring_mcp", "ring_pip", "ring_dip",
    "thumb_mcp", "thumb_pip", "thumb_dip",
    "thumb_tip", "index_tip", "middle_tip", "ring_tip", "pinky_tip",
]
THUMB_TIP_JOINT = 16
# The five fingertips, for fingertip-only reconstruction error.
FINGERTIP_JOINTS = [16, 17, 18, 19, 20]

# The 6 wrist DOFs of the URDF hand, separated from the 45 finger DOFs everywhere they are
# weighted differently (the wrist MUST travel during a transfer; the fingers should only give way).
WRIST_DOFS = ["palm_x_joint", "palm_y_joint", "palm_z_joint",
              "palm_yaw_joint", "palm_pitch_joint", "palm_roll_joint"]


def joint_owner_links() -> Dict[int, str]:
    """{joint index 0..20 -> the URDF link it is rigidly attached to}, derived from MANO_HAND_LINKS.

    `idxs[0]` is the joint at that link's origin. For a distal segment (thumb3/index3/...) `idxs[1]`
    is the fingertip, which has no link of its own and rides the distal link.
    """
    owner: Dict[int, str] = {}
    for name, idxs in MANO_HAND_LINKS:
        if name == "palm":
            owner[idxs[0]] = name          # joint 0 = wrist; the finger bases belong to the *1 links
            continue
        owner[idxs[0]] = name
        if name.endswith("3"):
            owner[idxs[1]] = name          # fingertip rides the distal segment
    assert sorted(owner) == list(range(NUM_JOINTS)), f"joint owners incomplete: {sorted(owner)}"
    return owner


JOINT_OWNER = joint_owner_links()
JOINT_OWNER_LINK_IDX = np.array([LINK_NAMES.index(JOINT_OWNER[j]) for j in range(NUM_JOINTS)])


def parse_urdf_chain(urdf_path):
    """URDF -> (joints, root_link), joints in topological order (URDF order is already valid).

    Each joint: dict(name, type, parent, child, origin(3,), axis(3,), limit(2,) or None).
    """
    root = ET.parse(str(urdf_path)).getroot()
    joints = []
    for j in root.findall("joint"):
        o, a, l = j.find("origin"), j.find("axis"), j.find("limit")
        joints.append(dict(
            name=j.get("name"), type=j.get("type"),
            parent=j.find("parent").get("link"), child=j.find("child").get("link"),
            origin=np.fromstring(o.get("xyz"), sep=" ") if o is not None else np.zeros(3),
            axis=(np.fromstring(a.get("xyz"), sep=" ") if a is not None else np.zeros(3)),
            limit=(np.array([float(l.get("lower")), float(l.get("upper"))])
                   if l is not None and l.get("lower") is not None else None),
        ))
    children = {j["child"] for j in joints}
    roots = [ln.get("name") for ln in root.findall("link") if ln.get("name") not in children]
    assert len(roots) == 1, f"expected one root link, got {roots}"
    return joints, roots[0]


class ManoFK:
    """Batched, differentiable FK for the MANO hand URDF. q is (B, ndof); returns (B, n_query,4,4).

    FK is defined on the RAW URDF tree, deliberately including links a simulator would merge away
    (Genesis folds `palm` into `palm_fixed_link`, which silently drops a quarter of the hand's
    contact surface). Owning the tree keeps `palm` addressable.
    """

    def __init__(self, urdf_path, query_links: Sequence[str] = tuple(LINK_NAMES),
                 device="cpu", dtype=torch.float32):
        self.urdf_path = str(urdf_path)
        self.joints, self.root = parse_urdf_chain(urdf_path)
        self.device, self.dtype = torch.device(device), dtype
        self.dof_names = [j["name"] for j in self.joints if j["type"] in ("revolute", "prismatic")]
        self.dof_index = {n: i for i, n in enumerate(self.dof_names)}
        self.query_links = list(query_links)

        lim = np.stack([j["limit"] if j["limit"] is not None else np.array([-np.inf, np.inf])
                        for j in self.joints if j["type"] in ("revolute", "prismatic")])
        self.limits = torch.tensor(lim, device=self.device, dtype=dtype)      # (ndof, 2)
        self._eye3 = torch.eye(3, device=self.device, dtype=dtype)
        self._eye4 = torch.eye(4, device=self.device, dtype=dtype)
        self._bottom = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=self.device, dtype=dtype)
        for j in self.joints:
            j["_origin_t"] = torch.tensor(j["origin"], device=self.device, dtype=dtype)
            n = np.linalg.norm(j["axis"])
            j["_axis_t"] = torch.tensor(j["axis"] / n if n > 1e-9 else j["axis"],
                                        device=self.device, dtype=dtype)
            if j["type"] == "revolute":
                # Rodrigues K and K@K depend only on the constant joint axis, so build them once.
                a = j["axis"] / n if n > 1e-9 else j["axis"]
                K = np.array([[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]])
                j["_K"] = torch.tensor(K, device=self.device, dtype=dtype)
                j["_KK"] = j["_K"] @ j["_K"]
            elif j["type"] != "prismatic":
                Tf = np.eye(4)
                Tf[:3, 3] = j["origin"]
                j["_fixed_T"] = torch.tensor(Tf, device=self.device, dtype=dtype)

        self.wrist_dof_idx = [self.dof_index[n] for n in WRIST_DOFS if n in self.dof_index]
        self.finger_dof_idx = [i for i in range(self.ndof) if i not in set(self.wrist_dof_idx)]

    @property
    def ndof(self) -> int:
        return len(self.dof_names)

    def dofs_from_cfg(self, cfg: Dict[str, np.ndarray], frames: int = None) -> torch.Tensor:
        """`configs_{side}` dict {joint name: (F,)} -> q (F, ndof). Missing joints default to 0."""
        if not cfg:
            raise ValueError("empty configs dict: this demo carries no MANO joint angles")
        F = len(next(iter(cfg.values()))) if frames is None else frames
        q = np.zeros((F, self.ndof))
        for name, vals in cfg.items():
            if name in self.dof_index:
                q[:, self.dof_index[name]] = np.asarray(vals, float)[:F]
        return torch.tensor(q, device=self.device, dtype=self.dtype)

    def cfg_from_dofs(self, q: torch.Tensor, cfg_like: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Inverse of `dofs_from_cfg`: write q back into a `configs_{side}`-shaped dict."""
        qn = q.detach().cpu().numpy()
        return {name: (qn[:, self.dof_index[name]].copy() if name in self.dof_index
                       else np.asarray(vals, float).copy())
                for name, vals in cfg_like.items()}

    def _joint_transform(self, j, q):
        """(B,4,4) local transform of joint j at angle/offset q (B,).

        Built by `cat`, NOT by writing into a cloned identity. The in-place version is ~100x slower
        in BACKWARD (measured 0.63 s/iter vs 0.006 s forward): each indexed write becomes its own
        autograd node, and there are 53 joints per FK call. Forward cost is identical; this is
        purely about the graph IK differentiates through.
        """
        B = q.shape[0]
        if j["type"] == "prismatic":
            R = self._eye3.expand(B, 3, 3)
            t = j["_origin_t"] + j["_axis_t"] * q[:, None]              # (B,3)
        else:                                                            # revolute
            s, c = torch.sin(q)[:, None, None], torch.cos(q)[:, None, None]
            R = self._eye3 + s * j["_K"] + (1 - c) * j["_KK"]            # (B,3,3)
            t = j["_origin_t"].expand(B, 3)
        return torch.cat([torch.cat([R, t[:, :, None]], dim=2),
                          self._bottom.expand(B, 1, 4)], dim=1)

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        """q (B, ndof) -> (B, len(query_links), 4, 4) world poses of the queried links."""
        B = q.shape[0]
        pose = {self.root: self._eye4.expand(B, 4, 4)}
        for j in self.joints:
            if j["name"] not in self.dof_index:              # fixed joint: constant transform
                pose[j["child"]] = pose[j["parent"]] @ j["_fixed_T"].expand(B, 4, 4)
                continue
            pose[j["child"]] = pose[j["parent"]] @ self._joint_transform(
                j, q[:, self.dof_index[j["name"]]])
        missing = [l for l in self.query_links if l not in pose]
        assert not missing, f"links not in URDF tree: {missing}"
        return torch.stack([pose[l] for l in self.query_links], dim=1)

    # ---------------------------------------------------------------- keypoints <-> hand pose
    def joint_local_offsets(self, q: torch.Tensor, joints_world: np.ndarray) -> torch.Tensor:
        """The 21 keypoints expressed in their OWNER LINK's frame, at hand pose `q` -> (F,21,3).

        This is the only correct way to relate the demo's MANO keypoints to the URDF hand: the
        offset is whatever the demo says it is, and is then carried rigidly by the link. See the
        module docstring on why the keypoints are not the link origins.
        """
        T = self.forward(q)[:, JOINT_OWNER_LINK_IDX]                    # (F,21,4,4)
        p = torch.as_tensor(joints_world, device=self.device, dtype=self.dtype)
        R, t = T[:, :, :3, :3], T[:, :, :3, 3]
        return torch.einsum("fnji,fnj->fni", R, p - t)                  # R^T (p - t)

    def joints_from_dofs(self, q: torch.Tensor, local: torch.Tensor) -> torch.Tensor:
        """Place keypoints with fixed link-local offsets `local` (F,21,3) at hand pose q -> (F,21,3).

        Exact inverse of `joint_local_offsets` at the same q, so a round trip is the identity and a
        pure wrist translation reduces to `joints + d` -- i.e. this cannot silently redefine the
        keypoints relative to the existing demos, it only applies whatever motion the pose found.
        """
        T = self.forward(q)[:, JOINT_OWNER_LINK_IDX]
        return torch.einsum("fnij,fnj->fni", T[:, :, :3, :3], local) + T[:, :, :3, 3]

    def link_frames(self, q: torch.Tensor) -> torch.Tensor:
        """(F, 16, 4, 4) world pose of every contact link -- the diagnostic view of a hand pose."""
        return self.forward(q)


_FK_CACHE: Dict[tuple, ManoFK] = {}


def get_fk(side: str, device="cpu", dtype=torch.float32, urdf_path=None) -> ManoFK:
    """Cached ManoFK per (side, device, dtype). Parsing the URDF twice is pure waste."""
    from src import paths
    path = str(urdf_path or paths.mano_urdf(side))
    key = (path, str(device), str(dtype))
    if key not in _FK_CACHE:
        _FK_CACHE[key] = ManoFK(path, LINK_NAMES, device=device, dtype=dtype)
    return _FK_CACHE[key]


# ------------------------------------------------------------------ the wrist as a rigid pose
# The URDF's six wrist joints all sit at the origin of their parent, in this order:
#     palm_x, palm_y, palm_z   prismatic along X, Y, Z
#     palm_yaw, palm_pitch, palm_roll   revolute about Z, Y, X
# so the wrist transform is exactly  Trans(x,y,z) @ Rz(yaw) @ Ry(pitch) @ Rx(roll)  -- a pure ZYX
# intrinsic Euler rotation with a free translation. That is what makes it invertible in closed form,
# which a transfer needs: placing the palm produces a rigid TRANSFORM, and the hand is parameterised
# by ANGLES, so the two have to be convertible without an optimiser.
WRIST_TRANSLATION_DOFS = ["palm_x_joint", "palm_y_joint", "palm_z_joint"]
WRIST_ROTATION_DOFS = ["palm_yaw_joint", "palm_pitch_joint", "palm_roll_joint"]


def wrist_matrix(x, y, z, yaw, pitch, roll) -> np.ndarray:
    """The six wrist DOFs -> (4,4). Batches over leading dimensions."""
    x, y, z = np.asarray(x, float), np.asarray(y, float), np.asarray(z, float)
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)
    R = np.empty(np.shape(x) + (3, 3))
    R[..., 0, 0] = cy * cp
    R[..., 0, 1] = cy * sp * sr - sy * cr
    R[..., 0, 2] = cy * sp * cr + sy * sr
    R[..., 1, 0] = sy * cp
    R[..., 1, 1] = sy * sp * sr + cy * cr
    R[..., 1, 2] = sy * sp * cr - cy * sr
    R[..., 2, 0] = -sp
    R[..., 2, 1] = cp * sr
    R[..., 2, 2] = cp * cr
    T = np.zeros(np.shape(x) + (4, 4))
    T[..., :3, :3] = R
    T[..., 0, 3], T[..., 1, 3], T[..., 2, 3] = x, y, z
    T[..., 3, 3] = 1.0
    return T


def wrist_dofs_from_matrix(T):
    """(4,4) or (...,4,4) -> (x, y, z, yaw, pitch, roll), the ZYX decomposition.

    GIMBAL LOCK is reported, not silently resolved: at pitch = +-90 degrees yaw and roll are not
    separable, and any convention picked there is a choice that would move the hand. The MANO wrist
    never reaches it in practice, so hitting it means something upstream is wrong.
    """
    T = np.asarray(T, dtype=np.float64)
    R = T[..., :3, :3]
    sp = np.clip(-R[..., 2, 0], -1.0, 1.0)
    pitch = np.arcsin(sp)
    if np.any(np.abs(np.abs(sp) - 1.0) < 1e-8):
        raise ValueError(
            "wrist decomposition hit gimbal lock (pitch = +-90 deg), where yaw and roll are not "
            "separable. Any convention chosen here would move the hand; refusing instead.")
    yaw = np.arctan2(R[..., 1, 0], R[..., 0, 0])
    roll = np.arctan2(R[..., 2, 1], R[..., 2, 2])
    return (T[..., 0, 3], T[..., 1, 3], T[..., 2, 3], yaw, pitch, roll)


class _WristMixin:
    pass


def wrist_pose_from_q(fk: ManoFK, q) -> np.ndarray:
    """(F, ndof) -> (F,4,4) world pose of the wrist frame."""
    q = q.detach().cpu().numpy() if hasattr(q, "detach") else np.asarray(q, float)
    idx = [fk.dof_index[n] for n in WRIST_TRANSLATION_DOFS + WRIST_ROTATION_DOFS]
    v = q[:, idx]
    return wrist_matrix(v[:, 0], v[:, 1], v[:, 2], v[:, 3], v[:, 4], v[:, 5])


def apply_wrist_transform(fk: ManoFK, q, T_delta) -> np.ndarray:
    """Left-multiply each frame's wrist pose by `T_delta` -> a new (F, ndof) q. Fingers untouched.

    This is what "place the palm" means mechanically: the whole hand is carried rigidly, and every
    finger joint keeps exactly the angle the human demonstrated.
    """
    q = q.detach().cpu().numpy().copy() if hasattr(q, "detach") else np.array(q, dtype=np.float64)
    T_old = wrist_pose_from_q(fk, q)
    T_new = np.asarray(T_delta, float) @ T_old
    x, y, z, yaw, pitch, roll = wrist_dofs_from_matrix(T_new)
    for name, val in zip(WRIST_TRANSLATION_DOFS + WRIST_ROTATION_DOFS,
                         (x, y, z, yaw, pitch, roll)):
        q[:, fk.dof_index[name]] = val
    return q
