"""Where dexcore finds assets and third-party code.

TWO SEARCH PATHS, because two kinds of thing live in different places and neither is a copy:

    ASSETS      an ordered list of roots. `third_party/dexmachina/dexmachina/assets` supplies the
                upstream ARCTIC objects, demos and MANO hands; `<repo>/assets` is where objects you
                bake yourself go. An object is looked up in each root in order, so a locally baked
                `box_s110` is found without having to be copied into the upstream tree, and an
                upstream `ketchup` is found without being copied here. Neither tree is modified.

    THIRD PARTY external repositories, used through the adapters in `src/thirdparty/` and never
                vendored. They are checkouts, not dependencies: dexcore imports them by path when
                it needs them and reports clearly when one is absent.

Override the asset search path with $DEXCORE_ASSET_PATH (colon-separated, highest priority first)
and the third-party root with $DEXCORE_THIRD_PARTY. Nothing else in dexcore hard-codes a path.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_ROOT = REPO_ROOT / "outputs"

# Object part ids as written into the 4th column of the contact arrays by process_arctic.py.
# 1 = top / lid / moving part, 2 = bottom / base. 0 only ever appears on a zero (no-contact) row.
LID_PART_ID = 1
BASE_PART_ID = 2
PART_NAME = {LID_PART_ID: "top", BASE_PART_ID: "bottom"}

# dexmachina steps ONE demo frame per env step at dt=1/60 (base_env.py), so the demo timeline is
# 60 Hz as far as anything trained on it is concerned.
DEFAULT_FPS = 60.0

SIDES = ("left", "right")


# --------------------------------------------------------------------------- third-party roots
def third_party_root() -> Path:
    return Path(os.environ.get("DEXCORE_THIRD_PARTY", REPO_ROOT / "third_party"))


def third_party(name: str, required: bool = True) -> Optional[Path]:
    """A checkout under `third_party/`. Returns None when absent and `required` is False.

    Absence is a normal, reportable state -- a baseline whose repository is not checked out should
    say so, not fail deep inside an import.
    """
    p = third_party_root() / name
    if p.exists():
        return p
    if not required:
        return None
    avail = sorted(d.name for d in third_party_root().iterdir()) \
        if third_party_root().exists() else []
    raise FileNotFoundError(
        f"third-party checkout {name!r} not found at {p}\n"
        f"Present: {', '.join(avail) or '(none)'}")


# ------------------------------------------------------------------------------- asset search
def _default_asset_path() -> List[Path]:
    """Repo-local assets first, then the upstream DexMachina tree.

    Local first so a locally baked object SHADOWS an upstream one of the same name -- which is what
    you want while iterating on a target, and which is visible because `resolve_object` reports the
    root an object came from.
    """
    roots = [REPO_ROOT / "assets"]
    dm = third_party("dexmachina", required=False)
    if dm is not None:
        roots.append(dm / "dexmachina" / "assets")
    return roots


def asset_path() -> List[Path]:
    """The ordered asset search path, honouring $DEXCORE_ASSET_PATH."""
    env = os.environ.get("DEXCORE_ASSET_PATH")
    roots = [Path(p) for p in env.split(os.pathsep) if p] if env else _default_asset_path()
    live = [r for r in roots if r.exists()]
    if not live:
        raise FileNotFoundError(
            "no asset root exists. Looked in:\n  " + "\n  ".join(str(r) for r in roots) +
            "\nSet $DEXCORE_ASSET_PATH, or check out third_party/dexmachina.")
    return live


def set_asset_path(*roots) -> None:
    """Point dexcore at a different asset search path for the rest of the process."""
    os.environ["DEXCORE_ASSET_PATH"] = os.pathsep.join(str(r) for r in roots)


def find_asset(relative: str) -> Path:
    """First match for `relative` across the asset search path, or a listing of where we looked."""
    tried = []
    for root in asset_path():
        p = root / relative
        tried.append(p)
        if p.exists():
            return p
    raise FileNotFoundError(
        f"asset {relative!r} not found. Looked in:\n  " + "\n  ".join(str(t) for t in tried))


def find_all(relative: str) -> List[Path]:
    """Every match across the search path, highest priority first. For reporting shadowing."""
    return [root / relative for root in asset_path() if (root / relative).exists()]


# --------------------------------------------------------------------------------- MANO hands
def mano_urdf(side: str) -> Path:
    if side not in SIDES:
        raise ValueError(f"side must be one of {SIDES}, got {side!r}")
    return find_asset(f"mano_hand/{side}/mano.urdf")


# ------------------------------------------------------------------------------------ objects
def object_dir(obj_name: str) -> Path:
    """Where `obj_name`'s URDF and meshes live, searched across every asset root.

    The object name is what resolves the GEOMETRY in DexMachina (`get_arctic_object_cfg` reads
    assets/arctic/<obj_name>/), which is why a demo generated for scaled geometry is named
    `box_s110`, not `box`. A locally baked object and an upstream one are found the same way.
    """
    try:
        return find_asset(f"arctic/{obj_name}")
    except FileNotFoundError:
        raise FileNotFoundError(
            f"object {obj_name!r} not found in any asset root.\n"
            f"Known objects: {', '.join(list_objects())}") from None


def object_root_of(obj_name: str) -> Path:
    """Which asset root supplied this object. Reported so shadowing is visible, never silent."""
    return object_dir(obj_name).parent.parent


def list_objects() -> List[str]:
    """Every object with a URDF, across all roots. Earlier roots shadow later ones by name."""
    seen = {}
    for root in asset_path():
        d = root / "arctic"
        if not d.exists():
            continue
        for sub in sorted(d.iterdir()):
            if sub.is_dir() and sub.name != "processed" and list(sub.glob("*.urdf")):
                seen.setdefault(sub.name, root)
    return sorted(seen)


# -------------------------------------------------------------------------------------- demos
def parse_demo_stem(stem: str):
    """'box_s090_use_090c' -> ('box_s090', '090c'). Returns (stem, None) with no _use_ token."""
    if "_use_" not in stem:
        return stem, None
    obj, clip = stem.rsplit("_use_", 1)
    return obj, clip


def demo_path(obj_name: str = "box", use_clip: str = "01", subject: str = "s01") -> Path:
    """`<root>/arctic/processed/<subject>/<obj>_use_<clip>.npy` -- the DexMachina convention."""
    return find_asset(f"arctic/processed/{subject}/{obj_name}_use_{use_clip}.npy")


def resolve_demo(path=None, obj_name: str = "box", use_clip: str = "01",
                 subject: str = "s01") -> Path:
    """An explicit path wins; otherwise search the asset path. Fails with what IS available.

    The explicit-path branch is what makes dexcore usable on a demo it generated itself, before
    that demo is installed into any asset tree.
    """
    if path:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"demo not found: {p}")
        return p
    try:
        return demo_path(obj_name, use_clip, subject)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"demo {obj_name}_use_{use_clip} not found for subject {subject!r}.\n"
            f"Installed: {', '.join(list_demos(subject)) or '(none)'}") from None


def list_demos(subject: str = "s01") -> List[str]:
    """Every installed demo stem for a subject, across all roots."""
    seen = {}
    for root in asset_path():
        d = root / "arctic" / "processed" / subject
        if d.exists():
            for f in sorted(d.glob("*.npy")):
                seen.setdefault(f.stem, root)
    return sorted(seen)


def processed_dir(subject: str = "s01", writable: bool = False) -> Path:
    """The processed-demo directory. With `writable`, the REPO-LOCAL one (never third_party).

    Installing a generated demo must not write into a third-party checkout: that tree is a
    checkout, and a modified checkout is one nobody can reason about. Generated references go to
    `<repo>/assets/arctic/processed/<subject>/`, which is earlier on the search path and therefore
    found first.
    """
    if writable:
        p = REPO_ROOT / "assets" / "arctic" / "processed" / subject
        p.mkdir(parents=True, exist_ok=True)
        return p
    return find_asset(f"arctic/processed/{subject}")
