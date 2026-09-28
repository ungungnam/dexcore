"""DexMachina adapter: make a dexcore-generated demonstration trainable, and read back ADD-AUC.

DexMachina IS NOT REIMPLEMENTED HERE, and nothing in this file imports it. It is an export plus a
command line, because that is all that is needed:

EXPORT IS A FILE COPY. dexcore writes DexMachina's own on-disk format (see src/data/demo.py), so
"export" means putting the .npy where `get_demo_data()` looks:

    assets/arctic/processed/<subject>/<obj_name>_use_<use_clip>.npy

TWO NAMING RULES, and both are load-bearing (dexmachina exp3/install_demos.py):
  * `obj_name` RESOLVES THE GEOMETRY -- `get_arctic_object_cfg(name=obj_name)` reads
    assets/arctic/<obj_name>/. So a demo generated against box_s110 must be named `box_s110`, never
    `box`, or it trains against the wrong mesh with no error.
  * THE STRATEGY VARIANT GOES IN THE CLIP FIELD, which `parse_clip_string` accepts as an arbitrary
    token. `box_s110_use_110pff.npy` is (obj_name=box_s110, use_clip=110pff).

Naming a demo after its strategy in the obj_name field would demand an object asset of that name.

ADD-AUC is computed by DexMachina's own `dexmachina/eval/compute_add.py`: per-part vertex ADD over
top/bottom, then AUC by trapezoid over thresholds 0.01..0.1 m. This module builds the command and
parses `add_stats.json`; it does not recompute the metric.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from src import paths
from src.data.demo import Demonstration

DEXMACHINA_ROOT = Path("/home/uhnam/workspace/dexmachina")


@dataclass
class ExportResult:
    installed_path: Path
    obj_name: str
    use_clip: str
    subject: str
    frame_start: int
    frame_end: int

    @property
    def clip_string(self) -> str:
        """What DexMachina training takes as `--clip`: obj-start-end-subject-uCLIP.

        The frame range is the demo's OWN window, taken from the file rather than left as a
        placeholder for someone to fill in -- a clip string with the wrong range trains on a
        different segment than the one that was generated, and nothing downstream would object.
        """
        return (f"{self.obj_name}-{self.frame_start}-{self.frame_end}-"
                f"{self.subject}-u{self.use_clip}")

    def describe(self) -> str:
        return (f"installed {self.installed_path}\n"
                f"  obj_name={self.obj_name}  use_clip={self.use_clip}  subject={self.subject}\n"
                f"  frames {self.frame_start}-{self.frame_end}\n"
                f"  clip string: {self.clip_string}")


def export(demo: Demonstration, use_clip: Optional[str] = None, subject: str = "s01",
           overwrite: bool = False, dry_run: bool = False,
           dest_root: Optional[Path] = None) -> ExportResult:
    """Install a generated demo into the DexMachina asset tree so training can read it.

    `use_clip` is where the strategy variant belongs -- pass e.g. "110pff" to distinguish this
    run from the original clip. Defaults to the demo's own clip, which will COLLIDE with the source
    if the object is unchanged, so overwrite must then be explicit.

    `dest_root` is the `arctic/processed` directory to install into, and it is EXPLICIT because the
    right answer depends on who is going to read the file. dexcore's own repo-local root is the
    default -- it is first on the asset search path, and writing there never touches a checkout.
    But DexMachina training resolves `ARCTIC_PROCESSED_DIR` inside ITS OWN tree and will not look
    at ours, so installing a demo for training means naming that tree. Guessing between the two
    silently produces a file nothing reads.
    """
    obj_name = demo.obj_name
    clip = use_clip or demo.use_clip
    obj_dir = paths.object_dir(obj_name)
    if not obj_dir.exists():
        raise FileNotFoundError(
            f"no object asset {obj_name!r} at {obj_dir}. DexMachina resolves geometry from the "
            f"obj_name field, so the demo cannot be installed until that asset exists "
            f"(bake it first). Known: {', '.join(paths.list_objects())}")

    # `writable=True` and the SUBJECT are both load-bearing. Without writable, `processed_dir`
    # returns whichever root won the search -- which can be the third_party checkout this
    # repository refuses to write into (see its docstring). And it already includes the subject,
    # so appending it again produced `.../processed/s01/s01/...`, a path DexMachina never reads.
    root = Path(dest_root) if dest_root is not None else paths.processed_dir(subject, writable=True)
    dest = root / f"{obj_name}_use_{clip}.npy"
    if dest.exists() and not overwrite:
        raise FileExistsError(
            f"{dest} already exists. Pass overwrite=True, or give a different use_clip -- the clip "
            f"field is where the strategy variant belongs (e.g. '{clip}pff').")
    if not dry_run:
        dest.parent.mkdir(parents=True, exist_ok=True)
        demo.save(dest)
    frames = demo.frames()
    return ExportResult(installed_path=dest, obj_name=obj_name, use_clip=clip, subject=subject,
                        frame_start=int(frames.min()), frame_end=int(frames.max()) + 1)


def add_auc_command(eval_path, obj_name: Optional[str] = None,
                    dexmachina_root: Path = DEXMACHINA_ROOT) -> list:
    """The `compute_add.py` command line for a DexMachina eval rollout. Not run here."""
    cmd = ["python", "-m", "dexmachina.eval.compute_add", "--input", str(eval_path)]
    if obj_name:
        cmd += ["--obj_name", obj_name]
    return cmd


def read_add_stats(stats_path) -> dict:
    """Parse `add_stats.json` written by DexMachina's compute_add.

    Returns the per-part and overall mean ADD and AUC. `auc.overall` is the headline number: the
    mean over the top and bottom parts of the area under the accuracy-vs-threshold curve.
    """
    p = Path(stats_path)
    if not p.exists():
        raise FileNotFoundError(
            f"{p} not found. Run a DexMachina eval rollout first, then:\n  "
            + " ".join(add_auc_command("<eval_ep*.npy>")))
    d = json.loads(p.read_text())
    for key in ("auc", "mean_add"):
        if key not in d:
            raise ValueError(f"{p} is missing {key!r}; is this a compute_add stats file?")
    return {"auc_overall": d["auc"].get("overall"),
            "auc_top": d["auc"].get("top"), "auc_bottom": d["auc"].get("bottom"),
            "mean_add_overall": d["mean_add"].get("overall"),
            "thresholds": d.get("thresholds"), "source": str(p)}
