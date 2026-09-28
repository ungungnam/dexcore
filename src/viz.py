"""Inspection plots. Saved files, never a GUI -- this runs over ssh on a shared box.

Everything here answers "what actually happened in this run": which frames were selected, where the
reconstruction is worst, which frames penetrate. Nothing here is a result to publish; these are the
pictures you look at before believing a number.

Matplotlib is imported lazily and with the Agg backend, so importing dexcore never opens a display.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np

from src.data.mano import JOINT_NAMES, THUMB_TIP_JOINT


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_selected_frames(demo, sparse, out_path, error: Optional[np.ndarray] = None,
                         title: str = "") -> Path:
    """Selected frames over time, optionally against a per-frame error curve.

    The picture to look at when asking "did the selector put its budget where the motion is?".
    """
    plt = _plt()
    frames = np.asarray(sparse.frames())
    T = demo.num_frames
    fig, ax = plt.subplots(figsize=(11, 3.2))
    if error is not None:
        ax.plot(np.arange(T), np.asarray(error) * 1000, lw=1.0, color="#4477aa",
                label="per-frame error")
        ax.set_ylabel("error (mm)")
    ax.vlines(frames, *ax.get_ylim() if error is not None else (0, 1),
              color="#cc3311", lw=0.8, alpha=0.8, label=f"selected ({len(frames)})")
    ax.set_xlabel("frame")
    ax.set_xlim(0, T - 1)
    ax.set_title(title or f"{demo.name}: {len(frames)}/{T} frames selected "
                          f"({len(frames)/T:.1%})")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    return _save(fig, out_path)


def plot_error_curve(per_frame: Dict[str, np.ndarray], out_path, ylabel: str = "error (mm)",
                     scale: float = 1000.0, title: str = "",
                     mark: Optional[Sequence[int]] = None) -> Path:
    """Per-frame curves for any metric, with the selected frames marked."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(11, 3.4))
    for name, v in sorted(per_frame.items()):
        ax.plot(np.asarray(v) * scale, lw=1.0, label=name)
    if mark is not None and len(mark):
        for f in np.asarray(mark):
            ax.axvline(f, color="#cc3311", lw=0.4, alpha=0.35)
    ax.set_xlabel("frame"); ax.set_ylabel(ylabel); ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return _save(fig, out_path)


def plot_trajectories(demos: Dict[str, object], out_path, side: str = "right",
                      joint: int = THUMB_TIP_JOINT, title: str = "") -> Path:
    """One keypoint's world path across several demos -- source vs selected vs reconstructed.

    The thumb tip by default: it moves the most and is where in-betweening shows. Its index is
    taken from `mano.THUMB_TIP_JOINT` rather than written here -- it is 16, not 20, and the
    hard-coded 20 this used to carry silently plotted the PINKY tip under a thumb_tip filename.
    """
    plt = _plt()
    fig, axes = plt.subplots(3, 1, figsize=(11, 6), sharex=True)
    for name, d in demos.items():
        p = d.joints[side][:, joint]
        f = np.asarray(d.frames())
        style = dict(marker="o", ms=3, lw=0, alpha=0.9) if d.is_sparse else dict(lw=1.2)
        for k, ax in enumerate(axes):
            ax.plot(f, p[:, k], label=name, **style)
    for k, ax in enumerate(axes):
        ax.set_ylabel("xyz"[k] + " (m)")
    axes[-1].set_xlabel("frame")
    axes[0].set_title(title or f"{side} joint {joint} world position")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    return _save(fig, out_path)


def plot_object_trajectory(traj, out_path, mark: Optional[Sequence[int]] = None) -> Path:
    """tau_O: root position and articulation over time. The half that never gets thinned."""
    plt = _plt()
    fig, axes = plt.subplots(2, 1, figsize=(11, 4.4), sharex=True)
    f = np.asarray(traj.frames())
    for k, lbl in enumerate("xyz"):
        axes[0].plot(f, traj.obj_pos[:, k], lw=1.1, label=lbl)
    axes[0].set_ylabel("root pos (m)"); axes[0].legend(fontsize=8)
    axes[1].plot(f, np.degrees(traj.obj_arti), lw=1.1, color="#228833")
    axes[1].set_ylabel("articulation (deg)"); axes[1].set_xlabel("frame")
    if mark is not None:
        for ax in axes:
            for x in np.asarray(mark):
                ax.axvline(x, color="#cc3311", lw=0.4, alpha=0.35)
    axes[0].set_title(f"tau_O [{traj.obj_name}] -- {traj.source}")
    fig.tight_layout()
    return _save(fig, out_path)


def plot_failure_frames(per_frame: np.ndarray, out_path, threshold: float,
                        label: str = "penetration (mm)", scale: float = 1000.0) -> Path:
    """Highlight the frames a metric flags, so they can be found and rendered."""
    plt = _plt()
    v = np.asarray(per_frame) * scale
    bad = np.flatnonzero(v > threshold * scale)
    fig, ax = plt.subplots(figsize=(11, 3.0))
    ax.plot(v, lw=1.0, color="#4477aa")
    ax.axhline(threshold * scale, color="#cc3311", ls="--", lw=0.9,
               label=f"threshold {threshold*scale:g}")
    if len(bad):
        ax.plot(bad, v[bad], "o", ms=3, color="#cc3311", label=f"{len(bad)} flagged frames")
    ax.set_xlabel("frame"); ax.set_ylabel(label); ax.legend(fontsize=8)
    ax.set_title(f"{label}: {len(bad)}/{len(v)} frames over threshold")
    fig.tight_layout()
    return _save(fig, out_path)


def export_keypoints_ply(demo, out_path, side: str = "right", frame: int = 0) -> Path:
    """One frame's 21 keypoints as an ASCII .ply, for loading next to the object mesh in a viewer.

    Deliberately dependency-free: an ASCII point cloud opens in meshlab/blender/open3d without
    dexcore having to pick a renderer.
    """
    p = np.asarray(demo.joints[side][frame], dtype=float)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = ["ply", "format ascii 1.0", f"element vertex {len(p)}",
             "property float x", "property float y", "property float z", "end_header"]
    lines += [f"{a:.6f} {b:.6f} {c:.6f}" for a, b, c in p]
    out.write_text("\n".join(lines) + "\n")
    return out


def _save(fig, out_path) -> Path:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    import matplotlib.pyplot as plt
    plt.close(fig)
    return out


def report(output, out_dir=None) -> Path:
    """Write the standard inspection set for a `PipelineOutput`. One call after a run.

    The selection figure is drawn only for a method that HAS a selection. A method with no stages
    still gets its object trajectory, its per-frame diagnostics and its keypoint path, because
    those are properties of the synthesis and not of any particular decomposition of it.
    """
    out = Path(out_dir or (Path(output.config.output_path()) / "figures"))
    out.mkdir(parents=True, exist_ok=True)

    staged = hasattr(output.result, "sparse_demo")
    sel = np.asarray(output.sparse_demo.frames()) if staged else None
    demos = {"source": output.source_demo, "generated": output.demo}

    if staged:
        from src.selection.reconstruction_aware import keypoint_criterion
        plot_selected_frames(output.source_demo, output.sparse_demo, out / "selected_frames.png",
                             error=keypoint_criterion(output.source_demo, sel))
        demos["selected"] = output.sparse_demo

    plot_object_trajectory(output.trajectory, out / "object_trajectory.png", mark=sel)
    per_frame = output.result.diagnostics.per_frame
    if per_frame:
        plot_error_curve(per_frame, out / "per_frame_diagnostics.png",
                         ylabel="residual (mm)",
                         title=f"{output.result.method}: per-frame diagnostics", mark=sel)
    plot_trajectories(demos, out / f"{JOINT_NAMES[THUMB_TIP_JOINT]}.png")
    return out
