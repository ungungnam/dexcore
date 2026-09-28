"""Per-sequence scalar features -- the quantities whose distribution the action is conditioned on.

ONE ROW PER SEQUENCE. Every function here maps an `HOITrajectory` to a flat dict of floats, so a
sweep is a dataframe and "the distribution of X given the verb" is a groupby. Keeping features
scalar is the constraint that makes them comparable across sequences of different lengths.

WHAT THE FEATURES ARE TRYING TO SEPARATE. The verbs in TACO differ along a few physical axes, and
each block below targets one of them:

    how long, how far      `duration_s`, `tool_path_m`      -- a `cut` is not a `pour in some`
    how fast, how sharply  speed / angular-speed quantiles  -- `hit` and `stir-fry` are impulsive
    how the tool is held   tilt range, height range         -- `pour in some` and `empty` TILT
    where the tool is      tool-target distance, `near_frac`-- `put in` ends inside the target
    rhythm                 autocorrelation period          -- `stir`, `brush`, `dust` repeat
    which hands work       per-hand speed, tool distance    -- one-handed vs bimanual actions

MISSING PARTS ARE NaN, NOT ZERO. ARCTIC has no target object, so its target features are NaN and
drop out of a mean instead of dragging it to zero.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from src import geometry as G
from src.analysis.schema import HandTrack, HOITrajectory, ObjectTrack

# A hand is "at" the tool when its root is within this of the tool's ORIGIN -- the mesh frame's
# origin, not the grasp point and not the surface. Contact would need the mesh, which this module
# deliberately never loads.
#
# READ `*_tool_near_frac` WITH THIS IN MIND. On TACO it comes out near 0.7 for short tools (brush,
# eraser) and exactly 0 for long ones (knife, spatula), whose origin sits a handle-length away from
# the hand. It therefore carries as much tool geometry as it does action, and it tops the by-verb
# ranking largely because TACO pairs each verb with its own tool. `distribution.compare_groupings`
# exists to make that visible rather than let it pass as an action effect.
#
# IT IS ALSO NOT COMPARABLE BETWEEN THE TWO DATASETS. One radius is applied to both, but the hand
# root means different things in each (see `schema.HandTrack`): ARCTIC's is the smplx `transl`,
# roughly 10 cm from the wrist -- an offset of the same order as the radius itself. Measured
# medians: `rh_tool_dist_mean_m` 0.18 m on TACO against 0.38 m on ARCTIC. Read `*_tool_near_frac`
# and `hand_sep_*` within a dataset only.
GRASP_RADIUS_M = 0.15
# Likewise for tool-to-target: "near" is tool-origin to target-origin, an interaction proxy.
NEAR_RADIUS_M = 0.25
# Human manipulation rhythm lives here. BOTH bounds are load-bearing.
#
# Upper: allowing shorter periods lets a smooth non-repeating trajectory score a "period" of a few
# frames, which is annotation smoothness, not rhythm.
#
# Lower: 0.8 Hz rather than 0.5 Hz because `rhythm` refuses to run at all unless the sequence is
# long enough to search the WHOLE band (see there). At 0.5 Hz that needs 152 frames and would
# discard 55% of TACO; at 0.8 Hz it needs 106 and discards 7%. The cost is real -- a repetition
# slower than 0.8 Hz is not reported -- but in a 5 s sequence such a period repeats three times at
# most, which is not evidence of rhythm anyway.
RHYTHM_BAND_HZ = (0.8, 3.0)
# Oscillations smaller than this (metres, RMS after the high-pass) are annotation noise, not
# motion. Positions are metres in both datasets, so the floor is a physical one.
MIN_RHYTHM_AMPLITUDE_M = 1e-4


# ------------------------------------------------------------------------------- primitives
def speeds(pos: np.ndarray, fps: float) -> np.ndarray:
    """(T,3) positions -> (T-1,) metres/second. Empty for a one-frame track."""
    if len(pos) < 2:
        return np.zeros(0)
    return np.linalg.norm(np.diff(pos, axis=0), axis=1) * fps


def angular_speeds(quat: np.ndarray, fps: float) -> np.ndarray:
    """(T,4) wxyz -> (T-1,) radians/second."""
    if len(quat) < 2:
        return np.zeros(0)
    return G.quat_geodesic(quat[:-1], quat[1:]) * fps


def path_length(pos: np.ndarray) -> float:
    if len(pos) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(pos, axis=0), axis=1).sum())


def _stats(x: np.ndarray, name: str, deg: bool = False, suffix: str = "") -> Dict[str, float]:
    """mean / p95 / max of a 1-D sample, NaN when empty.

    p95 as well as max because a single bad annotation frame owns the max, and several of these
    verbs (`hit`, `stir-fry`) are distinguished precisely by their high-speed tail.
    """
    keys = [f"{name}_{stat}{suffix}" for stat in ("mean", "p95", "max")]
    if len(x) == 0:
        return dict.fromkeys(keys, np.nan)
    x = np.degrees(x) if deg else x
    return dict(zip(keys, (float(np.mean(x)), float(np.percentile(x, 95)), float(np.max(x)))))


def tilt_angles(track: ObjectTrack) -> np.ndarray:
    """(T,) radians between the object's local +z and world +z.

    A pour or an empty swings the object's own axis away from vertical and back; a stir does not.
    The local axis is the mesh frame's, which is arbitrary per object -- so only the RANGE of this
    angle is comparable across objects, never its absolute value.
    """
    R = track.R()
    return np.arccos(np.clip(R[:, 2, 2], -1.0, 1.0))      # == angle(R @ e_z, e_z)


def _high_pass(x: np.ndarray, win: int):
    """(T,k) minus its centred moving average, with the incomplete-window edges dropped.

    Transport dominates raw manipulation signals: carrying a spoon across the table is a bigger,
    slower excursion than stirring with it. Removing a ~1 s moving average leaves the fast part.

    Returns `(signal, ok)`. `ok` is False when the window did not fit and all that could be done
    was to remove the mean -- which is NOT a high-pass, and leaves the transport component in.
    Callers must not treat that case as a filtered signal.
    """
    x = np.asarray(x, dtype=np.float64).reshape(len(x), -1)
    if win < 3 or win >= len(x):
        return x - x.mean(axis=0), False
    kernel = np.ones(win) / win
    smooth = np.stack([np.convolve(x[:, j], kernel, mode="same") for j in range(x.shape[1])], 1)
    edge = win // 2
    if len(x) <= 2 * edge + 16:
        return x - x.mean(axis=0), False
    return (x - smooth)[edge:len(x) - edge], True


def rhythm(signal: np.ndarray, fps: float, band=RHYTHM_BAND_HZ,
           highpass_s: float = 1.0, min_amplitude: float = MIN_RHYTHM_AMPLITUDE_M) -> Dict[str, float]:
    """Repetition rate of a (T,k) signal, by autocorrelation.

    Method: high-pass, then autocorrelate, then take the highest peak AFTER the first local
    minimum and inside `band`. The "after the first local minimum" step is what distinguishes a
    period from smoothness -- a non-repeating trajectory's autocorrelation just decays, and this
    returns NaN for it instead of reporting the shortest admissible lag.

    THE SEARCH WINDOW IS THE SAME FOR EVERY SEQUENCE, OR THERE IS NO RESULT. An earlier version
    capped the lag search at `T // 2`, which quietly RAISED the lowest reachable frequency for a
    short sequence: over TACO the reported frequency fell from a median 1.30 Hz for sequences under
    110 frames to 0.75 Hz over 220, a rank correlation of -0.45 with sequence length, and since
    length is itself verb-dependent (epsilon-squared 0.16) the feature was partly reporting
    duration. A frequency is now returned only when the whole of `band` is searchable, so two
    sequences' frequencies are always comparable.

    Returns NaN -- meaning "no period found", never a default number -- for a signal that is too
    short to search the band, too small in amplitude after the high-pass, or whose autocorrelation
    only decays.

    WHAT THIS FEATURE IS WORTH, measured rather than assumed. Over all 2317 TACO sequences the
    period features carry real action information: epsilon-squared by verb is moderate
    (`tool_autocorr_peak` 0.18, `rel_autocorr_peak` 0.20, `rel_freq_hz` 0.17) but their verb effect
    is the LEAST confounded by the tool of any feature here -- verb/tool effect ratios of 1.29-1.40
    against ~0.85 for the hand-tool distance features. A first probe on ten sequences per verb
    suggested the opposite and was simply underpowered; an FFT version of this feature, which the
    same probe also rejected, stayed rejected because the transport component owned the spectrum.
    """
    nan = {"freq_hz": np.nan, "autocorr_peak": np.nan}
    y, filtered = _high_pass(signal, int(round(highpass_s * fps)))
    if not filtered:                             # transport was never removed; see `_high_pass`
        return nan
    T = len(y)
    # The full band must fit, so that the search range does not depend on the sequence length.
    full_lag = int(np.floor(fps / band[0])) + 1
    if T // 2 < full_lag:
        return nan
    y = y - y.mean(axis=0)
    denom = float((y * y).sum())
    # An amplitude floor, not a tidiness check: a straight or stationary trajectory high-passes to
    # numerical noise, and the autocorrelation of noise has peaks like anything else. Without this
    # a tool being carried in a straight line reports a confident, meaningless frequency.
    if denom <= 0 or np.sqrt(denom / y.size) < min_amplitude:
        return nan
    max_lag, min_lag = full_lag, max(2, int(np.ceil(fps / band[1])))
    if max_lag <= min_lag + 1:
        return nan
    ac = np.array([float((y[:T - l] * y[l:]).sum()) for l in range(max_lag)]) / denom

    slope = np.diff(ac)
    first_min = next((i + 1 for i in range(len(slope) - 1)
                      if slope[i] <= 0 < slope[i + 1]), None)
    if first_min is None:                        # monotone decay: no repeat to find
        return nan
    lo = max(first_min, min_lag)
    if lo >= max_lag - 1:
        return nan
    seg = ac[lo:max_lag]
    k = int(np.argmax(seg))
    # A maximum sitting on either edge of the search window is where the window stops, not where
    # the signal repeats -- a 6 Hz oscillation would otherwise be reported as a confident 3 Hz.
    if k == len(seg) - 1 or (k == 0 and lo == min_lag):
        return nan
    return {"freq_hz": float(fps / (lo + k)), "autocorr_peak": float(seg[k])}


def relative_positions(moving: ObjectTrack, frame: ObjectTrack) -> np.ndarray:
    """`moving`'s origin expressed in `frame`'s moving coordinate system, (T,3) metres.

    Motion IN THE TARGET'S FRAME is what the verb is about: stirring a bowl that someone is also
    carrying is still stirring, and only the relative signal says so.
    """
    R = frame.R()                                          # (T,3,3)
    d = moving.pos - frame.pos                             # (T,3)
    return np.einsum("tji,tj->ti", R, d)                   # R^T d, per frame


# ---------------------------------------------------------------------------- feature blocks
#: The per-object feature names, in the order a row lays them out.
_OBJECT_KEYS = ("path_m", "disp_m", "tilt_range_deg", "z_range_m", "arti_range_deg",
                "speed_mean", "speed_p95", "speed_max",
                "angspeed_mean_deg", "angspeed_p95_deg", "angspeed_max_deg")


def _object_features(track: Optional[ObjectTrack], fps: float, prefix: str) -> Dict[str, float]:
    """Motion of one object. Absent object -> every key present and NaN, so rows stay aligned."""
    if track is None or len(track) == 0:
        return {f"{prefix}_{k}": np.nan for k in _OBJECT_KEYS}
    tilt = tilt_angles(track)
    out = {
        "path_m": path_length(track.pos),
        "disp_m": float(np.linalg.norm(track.pos[-1] - track.pos[0])),
        "tilt_range_deg": float(np.degrees(tilt.max() - tilt.min())),
        "z_range_m": float(track.pos[:, 2].max() - track.pos[:, 2].min()),
        "arti_range_deg": (float(np.degrees(track.arti.max() - track.arti.min()))
                           if track.is_articulated else np.nan),
        **_stats(speeds(track.pos, fps), "speed"),
        **_stats(angular_speeds(track.quat, fps), "angspeed", deg=True, suffix="_deg"),
    }
    return {f"{prefix}_{k}": out[k] for k in _OBJECT_KEYS}


def _interaction_features(tool: Optional[ObjectTrack], target: Optional[ObjectTrack],
                          fps: float) -> Dict[str, float]:
    keys = ("rel_dist_min_m", "rel_dist_mean_m", "rel_dist_range_m", "rel_speed_mean",
            "rel_path_m", "near_frac", "rel_freq_hz", "rel_autocorr_peak")
    if tool is None or target is None:
        return dict.fromkeys(keys, np.nan)
    rel = relative_positions(tool, target)
    dist = np.linalg.norm(rel, axis=1)
    spec = rhythm(rel, fps)
    return {"rel_dist_min_m": float(dist.min()), "rel_dist_mean_m": float(dist.mean()),
            "rel_dist_range_m": float(dist.max() - dist.min()),
            "rel_speed_mean": float(np.mean(speeds(rel, fps))) if len(rel) > 1 else np.nan,
            "rel_path_m": path_length(rel),
            "near_frac": float(np.mean(dist < NEAR_RADIUS_M)),
            "rel_freq_hz": spec["freq_hz"], "rel_autocorr_peak": spec["autocorr_peak"]}


def _hand_features(traj: HOITrajectory) -> Dict[str, float]:
    """Per-side motion, plus who is holding the tool and how busy the other hand is."""
    fps, tool = traj.fps, traj.tool
    out: Dict[str, float] = {}
    dist_to_tool: Dict[str, float] = {}
    for side in ("left", "right"):
        h: Optional[HandTrack] = traj.hands.get(side)
        p = side[0] + "h"                                   # "lh" / "rh"
        if h is None:
            out.update({f"{p}_path_m": np.nan, f"{p}_speed_mean": np.nan,
                        f"{p}_finger_std_deg": np.nan, f"{p}_tool_dist_mean_m": np.nan,
                        f"{p}_tool_near_frac": np.nan})
            continue
        v = speeds(h.root_pos, fps)
        out[f"{p}_path_m"] = path_length(h.root_pos)
        out[f"{p}_speed_mean"] = float(np.mean(v)) if len(v) else np.nan
        # Finger articulation ACTIVITY: the mean per-JOINT deviation of the axis-angle pose from
        # its own temporal mean. Per joint, not over the flat 45-vector, so the number reads as an
        # angle in degrees instead of scaling with the dimension count. A re-grasp or a squeeze
        # shows up here with no MANO forward pass.
        pose = h.finger_pose.reshape(len(h), -1, 3)
        out[f"{p}_finger_std_deg"] = float(np.degrees(
            np.linalg.norm(pose - pose.mean(axis=0), axis=2).mean()))
        if tool is not None:
            d = np.linalg.norm(h.root_pos - tool.pos, axis=1)
            out[f"{p}_tool_dist_mean_m"] = float(d.mean())
            out[f"{p}_tool_near_frac"] = float(np.mean(d < GRASP_RADIUS_M))
            dist_to_tool[side] = float(d.mean())
        else:
            out[f"{p}_tool_dist_mean_m"] = np.nan
            out[f"{p}_tool_near_frac"] = np.nan

    both = traj.hands.get("left"), traj.hands.get("right")
    if all(h is not None for h in both):
        sep = np.linalg.norm(both[0].root_pos - both[1].root_pos, axis=1)
        out["hand_sep_mean_m"] = float(sep.mean())
        out["hand_sep_range_m"] = float(sep.max() - sep.min())
    else:
        out["hand_sep_mean_m"] = out["hand_sep_range_m"] = np.nan

    # The acting hand is the one that stays nearer the tool over the sequence. `bimanual_ratio`
    # is the slower hand's mean speed over the faster's: ~0 when one hand does everything.
    out["acting_hand"] = min(dist_to_tool, key=dist_to_tool.get) if dist_to_tool else ""
    sp = [out[f"{s[0]}h_speed_mean"] for s in ("left", "right")]
    out["bimanual_ratio"] = (float(min(sp) / max(sp)) if all(np.isfinite(sp)) and max(sp) > 0
                             else np.nan)
    return out


# ------------------------------------------------------------------------------------ public
def sequence_features(traj: HOITrajectory) -> Dict[str, object]:
    """Every feature for one sequence, plus the labels a groupby needs. One dataframe row."""
    tool, target = traj.tool, traj.target
    row: Dict[str, object] = {
        "dataset": traj.dataset, "sequence_id": traj.sequence_id,
        "verb": traj.action.verb, "tool_name": traj.action.tool,
        "target_name": traj.action.target, "action": str(traj.action),
        "n_frames": traj.num_frames, "duration_s": traj.duration_s, "fps": traj.fps,
        "n_objects": len(traj.objects), "n_hands": len(traj.hands),
    }
    row.update(_object_features(tool, traj.fps, "tool"))
    row.update(_object_features(target, traj.fps, "target"))
    row.update(_interaction_features(tool, target, traj.fps))
    spec = rhythm(tool.pos, traj.fps) if tool is not None else {"freq_hz": np.nan,
                                                                "autocorr_peak": np.nan}
    row["tool_freq_hz"], row["tool_autocorr_peak"] = spec["freq_hz"], spec["autocorr_peak"]
    row.update(_hand_features(traj))
    row["notes"] = "; ".join(traj.notes)          # always present: every row has one key set
    return row


#: Columns that label a row rather than measure it -- everything else is a feature.
#: `n_frames` is here because `duration_s = n_frames / fps` and `fps` is constant within a dataset:
#: the two are the same quantity, and ranking both would put one physical fact in two rows of the
#: result and spend two of the report's plot slots on it.
LABEL_COLUMNS = ("dataset", "sequence_id", "verb", "tool_name", "target_name", "action",
                 "acting_hand", "notes", "fps", "n_frames")


def feature_columns(columns) -> list:
    """The numeric feature names in a dataframe, in order. One definition, used by every report."""
    return [c for c in columns if c not in LABEL_COLUMNS]
