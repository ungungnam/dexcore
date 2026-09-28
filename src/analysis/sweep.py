"""Run the feature extraction over a whole dataset, in parallel, without dying on one bad file.

The work is embarrassingly parallel and I/O-bound -- a TACO hand annotation is a 14 MB pickle per
side, so unpickling dominates and processes help. A worker returns a ROW, never a trajectory: the
trajectories are large and there is no reason to ship them back across the process boundary.
"""
from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional, Sequence

from src.analysis import loaders
from src.analysis.features import sequence_features

log = logging.getLogger(__name__)


def _extract(task) -> Optional[Dict[str, object]]:
    """One sequence -> one feature row, or None if it could not be read. Top-level so it pickles."""
    dataset, ref, root, fps = task
    mod = loaders.get(dataset)
    try:
        traj = mod.load(ref, root=root, **({"fps": fps} if fps is not None else {}))
        return sequence_features(traj)
    except Exception as e:                        # noqa: BLE001 -- one bad sequence is not a run
        log.warning("%s/%s: %s: %s", dataset, ref.sequence_id, type(e).__name__, e)
        return None


def run(dataset: str, root=None, refs: Optional[Sequence] = None, workers: int = 1,
        fps: Optional[float] = None, progress: bool = True, **index_kwargs) -> List[Dict[str, object]]:
    """Feature rows for every sequence of `dataset`. `index_kwargs` filter the index (verbs=...)."""
    mod = loaders.get(dataset)
    root = root or mod.default_root()
    refs = list(refs if refs is not None else mod.index(root, **index_kwargs))
    tasks = [(dataset, r, root, fps) for r in refs]
    log.info("%s: %d sequences from %s (workers=%d)", dataset, len(tasks), root, workers)

    rows: List[Dict[str, object]] = []
    if workers and workers > 1:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor
        # fork: the child inherits the already-imported modules, and nothing here holds a handle
        # that must not be duplicated.
        ctx = mp.get_context("fork")
        # sched_getaffinity, not cpu_count: on a shared box the affinity mask is the real budget.
        avail = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
        # ProcessPoolExecutor rather than mp.Pool: if the OOM killer takes a worker mid-sweep --
        # this is a shared machine and each TACO sequence unpickles two 14 MB files -- a Pool waits
        # for a result that will never come and the run hangs with no output. The executor raises.
        with ProcessPoolExecutor(max_workers=max(1, min(workers, avail)), mp_context=ctx) as ex:
            for i, row in enumerate(ex.map(_extract, tasks, chunksize=4), 1):
                if row is not None:
                    rows.append(row)
                _tick(i, len(tasks), progress)
    else:
        for i, task in enumerate(tasks, 1):
            row = _extract(task)
            if row is not None:
                rows.append(row)
            _tick(i, len(tasks), progress)

    failed = len(tasks) - len(rows)
    if failed:
        log.warning("%d/%d sequences failed and were skipped", failed, len(tasks))
    return rows


def _tick(i: int, n: int, progress: bool, every: int = 50) -> None:
    if progress and (i % every == 0 or i == n):
        log.info("  %d/%d", i, n)
