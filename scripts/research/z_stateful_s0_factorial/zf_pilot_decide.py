#!/usr/bin/env python
"""Apply the rule of zf_common.INIT_AUG to the four pilot runs (zf_pilot_init.py) and write OUT/init_aug_decision.json (CPU only, validation data only).
    python zf_pilot_decide.py [--step 2000]
Rule: the augmentation (3 later frames per sequence for the initial-state module) is adopted iff, on ANY dataset,
      val RMSE(z_hat_0; frame-0-only) > 1.5 x val RMSE(z_hat_0; augmented)  at the pilot's last common step.
A decision file that already exists is never overwritten silently: it is first copied to logs/ with a time stamp.
"""
from __future__ import annotations

import argparse
import shutil
import time

import zf_common as Z


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--step", type=int, default=2000); ap.add_argument("--note", default="")
    a = ap.parse_args(); ratio, readings = {}, {}
    for ds in Z.DATASETS:
        r = {}
        for k in (0, 3):
            j = Z.read_json(Z.ds_out(ds) / f"pilot_init_aug{k}.json"); readings[f"{ds}/aug{k}"] = j["history"]
            row = [h for h in j["history"] if h["step"] == a.step]; assert row, f"{ds} aug{k}: no reading at step {a.step}"
            r[k] = row[0]["val_rmse_z0"]
        ratio[ds] = r[0] / r[3]
    frames = 3 if any(v > Z.INIT_AUG["threshold"] for v in ratio.values()) else 0
    out = Z.OUT / "init_aug_decision.json"
    if out.exists():
        shutil.copy2(out, Z.OUT / "logs" / f"init_aug_decision_before_{time.strftime('%m%d_%H%M%S')}.json")
    Z.write_json(out, dict(init_aug_frames=frames, rule=f"augmentation is used iff the held-out RMSE of z_hat_0 with frame-0-only supervision exceeds {Z.INIT_AUG['threshold']} x the RMSE with augmentation in the pilot (any dataset)",
                           **{f"ratio_frame0_only_over_augmented_at_step_{a.step}": ratio}, readings=readings, note=a.note, decided=time.strftime("%Y-%m-%d %H:%M KST") + "; validation data only"))
    print("init_aug_frames =", frames, "| ratios", {k: round(v, 3) for k, v in ratio.items()})


if __name__ == "__main__":
    main()
