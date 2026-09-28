"""Render ONE processed demo (source or generated reference) to an mp4, in its own process.

demo_gen.render_demo plays the demo back kinematically -- object on its recorded trajectory, MANO
hands posed from configs_{side}, contact links as spheres. One Genesis init per process, so this is
deliberately a single-shot script the driver shells out to.

  python render_demo_one.py --npy <processed.npy> --target <registry name> --out <dir> [--tag name]
"""
import argparse
import os
import shutil
import sys

sys.path.insert(0, "/home/uhnam/workspace/dexmachina")
sys.path.insert(0, "/home/uhnam/workspace/dexcore")   # the ARCTIC-source fallback needs src.paths


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--npy", required=True)
    p.add_argument("--target", required=True, help="registry entry supplying scale/obj_dir/dataset")
    p.add_argument("--out", required=True)
    p.add_argument("--tag", default=None, help="basename for the copied mp4")
    p.add_argument("--frame_start", type=int, default=30)
    p.add_argument("--frames", type=int, default=200)
    p.add_argument("--res", type=int, default=480)
    p.add_argument("--fps", type=int, default=30)
    a = p.parse_args()

    os.chdir("/home/uhnam/workspace/dexmachina")
    from dexmachina.eval.experiments.exp3_demo_gen_baselines import registry as R, demo_gen
    from dexmachina.eval.experiments.exp3_demo_gen_baselines.paths import MANO_DIR

    # The registry describes BAKED targets, and its `kind` only admits scaled_box|pm. An ARCTIC
    # source object (box, laptop, microwave, …) is neither: it is the object the demo was recorded
    # on, shipped with DexMachina rather than baked here. Registering one would mean labelling a
    # microwave a "scaled_box", so the spec is built directly from the asset instead.
    try:
        spec = R.gen_spec_for(a.target)
    except KeyError:
        from src import paths as _paths
        obj_dir = _paths.object_dir(a.target)
        if not obj_dir.exists():
            raise
        spec = {"scale": 1.0, "obj_dir": str(obj_dir), "dataset": "arctic"}
        print(f"  (레지스트리 밖 ARCTIC 원본으로 처리: {obj_dir})")
    work = os.path.join(a.out, "_work_" + (a.tag or "demo"))
    os.makedirs(work, exist_ok=True)
    # render_demo's non-PM branch hardcodes `get_arctic_object_cfg(name="box")`: it was written when
    # box was the only ARCTIC source, so rendering the microwave demo silently drew a box. Point that
    # call at the object this spec actually describes, for the duration of this one render.
    import contextlib

    @contextlib.contextmanager
    def _arctic_object_is(name):
        if spec.get("dataset") == "pm" or name == "box":
            yield
            return
        # render_demo imports the helper INSIDE the function, so it resolves from the defining
        # module at call time -- patching demo_gen's namespace would never be seen.
        import dexmachina.envs.object as _obj
        orig = _obj.get_arctic_object_cfg

        def patched(*args, **kw):
            kw = {**kw, "name": name}
            return orig(*args, **kw)

        _obj.get_arctic_object_cfg = patched
        try:
            yield
        finally:
            _obj.get_arctic_object_cfg = orig

    with _arctic_object_is(a.target):
        mp4 = demo_gen.render_demo(a.npy, spec, work, MANO_DIR,
                                   res=a.res, fps=a.fps,
                                   frame_start=a.frame_start, frames=a.frames)
    if a.tag:
        dest = os.path.join(a.out, a.tag + ".mp4")
        shutil.copyfile(mp4, dest)
        mp4 = dest
    print("RENDERED " + mp4)


if __name__ == "__main__":
    main()
