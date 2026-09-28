#!/usr/bin/env python
"""Local web viewer for generated demonstrations. Upload a .npy, scrub a time slider, see the scene.

  python eval/visualization/serve.py            # then open http://127.0.0.1:8000

WHY A SERVER AND NOT A PURE STATIC PAGE. A processed demo is a pickled numpy dict; a browser cannot
read one. The server is the numpy process that turns it into a scene (see scene.py), so "upload a
demo" works on the files that actually exist rather than on some export format nobody produces.

IT BINDS TO LOCALHOST. This box is shared, and the server reads files off disk on request. Reads are
confined to `outputs/` and the DexMachina processed-demo tree; anything else is refused, so a stray
path in a URL cannot walk the filesystem. Pass --host to change the bind at your own risk.
"""

import pathlib as _pl
import sys as _sys

_HERE = str(_pl.Path(__file__).resolve().parent)
_REPO = str(_pl.Path(__file__).resolve().parents[2])
_sys.path[:] = [p for p in _sys.path if p not in ("", _HERE)]
if _REPO not in _sys.path:
    _sys.path.insert(0, _REPO)

import argparse
import json
import tempfile
import traceback
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

STATIC = Path(__file__).resolve().parent / "static"


def allowed_roots():
    """The only directories a scene may be read from."""
    from src import paths
    roots = [Path(paths.REPO_ROOT / "outputs").resolve()]
    # every processed-demo directory on the asset search path, not just the first: a locally
    # generated reference and an upstream one legitimately live in different roots, and the viewer
    # should be able to open either without the user knowing which root won.
    for root in paths.asset_path():
        d = (root / "arctic" / "processed").resolve()
        if d.exists() and d not in roots:
            roots.append(d)
    return roots


def resolve_demo_path(raw: str) -> Path:
    """Resolve a requested path, refusing anything outside the allowed roots."""
    p = Path(raw).expanduser()
    if not p.is_absolute():
        from src import paths
        p = (paths.REPO_ROOT / p)
    p = p.resolve()
    roots = allowed_roots()
    if not any(str(p).startswith(str(r) + "/") or p == r for r in roots):
        raise PermissionError(
            f"refusing to read {p}: outside the allowed roots "
            + ", ".join(str(r) for r in roots))
    if not p.exists():
        raise FileNotFoundError(f"{p} does not exist")
    return p


def list_demos() -> list:
    """Every .npy the viewer is willing to open, newest first, with a one-line label."""
    out = []
    for root in allowed_roots():
        if not root.exists():
            continue
        for f in sorted(root.rglob("*.npy")):
            try:
                rel = f.relative_to(root)
            except ValueError:
                rel = f
            sidecar = f.with_suffix(".json")
            label = ""
            if sidecar.exists():
                try:
                    meta = json.loads(sidecar.read_text())
                    label = (f"{meta.get('selector', '')} {meta.get('transfer', '')} "
                             f"{meta.get('reconstructor', '')}").strip()
                except Exception:
                    pass
            if f.stem == "selected_keyframes":
                label = "K_s SPARSE (selected, source object) · " + label
            elif f.stem == "transferred_keyframes":
                label = "K_t SPARSE (transferred, target object) · " + label
            out.append({"path": str(f), "name": f.stem, "group": root.name,
                        "rel": str(rel), "label": label,
                        "size_mb": round(f.stat().st_size / 1e6, 2),
                        "mtime": f.stat().st_mtime})
    out.sort(key=lambda d: -d["mtime"])
    return out


def scene_for(path: Path, start=None, end=None) -> dict:
    """Build a scene for a demo, optionally windowed by ABSOLUTE SOURCE FRAME.

    `start`/`end` are absolute frames in the ORIGINAL clip's numbering -- the same numbers that
    appear in a DexMachina clip string (`box-30-230-s01-u01`) -- NOT indices into this particular
    file. That distinction is the whole point: a generated demo already carries `frame_start=30`,
    so interpreting 30 as a file index would silently land on source frame 60. Asking for 30..230
    now means the same instants whether the file is the 889-frame source or a 200-frame demo cut
    from it, and on a demo that is already exactly that window it is a no-op.
    """
    from eval.visualization.scene import build_scene
    from src.data.demo import Demonstration

    demo = Demonstration.load(path=str(path))
    if start is None and end is None:
        return build_scene(demo)

    frames = demo.frames()
    lo = int(start) if start not in (None, "") else int(frames.min())
    hi = int(end) if end not in (None, "") else int(frames.max()) + 1
    rows = [i for i, f in enumerate(frames) if lo <= int(f) < hi]
    if len(rows) < 2:
        kind = "sparse keyframe set" if demo.is_sparse else "demo"
        raise ValueError(
            f"frames {lo}..{hi} keep only {len(rows)} of this {kind}'s {len(demo)} frames "
            f"(it covers source frames {int(frames.min())}..{int(frames.max())}). "
            f"Clear the frame window, or pick a range that overlaps it.")
    if len(rows) == len(demo):
        return build_scene(demo)
    return build_scene(demo.subset(rows))



def parse_single_upload(content_type: str, body: bytes):
    """Pull one (filename, bytes) out of a multipart/form-data body. -> (name, data).

    Hand-rolled because the standard library's `cgi` module, which used to do this, is REMOVED in
    Python 3.13 -- and the system python on this box is 3.13, so depending on it would leave a
    viewer that works today and dies on whichever interpreter someone reaches for next.
    Deliberately minimal: one file part, which is all the upload form sends.
    """
    ct = content_type or ""
    if "multipart/form-data" not in ct:
        raise ValueError(f"expected multipart/form-data, got {ct!r}")
    marker = "boundary="
    if marker not in ct:
        raise ValueError("multipart upload has no boundary")
    boundary = ct.split(marker, 1)[1].strip().strip('"')
    sep = b"--" + boundary.encode()

    for part in body.split(sep):
        if not part.strip(b"-\r\n"):
            continue
        head, _, data = part.partition(b"\r\n\r\n")
        if not _:
            continue
        headers = head.decode("utf-8", "replace")
        if "filename=" not in headers:
            continue
        name = headers.split("filename=", 1)[1].split("\r\n")[0].strip().strip('"')
        if not name:
            continue
        return name, data.rstrip(b"\r\n")
    raise ValueError("no file part found in the upload")


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=str(STATIC), **kw)

    def log_message(self, fmt, *args):
        if "--verbose" in _sys.argv:
            super().log_message(fmt, *args)

    # ------------------------------------------------------------------ helpers
    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _fail(self, exc, code=400):
        """Errors reach the browser with their real message. A viewer that says only 'failed' is
        useless for debugging a demo, which is the entire point of the viewer."""
        self._json({"error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc()}, code=code)

    # ------------------------------------------------------------------ routes
    def do_GET(self):
        u = urlparse(self.path)
        if u.path == "/api/demos":
            try:
                return self._json({"demos": list_demos()})
            except Exception as e:
                return self._fail(e, 500)
        if u.path == "/api/scene":
            q = parse_qs(u.query)
            raw = (q.get("path") or [""])[0]
            if not raw:
                return self._json({"error": "missing ?path="}, 400)
            try:
                return self._json(scene_for(resolve_demo_path(raw),
                                            start=(q.get("start") or [""])[0] or None,
                                            end=(q.get("end") or [""])[0] or None))
            except PermissionError as e:
                return self._fail(e, 403)
            except FileNotFoundError as e:
                return self._fail(e, 404)
            except Exception as e:
                return self._fail(e, 500)
        return super().do_GET()

    def do_POST(self):
        if urlparse(self.path).path != "/api/upload":
            return self._json({"error": "unknown endpoint"}, 404)
        try:
            name, blob = parse_single_upload(
                self.headers.get("Content-Type", ""),
                self.rfile.read(int(self.headers.get("Content-Length", 0))))
            with tempfile.TemporaryDirectory() as td:
                # the filename MATTERS: Demonstration.load parses obj_name/use_clip out of the
                # stem, and obj_name is what resolves the object geometry.
                p = Path(td) / Path(name).name
                p.write_bytes(blob)
                q = parse_qs(urlparse(self.path).query)
                return self._json(scene_for(p, start=(q.get("start") or [""])[0] or None,
                                            end=(q.get("end") or [""])[0] or None))
        except Exception as e:
            return self._fail(e, 400)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    if not STATIC.exists():
        raise SystemExit(f"static directory missing: {STATIC}")
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"dexcore viewer on http://{args.host}:{args.port}")
    print(f"  serving  {STATIC}")
    for r in allowed_roots():
        print(f"  reading  {r}")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
