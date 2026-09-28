#!/usr/bin/env python
"""Build a DexMachina-format asset from ARCTIC's raw object template.

ARCTIC ships each object as two millimetre-scale meshes (`top.obj`, `bottom.obj`) under
`meta/object_vtemplates/<name>/`. DexMachina wants the layout the existing six objects already use:

    assets/arctic/<name>/
        {top,bottom}_watertight_tiny.{obj,stl}     visual + contact mesh, metres
        <name>.urdf                                 the articulated object
        decomp/{top,bottom}_watertight_tiny_partN.obj
        decomp/<name>_decomp.urdf                   convex pieces, what the simulator collides with

Nothing here is invented: the joint is the convention every ARCTIC object already carries -- origin
at 0, axis (0,0,-1), limits 0..pi -- verified identical across box, laptop, ketchup, mixer, notebook
and waffleiron before this script was written. Masses come from the mesh volume at a stated density
rather than from ARCTIC, which does not publish them.

    python bake_arctic_object.py --name microwave \
        --src /home/uhnam/workspace/arctic/downloads/data/extracted/meta/object_vtemplates
"""

import pathlib as _pl
import sys as _sys

# `scripts/select.py` shadows the stdlib `select` (imported by subprocess, imported by numpy), so the
# script's own directory has to leave sys.path before anything else is imported.
_HERE = str(_pl.Path(__file__).resolve().parent)
_REPO = str(_pl.Path(__file__).resolve().parent.parent)
_sys.path[:] = [p for p in _sys.path if p not in ("", _HERE)]
if _REPO not in _sys.path:
    _sys.path.insert(0, _REPO)

import argparse
import shutil
from pathlib import Path

import numpy as np
import trimesh

MM_TO_M = 0.001
# ARCTIC publishes no masses. These are a stated assumption, not a measurement: a plastic-ish
# density applied to the watertight volume, floored so a thin panel still has usable inertia.
DENSITY = 300.0          # kg/m^3
MIN_MASS = 0.05          # kg


def load_part(src_dir: Path, part: str, target_faces: int) -> trimesh.Trimesh:
    """One part of the ARCTIC template, in metres, watertight, simplified to the shipped budget.

    The six objects DexMachina already carries store each part as a ~252-vertex / 500-face
    watertight mesh -- not a convex hull (measured volume ratio 0.84-0.87 against the hull), just a
    heavily reduced solid. A new object has to match that, because the same file is what the
    contact pipeline samples and what `<visual>` shows; leaving the raw 26k-vertex template in its
    place would describe a different geometry from every existing asset.

    Simplification runs AFTER hole-filling, so the reduction operates on a solid and the result
    stays watertight.
    """
    import open3d as o3d

    # process=True already merges duplicate vertices and drops degenerate/unreferenced ones; the
    # explicit removers that used to live here were removed from trimesh 4.
    m = trimesh.load(str(src_dir / f"{part}.obj"), process=True, force="mesh")
    m.apply_scale(MM_TO_M)
    if not target_faces or len(m.faces) <= target_faces:
        return m

    # Open3D's decimation, NOT trimesh's. Reproducing laptop from its raw template, this path lands
    # on 252 verts / 500 faces / watertight with volume ratio 1.000 and extents within 0.8 mm of the
    # shipped asset -- i.e. it is the reduction ARCTIC itself used. trimesh's `simplify_quadric_
    # decimation` loses 28% of the volume on the same input, and a voxel remesh inflates it by 79%.
    om = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(np.asarray(m.vertices, float)),
        o3d.utility.Vector3iVector(np.asarray(m.faces)))
    om.remove_duplicated_vertices()
    om.remove_degenerate_triangles()
    om.remove_duplicated_triangles()
    om.remove_non_manifold_edges()
    s = om.simplify_quadric_decimation(target_number_of_triangles=target_faces)

    d = trimesh.Trimesh(np.asarray(s.vertices), np.asarray(s.triangles), process=True)
    trimesh.repair.fill_holes(d)
    trimesh.repair.fix_normals(d)
    if not d.is_watertight:
        print(f"    ! {part}: 단순화 결과가 watertight 아님 -- 원본 유지")
        return m
    drift = abs(d.volume - m.volume) / max(abs(m.volume), 1e-9)
    if drift > 0.15:
        print(f"    ! {part}: 단순화가 부피를 {drift*100:.0f}% 바꿈 -- 원본 유지")
        return m
    return d


def inertial(mesh: trimesh.Trimesh):
    """(mass, centre of mass, 3x3 inertia) for a uniform-density solid.

    Falls back to the convex hull when the mesh is not watertight: a hull's volume is an
    over-estimate, but an open mesh has no signed volume at all and trimesh would report garbage.
    """
    solid = mesh if mesh.is_watertight else mesh.convex_hull
    vol = float(abs(solid.volume))
    mass = max(MIN_MASS, DENSITY * vol)
    solid = solid.copy()
    solid.density = mass / vol if vol > 0 else DENSITY
    return mass, np.asarray(solid.center_mass, float), np.asarray(solid.moment_inertia, float)


def convex_parts(mesh: trimesh.Trimesh, out_dir: Path, stem: str, threshold: float,
                 max_hulls: int):
    """CoACD pieces for one part, written next to each other; returns their filenames.

    `max_hulls` is what actually sets the count. On the laptop bottom, threshold alone gives 9
    pieces at 0.05 and collapses to 1 by 0.08 -- there is no threshold in between that lands on the
    2 the shipped asset carries. The cap reaches it directly, and piece count is the term that
    drives simulator memory: a laptop-sized target measured ~0.6 GB per collision piece on top of a
    ~19 GB floor for B=12288.
    """
    import coacd
    out_dir.mkdir(parents=True, exist_ok=True)
    cm = coacd.Mesh(np.asarray(mesh.vertices, float), np.asarray(mesh.faces))
    kw = {"threshold": threshold}
    if max_hulls:
        kw["max_convex_hull"] = int(max_hulls)
    parts = coacd.run_coacd(cm, **kw)
    names = []
    for i, (v, f) in enumerate(parts):
        fn = f"{stem}_part{i}.obj"
        trimesh.Trimesh(vertices=np.asarray(v), faces=np.asarray(f)).export(str(out_dir / fn))
        names.append(fn)
    return names


def link_block(link: str, visual: str, collisions, mass, com, I, prefix=""):
    cols = "\n".join(
        f'      <collision><origin rpy="0 0 0" xyz="0 0 0"/><geometry>'
        f'<mesh filename="{prefix}{c}" scale="1 1 1"/></geometry></collision>' for c in collisions)
    return f"""   <link name="{link}">
      <inertial>
         <origin rpy="0 0 0" xyz="{com[0]:.9g} {com[1]:.9g} {com[2]:.9g}"/>
         <mass value="{mass:.9g}"/>
         <inertia ixx="{I[0,0]:.9g}" ixy="{I[0,1]:.9g}" ixz="{I[0,2]:.9g}"
                  iyy="{I[1,1]:.9g}" iyz="{I[1,2]:.9g}" izz="{I[2,2]:.9g}"/>
      </inertial>
      <visual><origin rpy="0 0 0" xyz="0 0 0"/><geometry>
         <mesh filename="{prefix}{visual}" scale="1 1 1"/></geometry>
         <material name="obj_color"/></visual>
{cols}
   </link>"""


def write_urdf(path: Path, name: str, blocks):
    # The joint every ARCTIC object uses. Checked against all six shipped objects, not assumed.
    path.write_text(f"""<?xml version="1.0" ?>
<robot name="{name}">
   <material name="obj_color">
      <color rgba="1.0 0.423529411765 0.0392156862745 1.0"/>
   </material>
{blocks}
   <joint name="rotation" type="revolute">
      <origin xyz="0 0 0"/>
      <axis xyz="0 0 -1"/>
      <parent link="bottom"/>
      <child link="top"/>
      <dynamics damping="0.0" friction="0.000"/>
      <limit effort="1000" velocity="200" lower="0" upper="{np.pi}"/>
   </joint>
</robot>
""", encoding="utf-8")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True, help="ARCTIC object name, e.g. microwave")
    p.add_argument("--src", required=True, help="…/meta/object_vtemplates")
    p.add_argument("--asset_root",
                   default="/home/uhnam/workspace/dexmachina/dexmachina/assets/arctic")
    p.add_argument("--coacd_threshold", type=float, default=0.05)
    p.add_argument("--max_hulls", type=int, default=2,
                   help="convex pieces per part. The six shipped ARCTIC objects use 2; raise it "
                        "for a concave object whose grasp depends on the cavity. 0 = uncapped.")
    p.add_argument("--target_faces", type=int, default=500,
                   help="face budget for the simplified part mesh; the six shipped "
                        "ARCTIC objects all use 500. 0 disables simplification.")
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args()

    src = Path(a.src) / a.name
    out = Path(a.asset_root) / a.name
    if out.exists() and not a.overwrite:
        raise SystemExit(f"{out} already exists; pass --overwrite to rebuild")
    if out.exists():
        shutil.rmtree(out)
    (out / "decomp").mkdir(parents=True)

    blocks, decomp_blocks = [], []
    for part in ("bottom", "top"):
        m = load_part(src, part, a.target_faces)
        stem = f"{part}_watertight_tiny"
        m.export(str(out / f"{stem}.obj"))
        m.export(str(out / f"{stem}.stl"))
        mass, com, I = inertial(m)
        cols = convex_parts(m, out / "decomp", stem, a.coacd_threshold, a.max_hulls)
        print(f"  {part:7s} verts {len(m.vertices):6d}  watertight={m.is_watertight}  "
              f"size {np.round(m.extents, 3)}  mass {mass:.3f}kg  조각 {len(cols)}")
        # top-level urdf: one collision mesh, the part itself
        blocks.append(link_block(part, f"{stem}.obj", [f"{stem}.obj"], mass, com, I))
        # decomp urdf lives in decomp/, so its meshes are siblings and the visual is one level up
        decomp_blocks.append(link_block(part, f"../{stem}.obj", cols, mass, com, I))

    write_urdf(out / f"{a.name}.urdf", a.name, "\n".join(blocks))
    write_urdf(out / "decomp" / f"{a.name}_decomp.urdf", a.name, "\n".join(decomp_blocks))
    print(f"\n  baked -> {out}")
    print(f"  충돌 조각 총 {sum(1 for _ in (out / 'decomp').glob('*_part*.obj'))}개")


if __name__ == "__main__":
    main()
