"""Category-canonical contact representation on TACO: map every mesh of a category into one
shared frame so per-vertex contact can be written as a FIXED-length vector over K canonical
surface points, comparable across the different mesh instances of that category.

WHAT THIS IS, AND IS NOT. The backends shipped here are GEOMETRIC BASELINES. They put meshes into
a common coordinate frame (centring, scaling, PCA axes, rigid ICP) and let each canonical point read
the value of whatever instance vertex lands closest. Two spatulas overlapping in space after
alignment is a spatial coincidence, not a statement that "this vertex is the tip of the blade" --
aligned / normalised coordinates are NOT semantic correspondence. Treat the numbers they produce as
the floor a learned or functional correspondence has to beat, and read the `random_perm` backend as
the chance level for the same pipeline.

    backends/normalized.py   baseline B: centre + isotropic scale, no rotation (invertible)
    backends/aligned.py      the "method": medoid template, PCA axes, proper-rotation sign search,
                             similarity ICP refinement; still purely geometric
    backends/random_perm.py  sanity: `aligned` with the K canonical indices shuffled per mesh
    backends/dino.py         the SEMANTIC backend: `aligned` + DINOv2 multi-view correspondence of
                             every instance onto the aligned template (same K points); falls back
                             to `aligned` per point where the match is invalid or ambiguous
    fit_all.py               fits and caches the three geometric ones, writes the quality table
    fit_dino.py              runs the DINO correspondence for every mesh (worker processes, GPU)

Nothing here touches BimArt or its preprocessing (`src/analysis/bimart` is only READ for the mesh
dictionary); this is an analysis of the representation, not a retraining.
"""
from src.analysis.canonical.interface import (CanonicalBackend, canonical_to_object,
                                              object_to_canonical)

__all__ = ["CanonicalBackend", "canonical_to_object", "object_to_canonical"]
