"""Dataset analysis: what do the demonstrations actually look like, grouped by action?

Separate from `src/data` on purpose. `src/data` holds the pipeline's own demonstration format;
this package reads the SOURCE datasets (TACO, ARCTIC) through one neutral schema and measures
them. Nothing here is imported by the synthesis pipeline.
"""
from src.analysis.schema import ActionLabel, HandTrack, HOITrajectory, ObjectTrack

__all__ = ["ActionLabel", "HandTrack", "HOITrajectory", "ObjectTrack"]
