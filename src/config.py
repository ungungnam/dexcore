"""Configuration: a run is fully described by a config, and a config round-trips to disk.

No experiment parameter is hard-coded anywhere in dexcore. A `PipelineConfig` names the data, the
three stages and their options, and it is written into every result's provenance -- so a saved
demonstration can always say what produced it, and a run can be repeated from its config alone.

LAYERING, most specific wins:

    defaults (this file)  <-  configs/*.yaml  <-  --set key=value / CLI flags

Stage options live in nested dicts keyed by stage name, so adding a selector or a transfer with new
options requires no change here -- which is the point of the registries in each stage's `base.py`.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, Optional

from src import paths


def _load_yaml(path) -> dict:
    import yaml
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _deep_merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in (over or {}).items():
        out[k] = _deep_merge(out[k], v) if isinstance(out.get(k), dict) and isinstance(v, dict) else v
    return out


def _coerce(text: str) -> Any:
    """'0.1' -> 0.1, 'true' -> True, '[1,2]' -> [1,2], anything else -> the string."""
    low = text.strip().lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("none", "null"):
        return None
    try:
        return json.loads(text)
    except Exception:
        return text


@dataclass
class PipelineConfig:
    """Everything one synthesis run needs. Written verbatim into the output's provenance."""

    # ---- data
    obj_name: str = "box"                  # the SOURCE object; also resolves the source geometry
    use_clip: str = "01"
    subject: str = "s01"
    demo_path: Optional[str] = None        # explicit .npy; wins over (obj_name, use_clip, subject)
    frame_start: int = 0
    frame_end: Optional[int] = None

    # ---- target
    target_object: Optional[str] = None    # None -> the source object (the same-object experiment)

    # ---- the synthesis METHOD, by registry name. "staged" is dexcore's own three-stage pipeline
    # and reads the stage fields below; an end-to-end method (bimart) has no stages and reads
    # `synthesizer_options` instead. See src/synthesis/base.py on why these are one level.
    synthesizer: str = "staged"
    synthesizer_options: Dict[str, Any] = field(default_factory=dict)

    # ---- stages, by registry name. Only meaningful for synthesizer="staged".
    selector: str = "uniform"
    transfer: str = "identity"
    reconstructor: str = "linear_keypoint"
    selector_options: Dict[str, Any] = field(default_factory=lambda: {"ratio": 0.1})
    transfer_options: Dict[str, Any] = field(default_factory=dict)
    reconstructor_options: Dict[str, Any] = field(default_factory=dict)

    # ---- evaluation & output
    metrics: list = field(default_factory=lambda: ["penetration", "collision", "smoothness"])
    output_dir: str = "outputs"
    run_name: Optional[str] = None         # None -> derived from the stages, see `resolve_run_name`
    seed: int = 0
    asset_root: Optional[str] = None   # colon-separated search path; see src/paths.py

    # ------------------------------------------------------------------ derived
    @property
    def resolved_target(self) -> str:
        return self.target_object or self.obj_name

    @property
    def is_same_object(self) -> bool:
        return self.resolved_target == self.obj_name

    @property
    def is_staged(self) -> bool:
        return self.synthesizer == "staged"

    def resolve_synthesizer_options(self) -> Dict[str, Any]:
        """The kwargs the chosen synthesizer is built with.

        For "staged" the three stage fields ARE its options -- they stay top-level in the config
        because every experiment so far sweeps them, and moving them under a nested key would
        break every config file and every `--set selector_options.ratio=...` on record.
        """
        if not self.is_staged:
            return dict(self.synthesizer_options)
        return {"selector": self.selector, "transfer": self.transfer,
                "reconstructor": self.reconstructor,
                "selector_options": dict(self.selector_options),
                "transfer_options": dict(self.transfer_options),
                "reconstructor_options": dict(self.reconstructor_options),
                **self.synthesizer_options}

    def resolve_run_name(self) -> str:
        """A name that says what the run was, so two runs cannot collide silently."""
        if self.run_name:
            return self.run_name
        if not self.is_staged:
            return (f"{self.obj_name}_use_{self.use_clip}__to_{self.resolved_target}"
                    f"__{self.synthesizer}")
        sel = self.selector
        r = self.selector_options.get("ratio")
        n = self.selector_options.get("num_frames")
        budget = f"{r:g}" if r is not None else (f"n{n}" if n is not None else "all")
        return (f"{self.obj_name}_use_{self.use_clip}__to_{self.resolved_target}"
                f"__{sel}-{budget}__{self.transfer}__{self.reconstructor}")

    def output_path(self) -> Path:
        return Path(self.output_dir) / self.resolve_run_name()

    # ------------------------------------------------------------------ IO
    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "PipelineConfig":
        known = set(cls.__dataclass_fields__)
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}. Known: {sorted(known)}")
        return cls(**d)

    @classmethod
    def load(cls, *sources, overrides: Optional[Dict[str, Any]] = None) -> "PipelineConfig":
        """Merge YAML files left to right, then apply dotted `overrides`, then validate.

        Each source may be a path to a YAML file or a dict. A dotted override key writes into a
        nested dict: `--set selector_options.ratio=0.05`.
        """
        merged: dict = {}
        for s in sources:
            if s is None:
                continue
            merged = _deep_merge(merged, s if isinstance(s, dict) else _load_yaml(s))
        for key, val in (overrides or {}).items():
            node = merged
            parts = str(key).split(".")
            for p in parts[:-1]:
                node = node.setdefault(p, {})
            node[parts[-1]] = _coerce(val) if isinstance(val, str) else val
        return cls.from_dict(merged).validate()

    def save(self, path) -> Path:
        import yaml
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(self.to_dict(), sort_keys=False))
        return path

    def validate(self) -> "PipelineConfig":
        if self.asset_root:
            paths.set_asset_path(*[r.strip() for r in str(self.asset_root).split(":") if r.strip()])
        if self.frame_end is not None and self.frame_end <= self.frame_start:
            raise ValueError(f"frame_end {self.frame_end} <= frame_start {self.frame_start}")
        if not self.is_staged:
            # A method with no stages would SILENTLY IGNORE these, and a config that names a
            # selector next to a method that has none is a config whose author expected something
            # that will not happen.
            staged_only = {f: getattr(self, f) for f in
                           ("selector", "transfer", "reconstructor", "selector_options",
                            "transfer_options", "reconstructor_options")}
            defaults = PipelineConfig()
            set_anyway = [f for f, v in staged_only.items() if v != getattr(defaults, f)]
            if set_anyway:
                raise ValueError(
                    f"synthesizer={self.synthesizer!r} has no stages, but the config also sets "
                    f"{sorted(set_anyway)}. Those would be ignored. Remove them, or use "
                    f"synthesizer=staged.")
            return self
        if self.transfer == "identity" and not self.is_same_object \
                and not self.transfer_options.get("allow_different_object"):
            raise ValueError(
                f"transfer=identity with target {self.resolved_target!r} != source "
                f"{self.obj_name!r}. A scaled object is a DIFFERENT object: transferring onto it by "
                f"doing nothing is not a baseline. Use a real transfer operator, or set "
                f"transfer_options.allow_different_object=true to record an explicit no-op.")
        return self

    def describe(self) -> str:
        head = (f"{self.obj_name}_use_{self.use_clip} [{self.frame_start}:{self.frame_end}] "
                f"-> {self.resolved_target}\n"
                f"  synthesizer   {self.synthesizer}\n")
        body = (f"  selector      {self.selector} {self.selector_options}\n"
                f"  transfer      {self.transfer} {self.transfer_options}\n"
                f"  reconstructor {self.reconstructor} {self.reconstructor_options}\n"
                if self.is_staged else
                f"  options       {self.synthesizer_options}\n")
        return head + body + f"  seed {self.seed}   output {self.output_path()}"
