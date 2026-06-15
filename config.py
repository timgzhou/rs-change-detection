"""Typed run config for finetune_olmoearth_pastis.py, loaded from a YAML file.

Intentionally dependency-light (stdlib dataclasses + pyyaml only, no torch / olmoearth
imports) so it loads fast and can be unit-checked on its own.

    from config import load_config
    cfg = load_config("configs/base_s2s1_anyup.yaml")
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Literal

import yaml

ModelSize = Literal["nano", "tiny", "base", "large"]

# model_size -> full-package ModelID enum *value* (avoids importing the enum here).
# The full olmoearth_pretrain package only ships v1 in these 4 sizes (no v1_1), and
# we must use it because get_eval_wrapper dispatches on its encoder class.
MODEL_SIZE_TO_ID = {
    "nano": "OLMOEARTH_V1_NANO",
    "tiny": "OLMOEARTH_V1_TINY",
    "base": "OLMOEARTH_V1_BASE",
    "large": "OLMOEARTH_V1_LARGE",
}
ALLOWED_MODALITIES = ("sentinel2_l2a", "sentinel1")


@dataclass
class Config:
    """A single finetune run. All fields have defaults matching the previous hardcoded
    values, so a minimal YAML (or none) reproduces the old behavior."""

    model_size: ModelSize = "base"
    input_modalities: list[str] = field(default_factory=lambda: ["sentinel2_l2a", "sentinel1"])
    use_anyup: bool = False
    epochs: int = 64
    lr: float = 1e-3
    batch_size: int = 32
    num_workers: int = 0
    seed: int = 0
    data_splits: str = "data/pastis_olmoearth"
    dataset: str = "pastis"

    def __post_init__(self) -> None:
        if self.model_size not in MODEL_SIZE_TO_ID:
            raise ValueError(
                f"model_size={self.model_size!r} not in {list(MODEL_SIZE_TO_ID)}"
            )
        if not self.input_modalities:
            raise ValueError("input_modalities must be non-empty")
        bad = [m for m in self.input_modalities if m not in ALLOWED_MODALITIES]
        if bad:
            raise ValueError(f"input_modalities {bad} not in {list(ALLOWED_MODALITIES)}")

    # --- derived ---
    @property
    def model_id_name(self) -> str:
        """Name of the ModelID enum member (resolved against the enum in the script)."""
        return MODEL_SIZE_TO_ID[self.model_size]

    @property
    def head(self) -> str:
        return "anyup" if self.use_anyup else "lp"

    @property
    def run_name(self) -> str:
        """Unique, human-readable id for this run; used for the checkpoint filename so
        sweeps don't overwrite each other, e.g. oe_pastis_base_s2s1_anyup_lr3e-4_ep64."""
        mods = "".join(
            {"sentinel2_l2a": "s2", "sentinel1": "s1"}[m] for m in self.input_modalities
        )
        return (f"oe_{self.dataset}_{self.model_size}_{mods}_{self.head}"
                f"_lr{self.lr:g}_ep{self.epochs}")

    @property
    def ckpt_path(self) -> str:
        return f"checkpoints/{self.run_name}_best.pt"


def load_config(path: str) -> Config:
    """Load a Config from a YAML file. Unknown keys raise a clear error."""
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: top-level YAML must be a mapping, got {type(data).__name__}")
    known = set(Config.__dataclass_fields__)
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"{path}: unknown config keys {sorted(unknown)}; allowed: {sorted(known)}")
    return Config(**data)


def to_dict(cfg: Config) -> dict:
    """Plain dict of the stored fields (for logging/provenance)."""
    return asdict(cfg)
