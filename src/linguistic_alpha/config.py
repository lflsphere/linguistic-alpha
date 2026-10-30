"""Load the YAML run configuration and resolve data paths."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"


@dataclass(frozen=True)
class Paths:
    raw: Path
    interim: Path
    processed: Path
    external: Path
    legacy_panel: Path

    def ensure(self) -> None:
        for p in (self.raw, self.interim, self.processed, self.external):
            p.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class Config:
    raw: dict[str, Any]
    paths: Paths

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]


def _resolve(p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def load_config(path: str | Path | None = None) -> Config:
    """Read a config file; a custom file is merged shallowly over the default."""
    with open(DEFAULT_CONFIG) as f:
        cfg = yaml.safe_load(f)
    if path is not None and Path(path).resolve() != DEFAULT_CONFIG:
        with open(path) as f:
            override = yaml.safe_load(f) or {}
        for key, value in override.items():
            if isinstance(value, dict) and isinstance(cfg.get(key), dict):
                cfg[key] = {**cfg[key], **value}
            else:
                cfg[key] = value
    paths = Paths(**{k: _resolve(v) for k, v in cfg["paths"].items()})
    return Config(raw=cfg, paths=paths)
