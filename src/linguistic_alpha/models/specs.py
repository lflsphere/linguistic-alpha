"""Model specifications: which columns enter Model A and Model B."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    name: str
    target: str
    regressors: tuple[str, ...]
    fixed_effects: tuple[str, ...] = ()


def model_specs(cfg: dict, target: str = "car_2_20") -> dict[str, ModelSpec]:
    controls = tuple(cfg["controls"]["include"])
    features = tuple(m for family in cfg["features"].values() if isinstance(family, list)
                     for m in family)
    fe = tuple(cfg["models"]["fixed_effects"])
    return {
        "A": ModelSpec("A", target, controls, fe),
        "B": ModelSpec("B", target, controls + features, fe),
    }
