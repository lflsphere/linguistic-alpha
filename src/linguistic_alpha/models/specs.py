"""Model specifications: which columns enter Models A, B, C and C_alone.

  A       = controls (earnings surprise and firm controls)
  B       = A + dictionary measures (method 1: NLTK tokens + Loughran-McDonald lists)
  C       = B + FinBERT sentence-level sentiment (method 2)
  C_alone = A + FinBERT only

A, B and C nest, so B - A is the value of dictionary language and C - B the value FinBERT
adds on top. C_alone vs. B compares the two methods head to head; they are not nested, so
that comparison uses Delta IC / Delta R^2 with bootstrap p-values, not Clark-West.
``COMPARISONS`` lists every (base, alternative, nested) pair that is reported.

Every model runs in two legs that differ only in the reaction control added to Model A's
controls (``controls.reaction_legs``): the main leg uses CAR[-1,+1] (``car_m1_1``), the
comparison leg CAR[0,+1] (``car_0_1``). ``model_specs_by_leg`` returns both. Measures from methods 3-4 (embeddings, LLM rating) are
listed under ``models.measures.later`` and stay out of A/B/C for now.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    name: str
    target: str
    regressors: tuple[str, ...]
    fixed_effects: tuple[str, ...] = ()


COMPARISONS = (("A", "B", True), ("A", "C", True), ("A", "C_alone", True),
               ("B", "C", True), ("B", "C_alone", False))


def model_specs(cfg: dict, target: str = "car_2_20",
                reaction: str | None = None) -> dict[str, ModelSpec]:
    """Models A / B / C / C_alone for one leg; ``reaction`` defaults to the main leg."""
    reaction = reaction or cfg["controls"]["reaction_legs"]["main"]
    controls = tuple(cfg["controls"]["include"]) + (reaction,)
    m = cfg["models"]["measures"]
    dictionary, finbert = tuple(m["dictionary"]), tuple(m["finbert"])
    fe = tuple(cfg["models"]["fixed_effects"])
    return {
        "A": ModelSpec("A", target, controls, fe),
        "B": ModelSpec("B", target, controls + dictionary, fe),
        "C": ModelSpec("C", target, controls + dictionary + finbert, fe),
        "C_alone": ModelSpec("C_alone", target, controls + finbert, fe),
    }


def model_specs_by_leg(cfg: dict, target: str = "car_2_20") -> dict[str, dict[str, ModelSpec]]:
    """{leg name: models}, e.g. {"main": {...}, "comparison": {...}}."""
    return {leg: model_specs(cfg, target, reaction)
            for leg, reaction in cfg["controls"]["reaction_legs"].items()}
