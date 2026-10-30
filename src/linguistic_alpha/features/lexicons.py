"""Word lists: Loughran-McDonald Master Dictionary (Notre Dame SRAF), hedge phrases.

Status: structure only. The LM CSV goes in ``data/external/`` (not on WRDS, not committed).
"""

from __future__ import annotations

from pathlib import Path

LM_CATEGORIES = ("Positive", "Negative", "Uncertainty", "Weak_Modal", "Strong_Modal",
                 "Litigious", "Constraining")


def load_loughran_mcdonald(external_dir: Path) -> dict[str, frozenset[str]]:
    """Return ``{category_lower: words}`` from ``*MasterDictionary*.csv``."""
    raise NotImplementedError


def load_hedge_phrases(external_dir: Path) -> tuple[str, ...]:
    """Multi-word hedges ("we believe", "it is possible") not covered by LM single words."""
    raise NotImplementedError


def strip_safe_harbor(text: str) -> str:
    """Remove forward-looking-statement boilerplate before counting (scope F2)."""
    raise NotImplementedError
