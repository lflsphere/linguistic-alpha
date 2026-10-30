"""Common interface for feature families."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import pandas as pd

FEATURE_FAMILIES: dict[str, type[FeatureFamily]] = {}


def register(cls: type[FeatureFamily]) -> type[FeatureFamily]:
    FEATURE_FAMILIES[cls.name] = cls
    return cls


@dataclass
class FeatureInputs:
    """Everything a family may read. All tables are keyed by ``transcriptid``."""

    components: pd.DataFrame            # schemas.COMPONENTS
    qa_pairs: pd.DataFrame              # schemas.QA_PAIRS
    sections: pd.DataFrame              # transcripts.call_sections output
    events: pd.DataFrame                # keydevid, transcriptid, permno, call_date_et
    press_releases: pd.DataFrame | None = None   # F5 (wrdssec 8-K Ex.99.1), optional
    upstream: dict[str, pd.DataFrame] = field(default_factory=dict)  # earlier families


class FeatureFamily(ABC):
    """A Level-1 family. Output: one row per ``transcriptid``, one column per measure.

    Implementations must use only text of the call itself (and, for ``consistency``, the same
    firm's *earlier* calls), so every feature is known at the call date.
    """

    name: str
    measures: tuple[str, ...]
    depends_on: tuple[str, ...] = ()

    def __init__(self, cfg: dict):
        self.cfg = cfg

    @abstractmethod
    def compute(self, inputs: FeatureInputs) -> pd.DataFrame: ...
