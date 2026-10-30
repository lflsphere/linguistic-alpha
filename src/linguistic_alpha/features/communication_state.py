"""Family 1 - Communication State: F1 tone and F2 uncertainty / hedging."""

from __future__ import annotations

import pandas as pd

from linguistic_alpha.features.base import FeatureFamily, FeatureInputs, register


@register
class CommunicationState(FeatureFamily):
    """
    tone_prepared / tone_answers : (pos - neg) / (pos + neg), LM lists, per section
    tone_gap                     : tone_answers - tone_prepared
    uncertainty                  : LM Uncertainty words per 1,000 (safe harbor removed)
    hedging                      : LM Weak_Modal + hedge phrases per 1,000
    Extensions (nlp extra): FinBERT sentence scores averaged per section.
    """

    name = "communication_state"
    measures = ("tone_prepared", "tone_answers", "tone_gap", "uncertainty", "hedging")

    def compute(self, inputs: FeatureInputs) -> pd.DataFrame:
        raise NotImplementedError
