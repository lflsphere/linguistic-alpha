"""Family 3 - Communication Consistency / Deviation: F4 history change, F5 divergence.

These are transformations of Families 1-2, so they run after them (``depends_on``).
"""

from __future__ import annotations

import pandas as pd

from linguistic_alpha.features.base import FeatureFamily, FeatureInputs, register


@register
class Consistency(FeatureFamily):
    """
    d_tone, d_uncertainty, d_evasiveness : current minus mean of the firm's previous
        ``features.history_calls`` calls; NaN until ``history_min_calls`` exist
        (legacy notebook used min_periods=1, i.e. a 1-call baseline)
    novelty            : 1 - cos(emb_t, emb_{t-1}) for comparable sections
    div_qa_prepared    : tone / embedding distance, Q&A vs. prepared remarks
    div_press_release  : tone / embedding distance, prepared remarks vs. same-day 8-K Ex.99.1
    """

    name = "consistency"
    measures = ("d_tone", "d_uncertainty", "d_evasiveness", "novelty",
                "div_qa_prepared", "div_press_release")
    depends_on = ("communication_state", "interaction_quality")

    def compute(self, inputs: FeatureInputs) -> pd.DataFrame:
        raise NotImplementedError
