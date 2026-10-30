"""Family 2 - Interaction Quality: F3 analyst-question evasiveness."""

from __future__ import annotations

import pandas as pd

from linguistic_alpha.features.base import FeatureFamily, FeatureInputs, register


@register
class InteractionQuality(FeatureFamily):
    """
    qa_relevance    : mean cosine(question emb, answer emb) over Q&A pairs
    evasiveness     : baseline 1 - qa_relevance; refinement: LLM/classifier 0-1 score
                      calibrated on a hand-labeled sample in data/external/
    non_answer_rate : share of pairs flagged as explicit non-answers
    Operates on ``inputs.qa_pairs`` (one question + its answer turns).
    """

    name = "interaction_quality"
    measures = ("qa_relevance", "evasiveness", "non_answer_rate")

    def compute(self, inputs: FeatureInputs) -> pd.DataFrame:
        raise NotImplementedError
