"""Linguistic features (Model B only), organised in the framework's three Level-1 families.

  communication_state  -> F1 tone, F2 uncertainty / hedging   (what is said, how confidently)
  interaction_quality  -> F3 relevance, evasiveness, non-answers (how analysts' Qs are handled)
  consistency          -> F4 change vs. own history, F5 call vs. press release
                          (transformations of the families above, not primitive factors)

Status: structure only. Each family exposes ``compute`` with a fixed input/output contract.
"""

from linguistic_alpha.features.base import FEATURE_FAMILIES, FeatureFamily, register

__all__ = ["FEATURE_FAMILIES", "FeatureFamily", "register"]
