"""Linguistic alpha from earnings calls.

Pipeline stages (each reads/writes parquet under ``data/``):

    raw (WRDS exports)
      -> data.transcripts / data.linking / data.universe   -> interim
      -> events.timing (Day 0)                              -> processed/events
      -> targets.car, controls.*                            -> processed/targets, controls
      -> features.* (F1-F5 families)                        -> processed/features
      -> models.* (Model A vs Model B, walk-forward)        -> processed/predictions
      -> evaluation.* (IC, Fama-MacBeth, backtest)          -> reports
"""

__version__ = "0.1.0"
