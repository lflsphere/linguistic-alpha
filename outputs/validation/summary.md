# Step 1: measure validation (development period only)

Events: 167 (2010-2023), Q-A pairs: 2740. wrote 150 blind pairs to data/qa_label_sample.csv.

## Speaker structure
| index | value |
|---|---|
| question_count_match | 0.874 |
| pairs_per_call | 16.407 |
| pair_coverage_untruncated | 0.990 |

## Leakage and bias checks
| check | status | detail |
|---|---|---|
| day 0 consistent with call time (post-close -> next session, pre-open -> same day) | PASS | 0 inconsistent; sessions: {'post': 178, 'pre': 1} |
| no intraday calls (reaction day ambiguous) | PASS | 0 intraday calls |
| fiscal period ends before the call | PASS | min gap 5 days |
| no post-event quantity in Model A / Model B feature lists | PASS | [] |
| CAR[0,+1] excluded from the primary specification | PASS | AR01 only enters with --delayed |
| Delta features use only the previous calls of the same firm | PASS | max |recomputed - stored| = 2.22e-16 |
| NOVEL uses the previous call only | PASS | first call of each firm is NaN |
| time-series SUE scale uses prior surprises only | PASS | rolling SD of shifted surprises, window (8, 4) |
| unsupervised text models fitted without holdout text | PASS | pre-trained (no fit on this sample) |
| pre-call analyst consensus timestamps | UNVERIFIABLE | panel has no estimate dates; confirm the I/B/E/S statistical period precedes the call date |
| historical S&P 500 membership on the call date | UNVERIFIABLE | tickers ['AAPL', 'MSFT', 'NVDA']: all three were members over 2010-2024; no membership table in the panel |
| split-adjusted consensus rounding (Payne-Thomas) | WARN | 40 events with |consensus| < 0.05, by ticker {'NVDA': 40}; robustness: --sample highprec |
| transcript text truncated at the Excel cell limit | WARN | Q&A truncated in 47 calls, prepared in 12; densities stay unbiased only if language is homogeneous within the call |
| lexicon | PASS | MasterDictionary.csv |
| LLM look-ahead | WARN | no generative LLM scoring; pre-trained encoders receive anonymised text and are version-pinned in config |
| holdout coverage | WARN | 12 holdout events (2024-2024); CAR_2_60 missing for 3 |
| alternative abnormal-return benchmark (Fama-French) | UNVERIFIABLE | needs daily stock and factor returns, not in the panel; market-adjusted CAR only |

## Construct validity (expected signs)
| feature | against | expected_sign | spearman | p_value | n | sign_ok |
|---|---|---|---|---|---|---|
| TONE_P | SUE | 1 | 0.190 | 0.018 | 155 | True |
| TONE_Q | SUE | 1 | 0.168 | 0.036 | 155 | True |
| TONE_P | AR01 | 1 | 0.079 | 0.313 | 167 | True |
| TONE_Q | AR01 | 1 | 0.118 | 0.128 | 167 | True |
| dTONE | AR01 | 1 | 0.133 | 0.098 | 155 | True |
| UNC | VOL | 1 | -0.016 | 0.839 | 167 | False |
| UNC | TONE_M | -1 | -0.150 | 0.053 | 167 | True |
| EVADE | NONANS | 1 | -0.038 | 0.628 | 167 | False |
| HEDGE_Q | EVADE | 1 | -0.096 | 0.218 | 167 | False |

## Alternative measurements
| level | measure_a | measure_b | spearman | pearson | n |
|---|---|---|---|---|---|
| call | TONE_P | TONE_P_PW | 0.775 | 0.845 | 167 |
| call | TONE_Q | TONE_Q_PW | 0.870 | 0.875 | 167 |
| call | REL | REL_JAC | 0.486 | 0.511 | 167 |
| call | EVADE | NONANS | -0.038 | -0.079 | 167 |
| call | UNC | HEDGE | 0.034 | 0.014 | 167 |
| call | UNC_P | UNC_Q | -0.250 | -0.228 | 167 |
| pair | REL | REL_JAC | 0.463 | 0.428 | 2740 |

## Redundant pairs (|Spearman| > 0.8)
| feature_a | feature_b | spearman |
|---|---|---|
| EVADE | REL | -1.000 |

## Distributions
| index | mean | std | min | 50% | max | skew | missing_share |
|---|---|---|---|---|---|---|---|
| TONE_P | 0.610 | 0.198 | -0.056 | 0.654 | 0.957 | -0.998 | 0.000 |
| TONE_Q | 0.470 | 0.170 | -0.057 | 0.484 | 0.908 | -0.305 | 0.000 |
| TONE_GAP | -0.140 | 0.246 | -0.802 | -0.154 | 0.527 | 0.045 | 0.000 |
| UNC | 3.039 | 1.096 | 0.464 | 2.814 | 6.668 | 0.537 | 0.000 |
| HEDGE | 7.355 | 1.919 | 3.623 | 7.216 | 17.291 | 1.254 | 0.000 |
| EVADE | 0.476 | 0.055 | 0.370 | 0.472 | 0.655 | 0.636 | 0.000 |
| NONANS | 0.059 | 0.078 | 0.000 | 0.000 | 0.350 | 1.471 | 0.000 |
| dTONE | -0.003 | 0.141 | -0.490 | 0.007 | 0.393 | -0.347 | 0.072 |
| dUNC | -0.028 | 1.015 | -2.186 | -0.053 | 2.659 | 0.040 | 0.072 |
| dEVADE | -0.003 | 0.044 | -0.138 | -0.004 | 0.159 | 0.118 | 0.072 |
| NOVEL | 0.060 | 0.032 | 0.017 | 0.053 | 0.171 | 1.034 | 0.018 |
| DIV_QP | 0.177 | 0.054 | 0.093 | 0.164 | 0.399 | 1.374 | 0.000 |
| REL | 0.524 | 0.055 | 0.345 | 0.528 | 0.630 | -0.636 | 0.000 |
| REL_JAC | 0.071 | 0.012 | 0.039 | 0.070 | 0.105 | 0.224 | 0.000 |
| TONE_M | 0.540 | 0.136 | 0.022 | 0.557 | 0.854 | -0.624 | 0.000 |
| SUE | 1.035 | 2.556 | -22.727 | 0.914 | 8.749 | -4.731 | 0.072 |
| DISP | 0.030 | 0.029 | 0.000 | 0.028 | 0.200 | 1.880 | 0.078 |
| COV | 3.417 | 0.200 | 2.996 | 3.401 | 3.932 | 0.516 | 0.000 |
| SIZE | 26.385 | 1.715 | 22.386 | 26.798 | 28.753 | -0.949 | 0.000 |
| BM | -1.996 | 0.843 | -3.931 | -1.802 | -0.449 | -0.399 | 0.000 |
| MOM | 0.375 | 0.500 | -0.525 | 0.313 | 2.679 | 1.535 | 0.000 |
| VOL | 0.214 | 0.105 | 0.062 | 0.179 | 0.670 | 1.506 | 0.000 |
