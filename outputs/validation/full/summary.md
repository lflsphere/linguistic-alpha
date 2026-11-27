# Step 1 on the full-history source (text only; build period 2011-2023 unless stated)

Calls: 24661 build (2011-2023) + 5307 holdout (scored only); Q-A pairs: 563646 build. Firm key gvkey (780 firms). Labels matched: 414 pairs (`data/qa_labels.csv`).

Classifier OOF AUC (01): {"EVADE_CLF": 0.629, "NONANS_CLF": 0.674}

## Agreement with the hand labels (pair level, OOF for the classifiers)
| label | score | n | n_pos | base_rate | auc | auc_ci_lo | auc_ci_hi | kappa | precision | recall |
|---|---|---|---|---|---|---|---|---|---|---|
| evasive | EVADE_CLF | 414 | 61 | 0.147 | 0.629 | 0.553 | 0.704 |  |  |  |
| evasive | REL | 414 | 61 | 0.147 | 0.530 | 0.450 | 0.608 |  |  |  |
| evasive | REL_JAC | 414 | 61 | 0.147 | 0.576 | 0.501 | 0.652 |  |  |  |
| evasive | NONANS_RULE | 414 | 61 | 0.147 | 0.548 | 0.506 | 0.598 | 0.125 | 0.333 | 0.148 |
| evasive | HEDGE_A | 414 | 61 | 0.147 | 0.516 | 0.446 | 0.588 |  |  |  |
| non_answer | NONANS_CLF | 414 | 57 | 0.138 | 0.674 | 0.588 | 0.756 |  |  |  |
| non_answer | NONANS_RULE | 414 | 57 | 0.138 | 0.666 | 0.602 | 0.729 | 0.425 | 0.741 | 0.351 |
| non_answer | REL | 414 | 57 | 0.138 | 0.497 | 0.417 | 0.576 |  |  |  |
| non_answer | REL_JAC | 414 | 57 | 0.138 | 0.587 | 0.513 | 0.659 |  |  |  |
| non_answer | HEDGE_A | 414 | 57 | 0.138 | 0.497 | 0.417 | 0.577 |  |  |  |

## Classifier AUC by year and GICS sector
| by | group | EVADE_CLF_auc | NONANS_CLF_auc | EVADE_CLF_n | NONANS_CLF_n | EVADE_CLF_n_pos | NONANS_CLF_n_pos |
|---|---|---|---|---|---|---|---|
| gsector | 10 | 0.053 | 0.595 | 39.000 | 39.000 | 1.000 | 2.000 |
| gsector | 15 | 0.541 | 0.852 | 39.000 | 39.000 | 2.000 | 3.000 |
| gsector | 20 | 0.800 | 0.598 | 39.000 | 39.000 | 5.000 | 7.000 |
| gsector | 25 | 0.580 | 0.689 | 39.000 | 39.000 | 12.000 | 9.000 |
| gsector | 30 | 0.664 | 0.564 | 39.000 | 39.000 | 4.000 | 4.000 |
| gsector | 35 | 0.603 | 0.665 | 39.000 | 39.000 | 10.000 | 8.000 |
| gsector | 40 | 0.800 | 0.551 | 39.000 | 39.000 | 4.000 | 6.000 |
| gsector | 45 | 0.728 | 0.546 | 39.000 | 39.000 | 7.000 | 3.000 |
| gsector | 50 | 0.585 | 0.911 | 39.000 | 39.000 | 9.000 | 7.000 |
| gsector | 55 | 0.393 | 0.943 | 39.000 | 39.000 | 4.000 | 4.000 |
| gsector | 60 | 0.476 | 0.275 | 24.000 | 24.000 | 3.000 | 4.000 |
| year | 2011 | 0.798 | 0.896 | 30.000 | 30.000 | 4.000 | 6.000 |
| year | 2012 | 0.482 | 0.298 | 30.000 | 30.000 | 2.000 | 4.000 |
| year | 2013 | 0.982 | 0.827 | 30.000 | 30.000 | 2.000 | 4.000 |
| year | 2014 | 0.568 | 0.640 | 30.000 | 30.000 | 3.000 | 5.000 |
| year | 2015 | 0.592 | 0.679 | 30.000 | 30.000 | 5.000 | 3.000 |
| year | 2016 | 0.544 | 0.586 | 33.000 | 33.000 | 7.000 | 5.000 |
| year | 2017 | 0.791 | 0.700 | 33.000 | 33.000 | 7.000 | 3.000 |
| year | 2018 | 0.620 | 0.819 | 33.000 | 33.000 | 8.000 | 4.000 |
| year | 2019 | 0.474 | 0.344 | 33.000 | 33.000 | 4.000 | 3.000 |
| year | 2020 | 0.507 | 0.581 | 33.000 | 33.000 | 5.000 | 2.000 |
| year | 2021 | 0.639 | 0.587 | 33.000 | 33.000 | 9.000 | 11.000 |
| year | 2022 | 0.521 | 0.871 | 33.000 | 33.000 | 5.000 | 4.000 |
| year | 2023 |  | 0.756 | 33.000 | 33.000 | 0.000 | 3.000 |

## Classifier scores: missing rates and correlation with EVADE / NONANS (call level)
| index | call_missing_all | call_missing_dev | pair_missing_all |
|---|---|---|---|
| EVADE_CLF | 0.009 | 0.010 | 0.000 |
| NONANS_CLF | 0.009 | 0.010 | 0.000 |

| score | against | spearman | pearson | n |
|---|---|---|---|---|
| EVADE_CLF | EVADE | -0.246 | -0.263 | 24420 |
| EVADE_CLF | NONANS | 0.501 | 0.613 | 24420 |
| NONANS_CLF | EVADE | -0.060 | -0.138 | 24420 |
| NONANS_CLF | NONANS | 0.949 | 0.977 | 24420 |

## Text-only checks
| check | status | detail |
|---|---|---|
| Delta features use only the previous calls of the same firm (gvkey); history from 2008 | PASS | max |recomputed - stored| = 3.33e-16 on 26921 calls (TONE_M NaNs make the stored rolling mean NaN where the recomputed one is not: 0) |
| NOVEL uses the previous call of the same firm only | PASS | 367 calls with NaN NOVEL (first call of a gvkey since 2008) |
| unsupervised text models fitted without holdout text | PASS | sentence-transformers/multi-qa-MiniLM-L6-cos-v1: pre-trained (no fit on this sample) |
| label classifiers fitted on build-period labels only | PASS | 414 labels, 0 in the holdout; fit period call_date <= 2023-12-31 |
| every labelled pair_id matches a full-source pair | PASS | labels used {'EVADE_CLF': 414, 'NONANS_CLF': 414}; not matched {'EVADE_CLF': 0, 'NONANS_CLF': 0} |
| one call per firm and fiscal period (duplicate_calls: keep_first) | PASS | 111 duplicates dropped; 19 gvkey x fiscal-label repeats left are > 3 days apart (fiscal-year changes) |
| firm key | PASS | gvkey (tickers change over time) |
| lexicon | PASS | MasterDictionary.csv |
| LLM look-ahead | WARN | no generative LLM scoring; the pre-trained encoder receives text with the call's own firm name and years masked (other firms' names are not masked) |
| transcript text truncation | PASS | none: texts come from the CIQ components, not Excel cells |
| holdout coverage (scored, never used for fitting or distributions) | PASS | 5307 holdout calls (2024-2026), 90801 pairs |

## Skipped (need targets, controls or events)
| check | why |
|---|---|
| construct validity against SUE, AR01 and VOL | needs `la targets` / `la controls` |
| Model A distributions and correlations (SUE, DISP, COV, SIZE, BM, MOM, VOL) | needs `la controls` |
| day 0 consistent with the release time; call within Day 0 / Day +1 | needs `la events` (day0 is empty) |
| fiscal period ends before the call | needs I/B/E/S or Compustat period ends joined to the call |
| time-series SUE scale uses prior surprises only | needs `la controls` |
| pre-call analyst consensus timestamps | needs `la controls` |
| split-adjusted consensus rounding (LOWPREC) | needs `la controls` |
| holdout coverage of CAR windows | needs `la targets` |
| alternative abnormal-return benchmark | needs `la targets` |
| manual_read transcript excerpts | excerpts are no longer tracked in outputs/validation/manual_read |
| blind labeling sample | data/qa_label_sample_full.csv already exists (labeller A) |

## Construct validity (text-only expected signs)
| feature | against | expected_sign | spearman | p_value | n | sign_ok |
|---|---|---|---|---|---|---|
| UNC | TONE_M | -1 | -0.304 | 0.000 | 24659 | True |
| EVADE | NONANS | 1 | -0.107 | 0.000 | 24420 | False |
| HEDGE_Q | EVADE | 1 | 0.141 | 0.000 | 24420 | True |
| EVADE | EVADE_CLF | 1 | -0.246 | 0.000 | 24420 | False |
| NONANS | NONANS_CLF | 1 | 0.949 | 0.000 | 24420 | True |

## Alternative measurements
| level | measure_a | measure_b | spearman | pearson | n |
|---|---|---|---|---|---|
| call | TONE_P | TONE_P_PW | 0.893 | 0.890 | 24647 |
| call | TONE_Q | TONE_Q_PW | 0.950 | 0.938 | 24490 |
| call | REL | REL_JAC | 0.545 | 0.543 | 24420 |
| call | EVADE | NONANS | -0.107 | -0.172 | 24420 |
| call | EVADE | EVADE_CLF | -0.246 | -0.263 | 24420 |
| call | NONANS | NONANS_CLF | 0.949 | 0.977 | 24420 |
| call | UNC | HEDGE | 0.195 | 0.159 | 24659 |
| call | UNC_P | UNC_Q | 0.267 | 0.259 | 24485 |
| pair | REL | REL_JAC | 0.540 | 0.496 | 563623 |
| pair | REL | EVADE_CLF | -0.135 | -0.086 | 563646 |
| pair | NONANS_RULE | NONANS_CLF | 0.379 | 0.972 | 563646 |
| pair | EVADE_CLF | NONANS_CLF | 0.536 | 0.641 | 563646 |

## Redundant pairs (|Spearman| > 0.8)
| feature_a | feature_b | spearman |
|---|---|---|
| TONE_P | TONE_M | 0.844 |
| TONE_Q | TONE_M | 0.816 |
| HEDGE | HEDGE_Q | 0.863 |
| EVADE | REL | -1.000 |
| NONANS | NONANS_CLF | 0.949 |

## Distributions
| index | mean | std | min | 50% | max | skew | missing_share |
|---|---|---|---|---|---|---|---|
| TONE_P | 0.433 | 0.275 | -1.000 | 0.470 | 1.000 | -1.039 | 0.001 |
| TONE_Q | 0.320 | 0.244 | -1.000 | 0.342 | 1.000 | -0.546 | 0.007 |
| TONE_GAP | -0.114 | 0.275 | -1.714 | -0.125 | 1.848 | 0.791 | 0.007 |
| UNC | 4.832 | 1.978 | 0.000 | 4.584 | 29.955 | 1.305 | 0.000 |
| HEDGE | 9.075 | 3.626 | 0.000 | 8.594 | 36.246 | 1.101 | 0.000 |
| EVADE | 0.505 | 0.060 | 0.106 | 0.507 | 0.831 | -0.098 | 0.010 |
| NONANS | 0.055 | 0.066 | 0.000 | 0.040 | 0.727 | 1.894 | 0.010 |
| dTONE | 0.004 | 0.163 | -0.906 | 0.013 | 0.766 | -0.418 | 0.067 |
| dUNC | -0.029 | 1.363 | -9.412 | -0.084 | 8.548 | 0.261 | 0.067 |
| dEVADE | -0.002 | 0.045 | -0.403 | -0.003 | 0.352 | 0.054 | 0.090 |
| NOVEL | 0.070 | 0.055 | -0.000 | 0.059 | 0.960 | 6.809 | 0.013 |
| DIV_QP | 0.186 | 0.090 | -0.000 | 0.169 | 0.958 | 4.456 | 0.000 |
| REL | 0.495 | 0.060 | 0.169 | 0.493 | 0.894 | 0.098 | 0.010 |
| REL_JAC | 0.072 | 0.013 | 0.020 | 0.071 | 0.205 | 0.593 | 0.010 |
| TONE_M | 0.382 | 0.217 | -0.667 | 0.404 | 0.929 | -0.571 | 0.000 |
| UNC_P | 5.400 | 3.052 | 0.000 | 4.924 | 45.161 | 1.479 | 0.000 |
| UNC_Q | 4.367 | 2.033 | 0.000 | 4.087 | 19.834 | 0.872 | 0.007 |
| HEDGE_Q | 14.132 | 5.080 | 0.000 | 13.645 | 44.344 | 0.710 | 0.007 |
| EVADE_CLF | 0.146 | 0.022 | 0.040 | 0.144 | 0.296 | 0.821 | 0.010 |
| NONANS_CLF | 0.134 | 0.041 | 0.068 | 0.125 | 0.546 | 1.747 | 0.010 |

## Missing share by year (all years: coverage)
| year | TONE_P | TONE_Q | TONE_GAP | UNC | HEDGE | EVADE | NONANS | dTONE | dUNC | dEVADE | NOVEL | DIV_QP | REL | REL_JAC | TONE_M | UNC_P | UNC_Q | HEDGE_Q | EVADE_CLF | NONANS_CLF |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2011 | 0.002 | 0.014 | 0.015 | 0.001 | 0.001 | 0.016 | 0.016 | 0.360 | 0.360 | 0.396 | 0.043 | 0.000 | 0.016 | 0.016 | 0.001 | 0.002 | 0.014 | 0.014 | 0.016 | 0.016 |
| 2012 | 0.000 | 0.010 | 0.010 | 0.000 | 0.000 | 0.010 | 0.010 | 0.065 | 0.065 | 0.097 | 0.015 | 0.000 | 0.010 | 0.010 | 0.000 | 0.000 | 0.009 | 0.009 | 0.010 | 0.010 |
| 2013 | 0.000 | 0.012 | 0.012 | 0.000 | 0.000 | 0.013 | 0.013 | 0.049 | 0.049 | 0.073 | 0.008 | 0.000 | 0.013 | 0.013 | 0.000 | 0.000 | 0.012 | 0.012 | 0.013 | 0.013 |
| 2014 | 0.001 | 0.015 | 0.016 | 0.000 | 0.000 | 0.018 | 0.018 | 0.041 | 0.041 | 0.077 | 0.010 | 0.000 | 0.018 | 0.018 | 0.000 | 0.001 | 0.015 | 0.015 | 0.018 | 0.018 |
| 2015 | 0.000 | 0.011 | 0.011 | 0.000 | 0.000 | 0.013 | 0.013 | 0.042 | 0.042 | 0.080 | 0.011 | 0.000 | 0.013 | 0.013 | 0.000 | 0.000 | 0.011 | 0.011 | 0.013 | 0.013 |
| 2016 | 0.001 | 0.011 | 0.012 | 0.000 | 0.000 | 0.013 | 0.013 | 0.057 | 0.057 | 0.083 | 0.016 | 0.000 | 0.013 | 0.013 | 0.000 | 0.001 | 0.011 | 0.011 | 0.013 | 0.013 |
| 2017 | 0.001 | 0.007 | 0.008 | 0.000 | 0.000 | 0.011 | 0.011 | 0.058 | 0.058 | 0.085 | 0.015 | 0.000 | 0.011 | 0.011 | 0.000 | 0.001 | 0.007 | 0.007 | 0.011 | 0.011 |
| 2018 | 0.000 | 0.005 | 0.005 | 0.000 | 0.000 | 0.009 | 0.009 | 0.045 | 0.045 | 0.077 | 0.010 | 0.000 | 0.009 | 0.009 | 0.000 | 0.000 | 0.005 | 0.005 | 0.009 | 0.009 |
| 2019 | 0.001 | 0.004 | 0.004 | 0.000 | 0.000 | 0.006 | 0.006 | 0.045 | 0.045 | 0.063 | 0.009 | 0.000 | 0.006 | 0.006 | 0.000 | 0.000 | 0.004 | 0.004 | 0.006 | 0.006 |
| 2020 | 0.001 | 0.001 | 0.002 | 0.000 | 0.000 | 0.005 | 0.005 | 0.039 | 0.039 | 0.050 | 0.010 | 0.000 | 0.005 | 0.005 | 0.000 | 0.000 | 0.001 | 0.001 | 0.005 | 0.005 |
| 2021 | 0.001 | 0.000 | 0.001 | 0.000 | 0.000 | 0.005 | 0.005 | 0.034 | 0.034 | 0.041 | 0.008 | 0.000 | 0.005 | 0.005 | 0.000 | 0.000 | 0.000 | 0.000 | 0.005 | 0.005 |
| 2022 | 0.000 | 0.001 | 0.001 | 0.000 | 0.000 | 0.005 | 0.005 | 0.029 | 0.029 | 0.036 | 0.006 | 0.000 | 0.005 | 0.005 | 0.000 | 0.000 | 0.001 | 0.001 | 0.005 | 0.005 |
| 2023 | 0.002 | 0.001 | 0.003 | 0.000 | 0.000 | 0.004 | 0.004 | 0.027 | 0.027 | 0.037 | 0.008 | 0.000 | 0.004 | 0.004 | 0.000 | 0.001 | 0.001 | 0.001 | 0.004 | 0.004 |
| 2024 | 0.001 | 0.004 | 0.005 | 0.000 | 0.000 | 0.007 | 0.007 | 0.031 | 0.031 | 0.044 | 0.007 | 0.000 | 0.007 | 0.007 | 0.000 | 0.000 | 0.004 | 0.004 | 0.007 | 0.007 |
| 2025 | 0.002 | 0.003 | 0.005 | 0.000 | 0.000 | 0.005 | 0.005 | 0.030 | 0.030 | 0.038 | 0.008 | 0.000 | 0.005 | 0.005 | 0.000 | 0.002 | 0.003 | 0.003 | 0.005 | 0.005 |
| 2026 | 0.005 | 0.001 | 0.006 | 0.000 | 0.000 | 0.007 | 0.007 | 0.041 | 0.041 | 0.050 | 0.012 | 0.000 | 0.007 | 0.007 | 0.000 | 0.005 | 0.001 | 0.001 | 0.007 | 0.007 |

## Transcript structure by year
| year | words_prepared | words_answer | words_question | n_questions | n_pairs | pair_coverage | qa_share | q_count_match | no_qa | boiler_share | version_final | call_time_missing | calls | firms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2011 | 3440.073 | 3775.455 | 1505.482 | 29.363 | 28.528 | 0.961 | 0.524 | 1.000 | 0.016 | 0.044 | 0.699 | 0.000 | 1775 | 484 |
| 2012 | 3442.096 | 3851.510 | 1564.770 | 28.203 | 27.848 | 0.979 | 0.529 | 1.000 | 0.010 | 0.044 | 0.813 | 0.002 | 1839 | 496 |
| 2013 | 3373.081 | 3819.363 | 1531.198 | 26.578 | 26.404 | 0.981 | 0.532 | 1.000 | 0.013 | 0.043 | 0.867 | 0.001 | 1903 | 497 |
| 2014 | 3360.445 | 3865.324 | 1520.671 | 25.523 | 25.339 | 0.975 | 0.535 | 1.000 | 0.018 | 0.044 | 0.844 | 0.002 | 1898 | 500 |
| 2015 | 3357.289 | 3919.905 | 1514.838 | 24.795 | 24.638 | 0.981 | 0.540 | 1.000 | 0.013 | 0.040 | 0.852 | 0.001 | 1854 | 502 |
| 2016 | 3278.175 | 3945.149 | 1492.445 | 24.054 | 23.876 | 0.980 | 0.545 | 1.000 | 0.013 | 0.043 | 0.898 | 0.002 | 1894 | 503 |
| 2017 | 3214.002 | 3961.956 | 1469.279 | 23.268 | 23.117 | 0.983 | 0.550 | 1.000 | 0.011 | 0.043 | 0.877 | 0.002 | 1922 | 513 |
| 2018 | 3283.323 | 3947.495 | 1462.509 | 22.453 | 22.293 | 0.985 | 0.546 | 1.000 | 0.009 | 0.043 | 0.856 | 0.000 | 1922 | 504 |
| 2019 | 3163.391 | 3966.388 | 1449.414 | 21.641 | 21.500 | 0.989 | 0.556 | 1.000 | 0.006 | 0.043 | 0.863 | 0.000 | 1935 | 501 |
| 2020 | 3495.513 | 4130.183 | 1412.130 | 19.885 | 19.695 | 0.986 | 0.543 | 1.000 | 0.005 | 0.043 | 0.869 | 0.000 | 1909 | 503 |
| 2021 | 3293.698 | 4044.263 | 1364.528 | 18.747 | 18.512 | 0.984 | 0.554 | 1.000 | 0.005 | 0.045 | 0.900 | 0.000 | 1936 | 499 |
| 2022 | 3209.856 | 4019.022 | 1354.656 | 18.714 | 18.433 | 0.981 | 0.558 | 1.000 | 0.005 | 0.046 | 0.899 | 0.000 | 1924 | 496 |
| 2023 | 3119.291 | 3947.089 | 1305.989 | 18.093 | 17.833 | 0.983 | 0.560 | 1.000 | 0.004 | 0.049 | 0.893 | 0.001 | 1950 | 503 |
| 2024 | 3021.525 | 3938.971 | 1299.838 | 17.783 | 17.565 | 0.983 | 0.568 | 1.000 | 0.007 | 0.052 | 0.932 | 0.002 | 1938 | 500 |
| 2025 | 2987.757 | 4019.224 | 1288.740 | 17.289 | 17.081 | 0.984 | 0.576 | 1.000 | 0.005 | 0.055 | 0.924 | 0.001 | 1939 | 502 |
| 2026 | 2964.255 | 4095.341 | 1259.737 | 16.681 | 16.531 | 0.985 | 0.582 | 1.000 | 0.007 | 0.057 | 0.902 | 0.000 | 1430 | 499 |
