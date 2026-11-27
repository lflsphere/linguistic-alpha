# la events: attrition and checks

Built by `la events` (`src/linguistic_alpha/events/build.py`) from `data/interim/transcript_versions` and `data/raw/sp500_pit_2008`. Written 2026-11-27T18:00:12+00:00.

Switches: anchor `release`, placeholder_time_last `True`, duplicate_window_days `3`, release_match_days `[-3, 1]`, max_call_lag_sessions `1`, sample 2011-01-01 to 2026-08-31 (Day 0).

## Attrition (each call gets the first reason it fails, in this order)

| Step | Calls before | Calls after | Dropped |
|---|---|---|---|
| Earnings calls (event type) | 51,598 | 51,598 | 0 |
| Day 0 in sample.start .. sample.end (call date if no release) | 51,598 | 42,961 | 8,637 |
| CIQ companyid -> gvkey (date-valid) | 42,961 | 42,908 | 53 |
| gvkey -> permno (CCM LU/LC, date-valid) | 42,908 | 42,055 | 853 |
| S&P 500 member on Day 0 (call date if no release) | 42,055 | 30,763 | 11,292 |
| permno -> I/B/E/S ticker (score <= 2, date-valid) | 30,763 | 30,742 | 21 |
| I/B/E/S EPS release matched to the call (events.release_match_days) | 30,742 | 30,626 | 116 |
| Transcript version by the cutoff (no_version_before_cutoff: drop) | 30,626 | 30,015 | 611 |
| One call per firm and fiscal period (duplicate_calls: keep_first) | 30,015 | 29,854 | 161 |
| One call per I/B/E/S release (keep_first) | 29,854 | 29,854 | 0 |
| Call in Day 0 or Day +1 (events.max_call_lag_sessions) | 29,854 | 29,819 | 35 |
| Not in sample.exclude_gics_industries | 29,819 | 29,819 | 0 |
| At least sample.min_prior_return_days CRSP returns before Day 0 | 29,819 | 29,799 | 20 |

## Links on all calls (before the sample-period filter)

| Link (all calls, 2008 on) | Calls before | Calls after |
|---|---|---|
| CIQ companyid -> gvkey (date-valid) | 51,598 | 51,520 |
| gvkey -> permno (CCM LU/LC, date-valid) | 51,520 | 50,564 |
| permno -> I/B/E/S ticker (score <= 2, date-valid) | 50,564 | 50,501 |

## Checks (calls inside the sample period)

- Release time missing (anntims blank or 00:00:00, Day 0 = next session): 0 matched calls, 0 kept events.
- Call lag of 2+ sessions after the release: 12; call before the release's Day 0 (lag < 0): 96.
- No I/B/E/S ticker: 21 S&P 500 calls (exclusion reason); no release in the match window: 116.
- Calls with more than one member permno on Day 0: 119 (choice: {'primary': 116, 'mktcap': 3}).
- CIQ 00:00:00 UTC placeholder call times among kept events: 47.
- Kept events with no GICS sector on Day 0: 23.
- Joint calls (one keydevid, two CIQ companies; both are candidates): 6.

## Kept events by call lag

| call_lag_sessions | events |
|---|---|
| 0 | 29,661 |
| 1 | 138 |

## Kept events by release date minus call date

| release anndats - call date (days) | events |
|---|---|
| -3 | 3 |
| -2 | 7 |
| -1 | 4,694 |
| 0 | 25,092 |
| 1 | 3 |

## Events per year

| year (Day 0) | events |
|---|---|
| 2011 | 1,761 |
| 2012 | 1,823 |
| 2013 | 1,897 |
| 2014 | 1,892 |
| 2015 | 1,847 |
| 2016 | 1,883 |
| 2017 | 1,913 |
| 2018 | 1,919 |
| 2019 | 1,925 |
| 2020 | 1,899 |
| 2021 | 1,925 |
| 2022 | 1,915 |
| 2023 | 1,942 |
| 2024 | 1,926 |
| 2025 | 1,928 |
| 2026 | 1,404 |

## Events per GICS sector

| gsector | events |
|---|---|
| 10 | 1,881 |
| 15 | 1,696 |
| 20 | 4,123 |
| 25 | 4,192 |
| 30 | 2,215 |
| 35 | 3,609 |
| 40 | 4,193 |
| 45 | 4,219 |
| 50 | 834 |
| 55 | 1,642 |
| 60 | 1,172 |
| missing | 23 |

## Excluded calls by reason (all years)

| exclusion_reason | events |
|---|---|
| not_sp500_member | 11,292 |
| outside_sample_period | 8,637 |
| no_permno | 853 |
| no_transcript_version | 611 |
| duplicate_call | 161 |
| no_ibes_release | 116 |
| no_gvkey | 53 |
| call_lag | 35 |
| no_ibes_ticker | 21 |
| short_return_history | 20 |

## Duplicate calls dropped (161, all years)

<details><summary>All dropped duplicates</summary>

| keydevid | exclusion_reason | duplicate_of_keydevid | duplicate_release_of_keydevid | gvkey | companyname | headline | call_date | call_time_utc | fiscal_label |
|---|---|---|---|---|---|---|---|---|---|
| 118,543,116 | duplicate_call | 118,542,999 | <NA> | 004839 | Ford Motor Company | Ford Motor Co., Q4 2010 Earnings Call, Jan 28, 2011 | 2011-01-28 | 16:00:00 | Q4 2010 |
| 118,546,825 | duplicate_call | 62,649,614 | <NA> | 003505 | Molson Coors Beverage Company | Molson Coors Brewing Company, Q4 2010 Earnings Call, Feb 10, 2011 | 2011-02-10 | 19:00:00 | Q4 2010 |
| 130,287,464 | duplicate_call | 130,287,700 | <NA> | 004839 | Ford Motor Company | Ford Motor Co., Q1 2011 Earnings Call, Apr 26, 2011 | 2011-04-26 | 15:00:00 | Q1 2011 |
| 130,292,330 | duplicate_call | 98,234,885 | <NA> | 003505 | Molson Coors Beverage Company | Molson Coors Brewing Company, Q1 2011 Earnings Call, May 03, 2011 | 2011-05-03 | 18:00:00 | Q1 2011 |
| 137,660,050 | duplicate_call | 137,660,011 | <NA> | 004839 | Ford Motor Company | Ford Motor Co., Q2 2011 Earnings Call, Jul 26, 2011 | 2011-07-26 | 15:00:00 | Q2 2011 |
| 137,335,433 | duplicate_call | 137,395,438 | <NA> | 004885 | Franklin Templeton Inc. | Franklin Resources Inc., Q3 2011 Earnings Call, Aug 02, 2011 | 2011-08-02 | 20:30:00 | Q3 2011 |
| 141,871,103 | duplicate_call | 141,870,877 | <NA> | 004839 | Ford Motor Company | Ford Motor Co., Q3 2011 Earnings Call, Oct 26, 2011 | 2011-10-26 | 15:00:00 | Q3 2011 |
| 141,522,446 | duplicate_call | 141,522,431 | <NA> | 004885 | Franklin Templeton Inc. | Franklin Resources Inc., Q4 2011 Earnings Call, Oct 27, 2011 | 2011-10-27 | 00:00:00 | Q4 2011 |
| 141,543,967 | duplicate_call | 98,234,926 | <NA> | 003505 | Molson Coors Beverage Company | Molson Coors Brewing Company, Q3 2011 Earnings Call, Nov 02, 2011 | 2011-11-02 | 18:00:00 | Q3 2011 |
| 146,357,156 | duplicate_call | 146,356,887 | <NA> | 004839 | Ford Motor Company | Ford Motor Co., Q4 2011 Earnings Call, Jan 27, 2012 | 2012-01-27 | 16:00:00 | Q4 2011 |
| 145,980,875 | duplicate_call | 145,978,416 | <NA> | 004885 | Franklin Templeton Inc. | Franklin Resources Inc., Q1 2012 Earnings Call, Feb 01, 2012 | 2012-02-01 | 00:00:00 | Q1 2012 |
| 146,297,746 | duplicate_call | 98,235,005 | <NA> | 003505 | Molson Coors Beverage Company | Molson Coors Brewing Company, Q4 2011 Earnings Call, Feb 16, 2012 | 2012-02-16 | 19:00:00 | Q4 2011 |
| 184,476,420 | duplicate_call | 184,480,119 | <NA> | 004885 | Franklin Templeton Inc. | Franklin Resources Inc., Q2 2012 Pre Recorded Earnings Call, May 02, 2012 | 2012-05-02 | 00:00:00 | Q2 2012 |
| 193,455,454 | duplicate_call | 146,297,560 | <NA> | 003505 | Molson Coors Beverage Company | Molson Coors Brewing Company, Q1 2012 Earnings Call, May 08, 2012 | 2012-05-08 | 18:00:00 | Q1 2012 |
| 214,174,243 | duplicate_call | 214,174,301 | <NA> | 004885 | Franklin Templeton Inc. | Franklin Resources Inc., Q3 2012 Pre Recorded Earnings Call, Jul 30, 2012 | 2012-07-30 | 00:00:00 | Q3 2012 |
| 222,841,106 | duplicate_call | 222,841,092 | <NA> | 004885 | Franklin Templeton Inc. | Franklin Resources Inc., Q4 2012 Pre Recorded Earnings Call, Oct 25, 2012 | 2012-10-25 | 00:00:00 | Q4 2012 |
| 223,357,972 | duplicate_call | 146,297,634 | <NA> | 003505 | Molson Coors Beverage Company | Molson Coors Brewing Company, Q3 2012 Earnings Call, Nov 07, 2012 | 2012-11-07 | 19:00:00 | Q3 2012 |
| 225,974,384 | duplicate_call | 225,974,232 | <NA> | 004885 | Franklin Templeton Inc. | Franklin Resources Inc., Q1 2013 Pre Recorded Earnings Call, Feb 01, 2013 | 2013-02-01 | 00:00:00 | Q1 2013 |
| 235,948,626 | duplicate_call | 235,951,760 | <NA> | 004885 | Franklin Templeton Inc. | Franklin Resources Inc., Q2 2013 Pre Recorded Earnings Call, Apr 30, 2013 | 2013-04-30 | 00:00:00 | Q2 2013 |
| 236,745,109 | duplicate_call | 146,297,648 | <NA> | 003505 | Molson Coors Beverage Company | Molson Coors Brewing Company, Q1 2013 Earnings Call, May 07, 2013 | 2013-05-07 | 18:00:00 | Q1 2013 |
| 243,693,032 | duplicate_call | 242,420,083 | <NA> | 004885 | Franklin Templeton Inc. | Franklin Resources Inc., Q3 2013 Pre Recorded Earnings Call, Jul 29, 2013 | 2013-07-29 | 00:00:00 | Q3 2013 |
| 243,199,591 | duplicate_call | 146,297,665 | <NA> | 003505 | Molson Coors Beverage Company | Molson Coors Brewing Company, Q2 2013 Earnings Call, Aug 06, 2013 | 2013-08-06 | 18:00:00 | Q2 2013 |
| 247,855,378 | duplicate_call | 247,855,433 | <NA> | 004885 | Franklin Templeton Inc. | Franklin Resources Inc., Q4 2013 Pre Recorded Earnings Call, Oct 24, 2013 | 2013-10-24 | 00:00:00 | Q4 2013 |
| 247,987,514 | duplicate_call | 146,297,690 | <NA> | 003505 | Molson Coors Beverage Company | Molson Coors Brewing Company, Q3 2013 Earnings Call, Nov 06, 2013 | 2013-11-06 | 19:00:00 | Q3 2013 |
| 253,997,936 | duplicate_call | 253,997,954 | <NA> | 004885 | Franklin Templeton Inc. | Franklin Resources Inc., Q1 2014 Pre Recorded Earnings Call, Jan 30, 2014 | 2014-01-30 | 00:00:00 | Q1 2014 |
| 254,461,727 | duplicate_call | 146,297,699 | <NA> | 003505 | Molson Coors Beverage Company | Molson Coors Brewing Company, Q4 2013 Earnings Call, Feb 13, 2014 | 2014-02-13 | 19:00:00 | Q4 2013 |
| 261,915,714 | duplicate_call | 261,915,739 | <NA> | 004885 | Franklin Templeton Inc. | Franklin Resources Inc., Q2 2014 Pre Recorded Earnings Call, Apr 28, 2014 | 2014-04-28 | 00:00:00 | Q2 2014 |
| 261,905,182 | duplicate_call | 248,059,429 | <NA> | 003505 | Molson Coors Beverage Company | Molson Coors Brewing Company, Q1 2014 Earnings Call, May 07, 2014 | 2014-05-07 | 18:00:00 | Q1 2014 |
| 269,353,081 | duplicate_call | 268,103,075 | <NA> | 004885 | Franklin Templeton Inc. | Franklin Resources Inc., Q3 2014 Pre Recorded Earnings Call, Jul 30, 2014 | 2014-07-30 | 00:00:00 | Q3 2014 |
| 268,171,074 | duplicate_call | 248,059,743 | <NA> | 003505 | Molson Coors Beverage Company | Molson Coors Brewing Company, Q2 2014 Earnings Call, Aug 06, 2014 | 2014-08-06 | 18:00:00 | Q2 2014 |
| 274,265,406 | duplicate_call | 274,265,213 | <NA> | 004885 | Franklin Templeton Inc. | Franklin Resources Inc., Q4 2014 Pre Recorded Earnings Call, Oct 27, 2014 | 2014-10-27 | 00:00:00 | Q4 2014 |
| 248,059,782 | duplicate_call | 274,472,171 | <NA> | 003505 | Molson Coors Beverage Company | Molson Coors Brewing Company, Q3 2014 Earnings Call, Nov 06, 2014 | 2014-11-06 | 18:00:00 | Q3 2014 |
| 281,512,568 | duplicate_call | 281,512,590 | <NA> | 004885 | Franklin Templeton Inc. | Franklin Resources, Inc., Q1 2015 Pre Recorded Earnings Call, Jan 30, 2015 | 2015-01-30 | 00:00:00 | Q1 2015 |
| 281,884,033 | duplicate_call | 248,335,654 | <NA> | 003505 | Molson Coors Beverage Company | Molson Coors Brewing Company, Q4 2014 Earnings Call, Feb 10, 2015 | 2015-02-10 | 18:00:00 | Q4 2014 |
| 291,347,622 | duplicate_call | 291,347,258 | <NA> | 004885 | Franklin Templeton Inc. | Franklin Resources, Inc., Q2 2015 Pre Recorded Earnings Call, Apr 29, 2015 | 2015-04-29 | 00:00:00 | Q2 2015 |
| 290,797,780 | duplicate_call | 248,335,709 | <NA> | 003505 | Molson Coors Beverage Company | Molson Coors Brewing Company, Q1 2015 Earnings Call, May 07, 2015 | 2015-05-07 | 17:00:00 | Q1 2015 |
| 305,856,031 | duplicate_call | 305,855,823 | <NA> | 004885 | Franklin Templeton Inc. | Franklin Resources, Inc., Q3 2015 Pre Recorded Earnings Call, Jul 29, 2015 | 2015-07-29 | 00:00:00 | Q3 2015 |
| 307,200,857 | duplicate_call | 248,335,745 | <NA> | 003505 | Molson Coors Beverage Company | Molson Coors Brewing Company, Q2 2015 Earnings Call, Aug 06, 2015 | 2015-08-06 | 17:00:00 | Q2 2015 |
| 313,049,944 | duplicate_call | 313,049,987 | <NA> | 004885 | Franklin Templeton Inc. | Franklin Resources, Inc., Q4 2015 Pre Recorded Earnings Call, Oct 22, 2015 | 2015-10-22 | 00:00:00 | Q4 2015 |
| 313,260,434 | duplicate_call | 248,335,759 | <NA> | 003505 | Molson Coors Beverage Company | Molson Coors Brewing Company, Q3 2015 Earnings Call, Nov 05, 2015 | 2015-11-05 | 18:00:00 | Q3 2015 |
| 322,094,111 | duplicate_call | 322,083,503 | <NA> | 004885 | Franklin Templeton Inc. | Franklin Resources, Inc., Q1 2016 Pre Recorded Earnings Call, Feb 03, 2016 | 2016-02-03 | 00:00:00 | Q1 2016 |
| 330,709,118 | duplicate_call | 330,709,185 | <NA> | 004885 | Franklin Templeton Inc. | Franklin Resources, Inc., Q2 2016 Pre Recorded Earnings Call, Apr 27, 2016 | 2016-04-27 | 00:00:00 | Q2 2016 |
| 330,118,935 | duplicate_call | 312,033,274 | <NA> | 003505 | Molson Coors Beverage Company | Molson Coors Brewing Company, Q1 2016 Earnings Call, May 03, 2016 | 2016-05-03 | 17:00:00 | Q1 2016 |
| 373,871,223 | duplicate_call | 373,866,245 | <NA> | 004885 | Franklin Templeton Inc. | Franklin Resources, Inc., Q3 2016 Pre Recorded Earnings Call, Jul 28, 2016 | 2016-07-28 | 00:00:00 | Q3 2016 |
| 378,657,075 | duplicate_call | 312,033,288 | <NA> | 003505 | Molson Coors Beverage Company | Molson Coors Brewing Company, Q2 2016 Earnings Call, Aug 02, 2016 | 2016-08-02 | 17:00:00 | Q2 2016 |
| 404,558,490 | duplicate_call | 404,558,514 | <NA> | 004885 | Franklin Templeton Inc. | Franklin Resources, Inc., Q4 2016 Pre Recorded Earnings Call, Oct 26, 2016 | 2016-10-26 | 00:00:00 | Q4 2016 |
| 417,110,086 | duplicate_call | 312,033,319 | <NA> | 003505 | Molson Coors Beverage Company | Molson Coors Brewing Company, Q4 2016 Earnings Call, Feb 14, 2017 | 2017-02-14 | 18:00:00 | Q4 2016 |
| 429,438,581 | duplicate_call | 424,805,651 | <NA> | 003505 | Molson Coors Beverage Company | Molson Coors Brewing Company, Q1 2017 Earnings Call, May 03, 2017 | 2017-05-03 | 17:00:00 | Q1 2017 |
| 548,259,866 | duplicate_call | 552,839,473 | <NA> | 133768 | MetLife, Inc. | MetLife, Inc., Q4 2017 Earnings Call, Feb 14, 2018 | 2018-02-14 | 13:00:00 | Q4 2017 |
| 558,239,654 | duplicate_call | 558,238,866 | <NA> | 008463 | Pentair plc | Pentair plc, Q1 2018 Earnings Call, Apr 19, 2018 | 2018-04-19 | 12:45:00 | Q1 2018 |
| 570,749,046 | duplicate_call | 558,675,126 | <NA> | 008402 | Paychex, Inc. | Paychex, Inc., Q4 2018 Earnings Call, Jun 27, 2018 | 2018-06-27 | 15:00:00 | Q4 2018 |
| 626,254,666 | duplicate_call | 631,579,850 | <NA> | 133768 | MetLife, Inc. | MetLife, Inc., Q2 2019 Earnings Call, Aug 01, 2019 | 2019-08-01 | 13:00:00 | Q2 2019 |
| 665,836,898 | duplicate_call | 656,327,812 | <NA> | 008479 | PepsiCo, Inc. | PepsiCo, Inc., Q1 2020 Pre Recorded Earnings Call, Apr 28, 2020 | 2020-04-28 | 00:00:00 | Q1 2020 |
| 664,441,291 | duplicate_call | 664,440,950 | <NA> | 162129 | The Mosaic Company | The Mosaic Company, Q1 2020 Pre Recorded Earnings Call, May 05, 2020 | 2020-05-05 | 00:00:00 | Q1 2020 |
| 668,843,958 | duplicate_call | 668,842,382 | <NA> | 002663 | The Campbell's Company | Campbell Soup Company, Q3 2020 Pre Recorded Earnings Call, Jun 03, 2020 | 2020-06-03 | 00:00:00 | Q3 2020 |
| 674,201,331 | duplicate_call | 671,060,180 | <NA> | 005071 | General Mills, Inc. | General Mills, Inc., Q4 2020 Pre Recorded Earnings Call, Jul 01, 2020 | 2020-07-01 | 00:00:00 | Q4 2020 |
| 669,317,587 | duplicate_call | 669,319,808 | <NA> | 008479 | PepsiCo, Inc. | PepsiCo, Inc., Q2 2020 Pre Recorded Earnings Call, Jul 13, 2020 | 2020-07-13 | 00:00:00 | Q2 2020 |
| 673,591,179 | duplicate_call | 673,589,647 | <NA> | 005597 | The Hershey Company | The Hershey Company, Q2 2020 Pre Recorded Earnings Call, Jul 23, 2020 | 2020-07-23 | 00:00:00 | Q2 2020 |
| 684,519,754 | duplicate_call | 684,519,809 | <NA> | 002663 | The Campbell's Company | Campbell Soup Company, Q4 2020 Pre Recorded Earnings Call, Sep 03, 2020 | 2020-09-03 | 00:00:00 | Q4 2020 |
| 689,571,624 | duplicate_call | 686,864,469 | <NA> | 005071 | General Mills, Inc. | General Mills, Inc., Q1 2021 Pre Recorded Earnings Call, Sep 23, 2020 | 2020-09-23 | 00:00:00 | Q1 2021 |
| 685,485,009 | duplicate_call | 685,485,172 | <NA> | 008479 | PepsiCo, Inc. | PepsiCo, Inc., Q3 2020 Pre Recorded Earnings Call, Oct 01, 2020 | 2020-10-01 | 00:00:00 | Q3 2020 |
| 691,091,678 | duplicate_call | 694,033,977 | <NA> | 133768 | MetLife, Inc. | MetLife, Inc., Q3 2020 Earnings Call, Nov 05, 2020 | 2020-11-05 | 14:00:00 | Q3 2020 |
| 694,178,881 | duplicate_call | 690,028,802 | <NA> | 005597 | The Hershey Company | The Hershey Company, Q3 2020 Pre Recorded Earnings Call, Nov 06, 2020 | 2020-11-06 | 00:00:00 | Q3 2020 |
| 698,478,484 | duplicate_call | 695,486,417 | <NA> | 005071 | General Mills, Inc. | General Mills, Inc., Q2 2021 Pre Recorded Earnings Call, Dec 17, 2020 | 2020-12-17 | 00:00:00 | Q2 2021 |
| 703,270,895 | duplicate_call | 700,489,027 | <NA> | 005597 | The Hershey Company | The Hershey Company, Q4 2020 Pre Recorded Earnings Call, Feb 04, 2021 | 2021-02-04 | 00:00:00 | Q4 2020 |
| 709,029,722 | duplicate_call | 705,555,356 | <NA> | 005071 | General Mills, Inc. | General Mills, Inc., Q3 2021 Pre Recorded Earnings Call, Mar 24, 2021 | 2021-03-24 | 00:00:00 | Q3 2021 |
| 713,098,656 | duplicate_call | 710,025,406 | <NA> | 005597 | The Hershey Company | The Hershey Company, Q1 2021 Pre Recorded Earnings Call, Apr 29, 2021 | 2021-04-29 | 00:00:00 | Q1 2021 |
| 717,164,944 | duplicate_call | 717,162,423 | <NA> | 005071 | General Mills, Inc. | General Mills, Inc., Q4 2021 Pre Recorded Earnings Call, Jun 30, 2021 | 2021-06-30 | 00:00:00 | Q4 2021 |
| 1,675,574,598 | duplicate_call | 1,673,266,032 | <NA> | 005597 | The Hershey Company | The Hershey Company, Q2 2021 Pre Recorded Earnings Call, Jul 29, 2021 | 2021-07-29 | 00:00:00 | Q2 2021 |
| 1,676,352,902 | duplicate_call | 1,676,352,875 | <NA> | 009777 | The J. M. Smucker Company | The J. M. Smucker Company, Q1 2022 Pre Recorded Earnings Call, Aug 26, 2021 | 2021-08-26 | 00:00:00 | Q1 2022 |
| 1,680,023,928 | duplicate_call | 1,679,929,139 | <NA> | 005071 | General Mills, Inc. | General Mills, Inc., Q1 2022 Pre Recorded Earnings Call, Sep 22, 2021 | 2021-09-22 | 00:00:00 | Q1 2022 |
| 1,684,347,080 | duplicate_call | 1,684,346,852 | <NA> | 005568 | The Kraft Heinz Company | The Kraft Heinz Company, Q3 2021 Pre Recorded Earnings Call, Oct 27, 2021 | 2021-10-27 | 00:00:00 | Q3 2021 |
| 1,686,390,448 | duplicate_call | 1,684,379,604 | <NA> | 005597 | The Hershey Company | The Hershey Company, Q3 2021 Pre Recorded Earnings Call, Oct 28, 2021 | 2021-10-28 | 00:00:00 | Q3 2021 |
| 1,686,781,442 | duplicate_call | 1,683,344,097 | <NA> | 003121 | The Clorox Company | The Clorox Company, Q1 2022 Pre Recorded Earnings Call, Nov 01, 2021 | 2021-11-01 | 00:00:00 | Q1 2022 |
| 1,686,881,879 | duplicate_call | 1,679,748,069 | <NA> | 009777 | The J. M. Smucker Company | The J. M. Smucker Company, Q2 2022 Pre Recorded Earnings Call, Nov 23, 2021 | 2021-11-23 | 00:00:00 | Q2 2022 |
| 1,760,204,955 | duplicate_call | 1,760,204,891 | <NA> | 005071 | General Mills, Inc. | General Mills, Inc., Q2 2022 Pre Recorded Earnings Call, Dec 21, 2021 | 2021-12-21 | 00:00:00 | Q2 2022 |
| 1,769,743,460 | duplicate_call | 1,763,949,678 | <NA> | 005597 | The Hershey Company | The Hershey Company, Q4 2021 Pre Recorded Earnings Call, Feb 03, 2022 | 2022-02-03 | 00:00:00 | Q4 2021 |
| 1,763,149,697 | duplicate_call | 1,763,149,831 | <NA> | 003121 | The Clorox Company | The Clorox Company, Q2 2022 Pre Recorded Earnings Call, Feb 03, 2022 | 2022-02-03 | 00:00:00 | Q2 2022 |
| 1,764,635,357 | duplicate_call | 1,764,635,295 | <NA> | 005568 | The Kraft Heinz Company | The Kraft Heinz Company, Q4 2021 Pre Recorded Earnings Call, Feb 16, 2022 | 2022-02-16 | 00:00:00 | Q4 2021 |
| 1,770,248,551 | duplicate_call | 1,759,588,576 | <NA> | 009777 | The J. M. Smucker Company | The J. M. Smucker Company, Q3 2022 Pre Recorded Earnings Call, Mar 01, 2022 | 2022-03-01 | 00:00:00 | Q3 2022 |
| 1,773,231,639 | duplicate_call | 1,773,144,779 | <NA> | 005071 | General Mills, Inc. | General Mills, Inc., Q3 2022 Pre Recorded Earnings Call, Mar 23, 2022 | 2022-03-23 | 00:00:00 | Q3 2022 |
| 1,778,704,187 | duplicate_call | 1,773,655,013 | <NA> | 114628 | The Goldman Sachs Group, Inc. | The Goldman Sachs Group, Inc., Q1 2022 Pre Recorded Earnings Call, Apr 14, 2022 | 2022-04-14 | 00:00:00 | Q1 2022 |
| 1,777,110,324 | duplicate_call | 1,777,110,363 | <NA> | 005568 | The Kraft Heinz Company | The Kraft Heinz Company, Q1 2022 Pre Recorded Earnings Call, Apr 27, 2022 | 2022-04-27 | 00:00:00 | Q1 2022 |
| 1,780,869,914 | duplicate_call | 1,777,105,743 | <NA> | 005597 | The Hershey Company | The Hershey Company, Q1 2022 Pre Recorded Earnings Call, Apr 28, 2022 | 2022-04-28 | 00:00:00 | Q1 2022 |
| 1,780,966,794 | duplicate_call | 1,776,953,182 | <NA> | 003121 | The Clorox Company | The Clorox Company, Q3 2022 Pre Recorded Earnings Call, May 02, 2022 | 2022-05-02 | 00:00:00 | Q3 2022 |
| 1,777,273,705 | duplicate_call | 1,781,632,655 | <NA> | 133768 | MetLife, Inc. | MetLife, Inc., Q1 2022 Earnings Call, May 05, 2022 | 2022-05-05 | 13:00:00 | Q1 2022 |
| 1,783,615,448 | duplicate_call | 1,782,009,583 | <NA> | 009777 | The J. M. Smucker Company | The J. M. Smucker Company, Q4 2022 Pre Recorded Earnings Call, Jun 07, 2022 | 2022-06-07 | 00:00:00 | Q4 2022 |
| 1,785,257,882 | duplicate_call | 1,785,257,860 | <NA> | 005071 | General Mills, Inc. | General Mills, Inc., Q4 2022 Pre Recorded Earnings Call, Jun 29, 2022 | 2022-06-29 | 00:00:00 | Q4 2022 |
| 1,785,263,519 | duplicate_call | 1,785,168,411 | <NA> | 007343 | Micron Technology, Inc. | Micron Technology, Inc., Q3 2022 Earnings Call, Jun 30, 2022 | 2022-06-30 | 22:00:00 | Q3 2022 |
| 1,789,593,261 | duplicate_call | 1,789,592,994 | <NA> | 005568 | The Kraft Heinz Company | The Kraft Heinz Company, Q2 2022 Pre Recorded Earnings Call, Jul 27, 2022 | 2022-07-27 | 00:00:00 | Q2 2022 |
| 1,793,528,262 | duplicate_call | 1,789,861,981 | <NA> | 005597 | The Hershey Company | The Hershey Company, Q2 2022 Pre Recorded Earnings Call, Jul 28, 2022 | 2022-07-28 | 00:00:00 | Q2 2022 |
| 1,794,347,464 | duplicate_call | 1,794,031,230 | <NA> | 009777 | The J. M. Smucker Company | The J. M. Smucker Company, Q1 2023 Earnings Call, Aug 23, 2022 | 2022-08-23 | 13:00:00 | Q1 2023 |
| 1,799,912,234 | duplicate_call | 1,798,756,366 | <NA> | 005071 | General Mills, Inc. | General Mills, Inc., Q1 2023 Pre Recorded Earnings Call, Sep 21, 2022 | 2022-09-21 | 00:00:00 | Q1 2023 |
| 1,803,043,168 | duplicate_call | 1,803,043,198 | <NA> | 005568 | The Kraft Heinz Company | The Kraft Heinz Company, Q3 2022 Pre Recorded Earnings Call, Oct 26, 2022 | 2022-10-26 | 00:00:00 | Q3 2022 |
| 1,814,136,926 | duplicate_call | 1,803,043,191 | <NA> | 005597 | The Hershey Company | The Hershey Company, Q3 2022 Pre Recorded Earnings Call, Nov 04, 2022 | 2022-11-04 | 00:00:00 | Q3 2022 |
| 1,813,344,622 | duplicate_call | 1,813,253,120 | <NA> | 009777 | The J. M. Smucker Company | The J. M. Smucker Company, Q2 2023 Pre Recorded Earnings Call, Nov 21, 2022 | 2022-11-21 | 00:00:00 | Q2 2023 |
| 1,816,852,237 | duplicate_call | 1,816,731,153 | <NA> | 005071 | General Mills, Inc. | General Mills, Inc., Q2 2023 Pre Recorded Earnings Call, Dec 20, 2022 | 2022-12-20 | 00:00:00 | Q2 2023 |
| 1,819,409,582 | duplicate_call | 1,804,934,445 | <NA> | 007343 | Micron Technology, Inc. | Micron Technology, Inc., Q1 2023 Earnings Call, Dec 21, 2022 | 2022-12-21 | 23:00:00 | Q1 2023 |
| 1,823,950,133 | duplicate_call | 1,820,840,796 | <NA> | 005597 | The Hershey Company | The Hershey Company, Q4 2022 Pre Recorded Earnings Call, Feb 02, 2023 | 2023-02-02 | 00:00:00 | Q4 2022 |
| 1,823,040,476 | duplicate_call | 1,823,040,565 | <NA> | 005568 | The Kraft Heinz Company | The Kraft Heinz Company, Q4 2022 Pre Recorded Earnings Call, Feb 15, 2023 | 2023-02-15 | 00:00:00 | Q4 2022 |
| 1,827,630,822 | duplicate_call | 1,816,061,612 | <NA> | 009777 | The J. M. Smucker Company | The J. M. Smucker Company, Q3 2023 Pre Recorded Earnings Call, Feb 28, 2023 | 2023-02-28 | 00:00:00 | Q3 2023 |
| 1,828,510,580 | duplicate_call | 1,828,180,345 | <NA> | 005071 | General Mills, Inc. | General Mills, Inc., Q3 2023 Pre Recorded Earnings Call, Mar 23, 2023 | 2023-03-23 | 00:00:00 | Q3 2023 |
| 1,835,135,595 | duplicate_call | 1,832,349,513 | <NA> | 005597 | The Hershey Company | The Hershey Company, Q1 2023 Pre Recorded Earnings Call, Apr 27, 2023 | 2023-04-27 | 00:00:00 | Q1 2023 |
| 1,832,689,694 | duplicate_call | 1,832,689,204 | <NA> | 005568 | The Kraft Heinz Company | The Kraft Heinz Company, Q1 2023 Pre Recorded Earnings Call, May 03, 2023 | 2023-05-03 | 00:00:00 | Q1 2023 |
| 1,837,865,848 | duplicate_call | 1,829,082,216 | <NA> | 009777 | The J. M. Smucker Company | The J. M. Smucker Company, Q4 2023 Pre Recorded Earnings Call, Jun 06, 2023 | 2023-06-06 | 00:00:00 | Q4 2023 |
| 1,840,016,849 | duplicate_call | 1,840,017,065 | <NA> | 005071 | General Mills, Inc. | General Mills, Inc., Q4 2023 Earnings Call, Jun 28, 2023 | 2023-06-28 | 13:00:00 | Q4 2023 |
| 1,849,091,547 | duplicate_call | 1,847,114,816 | <NA> | 005597 | The Hershey Company | The Hershey Company, Q2 2023 Pre Recorded Earnings Call, Jul 27, 2023 | 2023-07-27 | 00:00:00 | Q2 2023 |
| 1,841,131,862 | duplicate_call | 1,846,953,831 | <NA> | 001449 | Aflac Incorporated | Aflac Incorporated, Q2 2023 Earnings Call, Aug 02, 2023 | 2023-08-02 | 12:00:00 | Q2 2023 |
| 1,847,526,923 | duplicate_call | 1,847,528,135 | <NA> | 005568 | The Kraft Heinz Company | The Kraft Heinz Company, Q2 2023 Pre Recorded Earnings Call, Aug 02, 2023 | 2023-08-02 | 00:00:00 | Q2 2023 |
| 1,851,444,381 | duplicate_call | 1,851,438,442 | <NA> | 009777 | The J. M. Smucker Company | The J. M. Smucker Company, Q1 2024 Pre Recorded Earnings Call, Aug 29, 2023 | 2023-08-29 | 00:00:00 | Q1 2024 |
| 1,853,487,426 | duplicate_call | 1,853,487,412 | <NA> | 005071 | General Mills, Inc. | General Mills, Inc., Q1 2024 Pre Recorded Earnings Call, Sep 20, 2023 | 2023-09-20 | 00:00:00 | Q1 2024 |
| 1,859,999,685 | duplicate_call | 1,858,166,734 | <NA> | 005597 | The Hershey Company | The Hershey Company, Q3 2023 Pre Recorded Earnings Call, Oct 26, 2023 | 2023-10-26 | 00:00:00 | Q3 2023 |
| 1,858,513,524 | duplicate_call | 1,858,513,596 | <NA> | 005568 | The Kraft Heinz Company | The Kraft Heinz Company, Q3 2023 Pre Recorded Earnings Call, Nov 01, 2023 | 2023-11-01 | 00:00:00 | Q3 2023 |
| 1,854,771,854 | duplicate_call | 1,858,085,856 | <NA> | 001449 | Aflac Incorporated | Aflac Incorporated, Q3 2023 Earnings Call, Nov 02, 2023 | 2023-11-02 | 12:00:00 | Q3 2023 |
| 1,861,454,138 | duplicate_call | 1,861,454,047 | <NA> | 009777 | The J. M. Smucker Company | The J. M. Smucker Company, Q2 2024 Pre Recorded Earnings Call, Dec 05, 2023 | 2023-12-05 | 00:00:00 | Q2 2024 |
| 1,862,569,135 | duplicate_call | 1,862,568,217 | <NA> | 005071 | General Mills, Inc. | General Mills, Inc., Q2 2024 Pre Recorded Earnings Call, Dec 20, 2023 | 2023-12-20 | 00:00:00 | Q2 2024 |
| 1,863,628,374 | duplicate_call | 1,867,561,501 | <NA> | 001449 | Aflac Incorporated | Aflac Incorporated, Q4 2023 Earnings Call, Feb 01, 2024 | 2024-02-01 | 13:00:00 | Q4 2023 |
| 1,867,586,167 | duplicate_call | 1,871,218,798 | <NA> | 005597 | The Hershey Company | The Hershey Company, Q4 2023 Earnings Call, Feb 08, 2024 | 2024-02-08 | 13:30:00 | Q4 2023 |
| 1,868,664,945 | duplicate_call | 1,868,665,065 | <NA> | 005568 | The Kraft Heinz Company | The Kraft Heinz Company, Q4 2023 Pre Recorded Earnings Call, Feb 14, 2024 | 2024-02-14 | 00:00:00 | Q4 2023 |
| 1,868,920,633 | duplicate_call | 1,868,920,637 | <NA> | 007881 | Newmont Corporation | Newmont Corporation, Q4 2023 Earnings Call, Feb 22, 2024 | 2024-02-22 | 21:00:00 | Q4 2023 |
| 1,871,010,319 | duplicate_call | 1,871,010,463 | <NA> | 009777 | The J. M. Smucker Company | The J. M. Smucker Company, Q3 2024 Pre Recorded Earnings Call, Feb 27, 2024 | 2024-02-27 | 00:00:00 | Q3 2024 |
| 1,872,903,745 | duplicate_call | 1,872,903,946 | <NA> | 005071 | General Mills, Inc. | General Mills, Inc., Q3 2024 Pre Recorded Earnings Call, Mar 20, 2024 | 2024-03-20 | 00:00:00 | Q3 2024 |
| 1,877,205,594 | duplicate_call | 1,873,609,509 | <NA> | 003362 | Conagra Brands, Inc. | Conagra Brands, Inc., Q3 2024 Pre Recorded Earnings Call, Apr 04, 2024 | 2024-04-04 | 00:00:00 | Q3 2024 |
| 1,877,687,769 | duplicate_call | 1,877,684,339 | <NA> | 004503 | ExxonMobil Holdings Corporation | Exxon Mobil Corporation, Q1 2024 Pre Recorded Earnings Call, Apr 28, 2024 | 2024-04-28 | 00:00:00 | Q1 2024 |
| 1,877,508,245 | duplicate_call | 1,877,508,252 | <NA> | 005568 | The Kraft Heinz Company | The Kraft Heinz Company, Q1 2024 Pre Recorded Earnings Call, May 01, 2024 | 2024-05-01 | 00:00:00 | Q1 2024 |
| 1,880,022,904 | duplicate_call | 1,877,628,698 | <NA> | 005597 | The Hershey Company | The Hershey Company, Q1 2024 Pre Recorded Earnings Call, May 03, 2024 | 2024-05-03 | 00:00:00 | Q1 2024 |
| 1,881,245,025 | duplicate_call | 1,881,245,967 | <NA> | 009777 | The J. M. Smucker Company | The J. M. Smucker Company, Q4 2024 Pre Recorded Earnings Call, Jun 06, 2024 | 2024-06-06 | 00:00:00 | Q4 2024 |
| 1,882,254,713 | duplicate_call | 1,882,254,638 | <NA> | 005071 | General Mills, Inc. | General Mills, Inc., Q4 2024 Pre Recorded Earnings Call, Jun 26, 2024 | 2024-06-26 | 00:00:00 | Q4 2024 |
| 1,884,463,515 | duplicate_call | 1,884,464,025 | <NA> | 003362 | Conagra Brands, Inc. | Conagra Brands, Inc., Q4 2024 Pre Recorded Earnings Call, Jul 11, 2024 | 2024-07-11 | 00:00:00 | Q4 2024 |
| 1,887,394,127 | duplicate_call | 1,887,394,178 | <NA> | 005568 | The Kraft Heinz Company | The Kraft Heinz Company, Q2 2024 Pre Recorded Earnings Call, Jul 31, 2024 | 2024-07-31 | 00:00:00 | Q2 2024 |
| 1,884,135,954 | duplicate_call | 1,885,824,306 | <NA> | 001449 | Aflac Incorporated | Aflac Incorporated, Q2 2024 Earnings Call, Aug 01, 2024 | 2024-08-01 | 12:00:00 | Q2 2024 |
| 1,889,486,164 | duplicate_call | 1,886,209,541 | <NA> | 005597 | The Hershey Company | The Hershey Company, Q2 2024 Pre Recorded Earnings Call, Aug 01, 2024 | 2024-08-01 | 00:00:00 | Q2 2024 |
| 1,889,428,605 | duplicate_call | 1,887,320,057 | <NA> | 133768 | MetLife, Inc. | MetLife, Inc., Q2 2024 Pre Recorded Earnings Call, Aug 01, 2024 | 2024-08-01 | 00:00:00 | Q2 2024 |
| 1,892,415,646 | duplicate_call | 1,892,415,631 | <NA> | 005071 | General Mills, Inc. | General Mills, Inc., Q1 2025 Pre Recorded Earnings Call, Sep 18, 2024 | 2024-09-18 | 00:00:00 | Q1 2025 |
| 1,893,364,834 | duplicate_call | 1,893,363,962 | <NA> | 003362 | Conagra Brands, Inc. | Conagra Brands, Inc., Q1 2025 Pre Recorded Earnings Call, Oct 02, 2024 | 2024-10-02 | 00:00:00 | Q1 2025 |
| 1,901,425,005 | duplicate_call | 1,857,971,841 | <NA> | 002136 | Verizon Communications Inc. | Verizon Communications Inc., Q3 2024 Pre Recorded Earnings Call, Oct 22, 2024 | 2024-10-22 | 00:00:00 | Q3 2024 |
| 1,902,114,963 | duplicate_call | 1,899,846,307 | <NA> | 005568 | The Kraft Heinz Company | The Kraft Heinz Company, Q3 2024 Pre Recorded Earnings Call, Oct 30, 2024 | 2024-10-30 | 00:00:00 | Q3 2024 |
| 1,903,109,779 | duplicate_call | 1,903,109,120 | <NA> | 009777 | The J. M. Smucker Company | The J. M. Smucker Company, Q2 2025 Pre Recorded Earnings Call, Nov 26, 2024 | 2024-11-26 | 00:00:00 | Q2 2025 |
| 1,911,358,223 | duplicate_call | 1,911,357,788 | <NA> | 005071 | General Mills, Inc. | General Mills, Inc., Q2 2025 Pre Recorded Earnings Call, Dec 18, 2024 | 2024-12-18 | 00:00:00 | Q2 2025 |
| 1,916,087,618 | duplicate_call | 1,930,282,821 | <NA> | 005597 | The Hershey Company | The Hershey Company, Q4 2024 Earnings Call, Feb 06, 2025 | 2025-02-06 | 13:30:00 | Q4 2024 |
| 1,917,573,387 | duplicate_call | 1,917,573,421 | <NA> | 005568 | The Kraft Heinz Company | The Kraft Heinz Company, Q4 2024 Pre Recorded Earnings Call, Feb 12, 2025 | 2025-02-12 | 00:00:00 | Q4 2024 |
| 1,932,605,736 | duplicate_call | 1,932,605,838 | <NA> | 005071 | General Mills, Inc. | General Mills, Inc., Q3 2025 Pre Recorded Earnings Call, Mar 19, 2025 | 2025-03-19 | 00:00:00 | Q3 2025 |
| 1,934,325,177 | duplicate_call | 1,931,952,633 | <NA> | 007343 | Micron Technology, Inc. | Micron Technology, Inc., Q2 2025 Earnings Call, Mar 20, 2025 | 2025-03-20 | 22:00:00 | Q2 2025 |
| 1,938,312,227 | duplicate_call | 1,899,018,636 | <NA> | 002136 | Verizon Communications Inc. | Verizon Communications Inc., Q1 2025 Pre Recorded Earnings Call, Apr 22, 2025 | 2025-04-22 | 00:00:00 | Q1 2025 |
| 1,938,314,750 | duplicate_call | 1,938,314,940 | <NA> | 005568 | The Kraft Heinz Company | The Kraft Heinz Company, Q1 2025 Pre Recorded Earnings Call, Apr 29, 2025 | 2025-04-29 | 00:00:00 | Q1 2025 |
| 1,941,575,314 | duplicate_call | 1,937,716,076 | <NA> | 005597 | The Hershey Company | The Hershey Company, Q1 2025 Pre Recorded Earnings Call, May 01, 2025 | 2025-05-01 | 00:00:00 | Q1 2025 |
| 1,941,543,896 | duplicate_call | 1,938,148,556 | <NA> | 004503 | ExxonMobil Holdings Corporation | Exxon Mobil Corporation, Q1 2025 Pre Recorded Earnings Call, May 02, 2025 | 2025-05-02 | 00:00:00 | Q1 2025 |
| 1,943,326,438 | duplicate_call | 1,943,326,279 | <NA> | 009777 | The J. M. Smucker Company | The J. M. Smucker Company, Q4 2025 Pre Recorded Earnings Call, Jun 10, 2025 | 2025-06-10 | 00:00:00 | Q4 2025 |
| 1,946,103,325 | duplicate_call | 1,946,104,046 | <NA> | 005071 | General Mills, Inc. | General Mills, Inc., Q4 2025 Pre Recorded Earnings Call, Jun 25, 2025 | 2025-06-25 | 00:00:00 | Q4 2025 |
| 1,944,666,647 | duplicate_call | 1,944,415,840 | <NA> | 007343 | Micron Technology, Inc. | Micron Technology, Inc., Q3 2025 Earnings Call, Jun 25, 2025 | 2025-06-25 | 22:00:00 | Q3 2025 |
| 1,947,526,497 | duplicate_call | 1,947,525,957 | <NA> | 003362 | Conagra Brands, Inc. | Conagra Brands, Inc., Q4 2025 Pre Recorded Earnings Call, Jul 10, 2025 | 2025-07-10 | 00:00:00 | Q4 2025 |
| 1,951,586,257 | duplicate_call | 1,955,097,054 | <NA> | 005597 | The Hershey Company | The Hershey Company, Q2 2025 Earnings Call, Jul 30, 2025 | 2025-07-30 | 12:15:00 | Q2 2025 |
| 1,955,883,220 | duplicate_call | 1,955,883,853 | <NA> | 009777 | The J. M. Smucker Company | The J. M. Smucker Company, Q1 2026 Pre Recorded Earnings Call, Aug 27, 2025 | 2025-08-27 | 00:00:00 | Q1 2026 |
| 1,959,051,081 | duplicate_call | 1,958,584,927 | <NA> | 007343 | Micron Technology, Inc. | Micron Technology, Inc., Q4 2025 Earnings Call, Sep 23, 2025 | 2025-09-23 | 22:00:00 | Q4 2025 |
| 1,959,051,630 | duplicate_call | 1,959,051,721 | <NA> | 003362 | Conagra Brands, Inc. | Conagra Brands, Inc., Q1 2026 Pre Recorded Earnings Call, Oct 01, 2025 | 2025-10-01 | 00:00:00 | Q1 2026 |
| 1,971,299,962 | duplicate_call | 1,971,036,965 | <NA> | 007343 | Micron Technology, Inc. | Micron Technology, Inc., Q1 2026 Earnings Call, Dec 17, 2025 | 2025-12-17 | 23:00:00 | Q1 2026 |
| 1,970,999,015 | duplicate_call | 1,970,998,759 | <NA> | 003362 | Conagra Brands, Inc. | Conagra Brands, Inc., Q2 2026 Pre Recorded Earnings Call, Dec 19, 2025 | 2025-12-19 | 00:00:00 | Q2 2026 |
| 1,982,721,500 | duplicate_call | 1,977,292,890 | <NA> | 005597 | The Hershey Company | The Hershey Company, Q4 2025 Pre Recorded Earnings Call, Feb 05, 2026 | 2026-02-05 | 00:00:00 | Q4 2025 |
| 1,989,973,680 | duplicate_call | 1,986,336,424 | <NA> | 007343 | Micron Technology, Inc. | Micron Technology, Inc., Q2 2026 Earnings Call, Mar 18, 2026 | 2026-03-18 | 22:00:00 | Q2 2026 |
| 2,004,549,982 | duplicate_call | 2,002,504,482 | <NA> | 007343 | Micron Technology, Inc. | Micron Technology, Inc., Q3 2026 Earnings Call, Jun 24, 2026 | 2026-06-24 | 22:00:00 | Q3 2026 |
| 2,010,260,979 | duplicate_call | 2,010,265,117 | <NA> | 005568 | The Kraft Heinz Company | The Kraft Heinz Company, Q2 2026 Pre Recorded Earnings Call, Aug 05, 2026 | 2026-08-05 | 00:00:00 | Q2 2026 |

</details>
