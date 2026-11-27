# Full-history source: call table and run log

Built by `scripts/01_build_features.py --source full --embed st` (`scripts/full_source.py`) from `data/interim` (`la transcripts`) and `data/raw/sp500_pit_2008`. Firm key: gvkey.

Run time: 193.7 min. Written 2026-11-27T05:15:05+00:00.

## Call attrition (calls from 2008, so the history features are complete in 2011)

| Step | Calls before | Calls after | Dropped |
|---|---|---|---|
| Earnings calls (event type) | 44,107 | 44,107 | 0 |
| Call date 2008-01-01 to 2026-08-31 | 44,107 | 44,075 | 32 |
| CIQ companyid -> gvkey (date-valid) | 44,075 | 44,024 | 51 |
| gvkey -> permno (CCM LU/LC, date-valid) | 44,024 | 43,233 | 791 |
| S&P 500 member on call date | 43,233 | 32,085 | 11,148 |
| One permno per call (share_class_rule) | 32,085 | 32,085 | 0 |
| One call per firm and fiscal period (duplicate_calls: keep_first) | 32,085 | 31,974 | 111 |
| Has call sections | 31,974 | 31,974 | 0 |
| Features written for 2011-01-01 to 2026-08-31 (earlier calls feed history only) | 31,974 | 29,968 | 2,006 |

- Calls with more than one member permno: 190; calls with no CCM linkprim P link: 884, of which 0 needed the market-cap rule.
- Calls with no GICS sector valid on the call date (kept, sector blank): 32.
- Calls with no fiscal label in the headline (never treated as duplicates): 1.
- Calls at 00:00:00 UTC, CIQ's placeholder time (ordered last within their date for keep_first; session = post): 95.
- Same gvkey and fiscal label but more than 3 days apart (fiscal-year changes; kept as separate releases): 19.
- Q&A pairs of the kept calls (2008 on): 719,441 parsed, 708,011 with a non-empty answer; 654,447 pairs in the written calls.

## Duplicate calls dropped (same gvkey and fiscal label within 3 days, later call time)

111 calls dropped.

| year | dropped |
|---|---|
| 2010 | 2 |
| 2011 | 9 |
| 2012 | 8 |
| 2013 | 7 |
| 2014 | 7 |
| 2015 | 7 |
| 2016 | 5 |
| 2017 | 2 |
| 2018 | 2 |
| 2020 | 8 |
| 2021 | 5 |
| 2022 | 11 |
| 2023 | 8 |
| 2024 | 14 |
| 2025 | 13 |
| 2026 | 3 |

<details><summary>All dropped calls</summary>

| keydevid | transcriptid | gvkey | companyname | headline | call_date | call_time_utc | fiscal_label | kept_transcriptid |
|---|---|---|---|---|---|---|---|---|
| 114668627 | 81711 | 004839 | Ford Motor Company | Ford Motor Co., Q3 2010 Earnings Call, Oct 26, 2010 | 2010-10-26 | 15:00:00 | Q3 2010 | 81725 |
| 114467023 | 83185 | 004885 | Franklin Templeton Inc. | Franklin Resources Inc., Q4 2010 Earnings Call, Oct 28, 2010 | 2010-10-28 | 20:30:00 | Q4 2010 | 82721 |
| 118543116 | 96908 | 004839 | Ford Motor Company | Ford Motor Co., Q4 2010 Earnings Call, Jan 28, 2011 | 2011-01-28 | 16:00:00 | Q4 2010 | 96738 |
| 118546825 | 101551 | 003505 | Molson Coors Beverage Company | Molson Coors Brewing Company, Q4 2010 Earnings Call, Feb 10, 2011 | 2011-02-10 | 19:00:00 | Q4 2010 | 101416 |
| 130287464 | 117592 | 004839 | Ford Motor Company | Ford Motor Co., Q1 2011 Earnings Call, Apr 26, 2011 | 2011-04-26 | 15:00:00 | Q1 2011 | 117617 |
| 130292330 | 121204 | 003505 | Molson Coors Beverage Company | Molson Coors Brewing Company, Q1 2011 Earnings Call, May 03, 2011 | 2011-05-03 | 18:00:00 | Q1 2011 | 121098 |
| 137660050 | 146186 | 004839 | Ford Motor Company | Ford Motor Co., Q2 2011 Earnings Call, Jul 26, 2011 | 2011-07-26 | 15:00:00 | Q2 2011 | 145870 |
| 137335433 | 151038 | 004885 | Franklin Templeton Inc. | Franklin Resources Inc., Q3 2011 Earnings Call, Aug 02, 2011 | 2011-08-02 | 20:30:00 | Q3 2011 | 150392 |
| 141871103 | 178303 | 004839 | Ford Motor Company | Ford Motor Co., Q3 2011 Earnings Call, Oct 26, 2011 | 2011-10-26 | 15:00:00 | Q3 2011 | 178283 |
| 141522446 | 179196 | 004885 | Franklin Templeton Inc. | Franklin Resources Inc., Q4 2011 Earnings Call, Oct 27, 2011 | 2011-10-27 | 00:00:00 | Q4 2011 | 180058 |
| 141543967 | 183047 | 003505 | Molson Coors Beverage Company | Molson Coors Brewing Company, Q3 2011 Earnings Call, Nov 02, 2011 | 2011-11-02 | 18:00:00 | Q3 2011 | 183096 |
| 146357156 | 212300 | 004839 | Ford Motor Company | Ford Motor Co., Q4 2011 Earnings Call, Jan 27, 2012 | 2012-01-27 | 16:00:00 | Q4 2011 | 212282 |
| 145980875 | 213605 | 004885 | Franklin Templeton Inc. | Franklin Resources Inc., Q1 2012 Earnings Call, Feb 01, 2012 | 2012-02-01 | 00:00:00 | Q1 2012 | 214258 |
| 146297746 | 259008 | 003505 | Molson Coors Beverage Company | Molson Coors Brewing Company, Q4 2011 Earnings Call, Feb 16, 2012 | 2012-02-16 | 19:00:00 | Q4 2011 | 259103 |
| 184476420 | 295183 | 004885 | Franklin Templeton Inc. | Franklin Resources Inc., Q2 2012 Pre Recorded Earnings Call, May 02, 2012 | 2012-05-02 | 00:00:00 | Q2 2012 | 295801 |
| 193455454 | 300559 | 003505 | Molson Coors Beverage Company | Molson Coors Brewing Company, Q1 2012 Earnings Call, May 08, 2012 | 2012-05-08 | 18:00:00 | Q1 2012 | 300650 |
| 214174243 | 334413 | 004885 | Franklin Templeton Inc. | Franklin Resources Inc., Q3 2012 Pre Recorded Earnings Call, Jul 30, 2012 | 2012-07-30 | 00:00:00 | Q3 2012 | 334788 |
| 222841106 | 369838 | 004885 | Franklin Templeton Inc. | Franklin Resources Inc., Q4 2012 Pre Recorded Earnings Call, Oct 25, 2012 | 2012-10-25 | 00:00:00 | Q4 2012 | 370819 |
| 223357972 | 380613 | 003505 | Molson Coors Beverage Company | Molson Coors Brewing Company, Q3 2012 Earnings Call, Nov 07, 2012 | 2012-11-07 | 19:00:00 | Q3 2012 | 380647 |
| 225974384 | 408762 | 004885 | Franklin Templeton Inc. | Franklin Resources Inc., Q1 2013 Pre Recorded Earnings Call, Feb 01, 2013 | 2013-02-01 | 00:00:00 | Q1 2013 | 408843 |
| 235948626 | 451171 | 004885 | Franklin Templeton Inc. | Franklin Resources Inc., Q2 2013 Pre Recorded Earnings Call, Apr 30, 2013 | 2013-04-30 | 00:00:00 | Q2 2013 | 450959 |
| 236745109 | 457602 | 003505 | Molson Coors Beverage Company | Molson Coors Brewing Company, Q1 2013 Earnings Call, May 07, 2013 | 2013-05-07 | 18:00:00 | Q1 2013 | 457615 |
| 243693032 | 490765 | 004885 | Franklin Templeton Inc. | Franklin Resources Inc., Q3 2013 Pre Recorded Earnings Call, Jul 29, 2013 | 2013-07-29 | 00:00:00 | Q3 2013 | 490927 |
| 243199591 | 498298 | 003505 | Molson Coors Beverage Company | Molson Coors Brewing Company, Q2 2013 Earnings Call, Aug 06, 2013 | 2013-08-06 | 18:00:00 | Q2 2013 | 498313 |
| 247855378 | 532937 | 004885 | Franklin Templeton Inc. | Franklin Resources Inc., Q4 2013 Pre Recorded Earnings Call, Oct 24, 2013 | 2013-10-24 | 00:00:00 | Q4 2013 | 532963 |
| 247987514 | 543354 | 003505 | Molson Coors Beverage Company | Molson Coors Brewing Company, Q3 2013 Earnings Call, Nov 06, 2013 | 2013-11-06 | 19:00:00 | Q3 2013 | 543336 |
| 253997936 | 574098 | 004885 | Franklin Templeton Inc. | Franklin Resources Inc., Q1 2014 Pre Recorded Earnings Call, Jan 30, 2014 | 2014-01-30 | 00:00:00 | Q1 2014 | 574218 |
| 254461727 | 583762 | 003505 | Molson Coors Beverage Company | Molson Coors Brewing Company, Q4 2013 Earnings Call, Feb 13, 2014 | 2014-02-13 | 19:00:00 | Q4 2013 | 583450 |
| 261915714 | 617360 | 004885 | Franklin Templeton Inc. | Franklin Resources Inc., Q2 2014 Pre Recorded Earnings Call, Apr 28, 2014 | 2014-04-28 | 00:00:00 | Q2 2014 | 617619 |
| 261905182 | 627443 | 003505 | Molson Coors Beverage Company | Molson Coors Brewing Company, Q1 2014 Earnings Call, May 07, 2014 | 2014-05-07 | 18:00:00 | Q1 2014 | 627448 |
| 269353081 | 667862 | 004885 | Franklin Templeton Inc. | Franklin Resources Inc., Q3 2014 Pre Recorded Earnings Call, Jul 30, 2014 | 2014-07-30 | 00:00:00 | Q3 2014 | 667765 |
| 268171074 | 675850 | 003505 | Molson Coors Beverage Company | Molson Coors Brewing Company, Q2 2014 Earnings Call, Aug 06, 2014 | 2014-08-06 | 18:00:00 | Q2 2014 | 675631 |
| 248059782 | 722505 | 003505 | Molson Coors Beverage Company | Molson Coors Brewing Company, Q3 2014 Earnings Call, Nov 06, 2014 | 2014-11-06 | 18:00:00 | Q3 2014 | 722260 |
| 281512568 | 752463 | 004885 | Franklin Templeton Inc. | Franklin Resources, Inc., Q1 2015 Pre Recorded Earnings Call, Jan 30, 2015 | 2015-01-30 | 00:00:00 | Q1 2015 | 752642 |
| 281884033 | 758575 | 003505 | Molson Coors Beverage Company | Molson Coors Brewing Company, Q4 2014 Earnings Call, Feb 10, 2015 | 2015-02-10 | 18:00:00 | Q4 2014 | 758704 |
| 291347622 | 797575 | 004885 | Franklin Templeton Inc. | Franklin Resources, Inc., Q2 2015 Pre Recorded Earnings Call, Apr 29, 2015 | 2015-04-29 | 00:00:00 | Q2 2015 | 797533 |
| 290797780 | 807081 | 003505 | Molson Coors Beverage Company | Molson Coors Brewing Company, Q1 2015 Earnings Call, May 07, 2015 | 2015-05-07 | 17:00:00 | Q1 2015 | 807173 |
| 305856031 | 841479 | 004885 | Franklin Templeton Inc. | Franklin Resources, Inc., Q3 2015 Pre Recorded Earnings Call, Jul 29, 2015 | 2015-07-29 | 00:00:00 | Q3 2015 | 841230 |
| 307200857 | 851240 | 003505 | Molson Coors Beverage Company | Molson Coors Brewing Company, Q2 2015 Earnings Call, Aug 06, 2015 | 2015-08-06 | 17:00:00 | Q2 2015 | 850896 |
| 313260434 | 893943 | 003505 | Molson Coors Beverage Company | Molson Coors Brewing Company, Q3 2015 Earnings Call, Nov 05, 2015 | 2015-11-05 | 18:00:00 | Q3 2015 | 893543 |
| 322094111 | 925642 | 004885 | Franklin Templeton Inc. | Franklin Resources, Inc., Q1 2016 Pre Recorded Earnings Call, Feb 03, 2016 | 2016-02-03 | 00:00:00 | Q1 2016 | 925707 |
| 330709118 | 964420 | 004885 | Franklin Templeton Inc. | Franklin Resources, Inc., Q2 2016 Pre Recorded Earnings Call, Apr 27, 2016 | 2016-04-27 | 00:00:00 | Q2 2016 | 964579 |
| 330118935 | 971038 | 003505 | Molson Coors Beverage Company | Molson Coors Brewing Company, Q1 2016 Earnings Call, May 03, 2016 | 2016-05-03 | 17:00:00 | Q1 2016 | 970715 |
| 378657075 | 1021662 | 003505 | Molson Coors Beverage Company | Molson Coors Brewing Company, Q2 2016 Earnings Call, Aug 02, 2016 | 2016-08-02 | 17:00:00 | Q2 2016 | 1021430 |
| 404558490 | 1061754 | 004885 | Franklin Templeton Inc. | Franklin Resources, Inc., Q4 2016 Pre Recorded Earnings Call, Oct 26, 2016 | 2016-10-26 | 00:00:00 | Q4 2016 | 1061982 |
| 417110086 | 1131747 | 003505 | Molson Coors Beverage Company | Molson Coors Brewing Company, Q4 2016 Earnings Call, Feb 14, 2017 | 2017-02-14 | 18:00:00 | Q4 2016 | 1131539 |
| 429438581 | 1187999 | 003505 | Molson Coors Beverage Company | Molson Coors Brewing Company, Q1 2017 Earnings Call, May 03, 2017 | 2017-05-03 | 17:00:00 | Q1 2017 | 1187585 |
| 558239654 | 1432475 | 008463 | Pentair plc | Pentair plc, Q1 2018 Earnings Call, Apr 19, 2018 | 2018-04-19 | 12:45:00 | Q1 2018 | 1432482 |
| 570749046 | 1495479 | 008402 | Paychex, Inc. | Paychex, Inc., Q4 2018 Earnings Call, Jun 27, 2018 | 2018-06-27 | 15:00:00 | Q4 2018 | 1495397 |
| 665836898 | 1962101 | 008479 | PepsiCo, Inc. | PepsiCo, Inc., Q1 2020 Pre Recorded Earnings Call, Apr 28, 2020 | 2020-04-28 | 00:00:00 | Q1 2020 | 1961725 |
| 668843958 | 2006691 | 002663 | The Campbell's Company | Campbell Soup Company, Q3 2020 Pre Recorded Earnings Call, Jun 03, 2020 | 2020-06-03 | 00:00:00 | Q3 2020 | 2006246 |
| 674201331 | 2030407 | 005071 | General Mills, Inc. | General Mills, Inc., Q4 2020 Pre Recorded Earnings Call, Jul 01, 2020 | 2020-07-01 | 00:00:00 | Q4 2020 | 2030355 |
| 669317587 | 2035750 | 008479 | PepsiCo, Inc. | PepsiCo, Inc., Q2 2020 Pre Recorded Earnings Call, Jul 13, 2020 | 2020-07-13 | 00:00:00 | Q2 2020 | 2035850 |
| 673591179 | 2044518 | 005597 | The Hershey Company | The Hershey Company, Q2 2020 Pre Recorded Earnings Call, Jul 23, 2020 | 2020-07-23 | 00:00:00 | Q2 2020 | 2044266 |
| 684519754 | 2089466 | 002663 | The Campbell's Company | Campbell Soup Company, Q4 2020 Pre Recorded Earnings Call, Sep 03, 2020 | 2020-09-03 | 00:00:00 | Q4 2020 | 2089313 |
| 689571624 | 2102813 | 005071 | General Mills, Inc. | General Mills, Inc., Q1 2021 Pre Recorded Earnings Call, Sep 23, 2020 | 2020-09-23 | 00:00:00 | Q1 2021 | 2102824 |
| 685485009 | 2105902 | 008479 | PepsiCo, Inc. | PepsiCo, Inc., Q3 2020 Pre Recorded Earnings Call, Oct 01, 2020 | 2020-10-01 | 00:00:00 | Q3 2020 | 2106006 |
| 703270895 | 2192260 | 005597 | The Hershey Company | The Hershey Company, Q4 2020 Pre Recorded Earnings Call, Feb 04, 2021 | 2021-02-04 | 00:00:00 | Q4 2020 | 2192088 |
| 713098656 | 2263005 | 005597 | The Hershey Company | The Hershey Company, Q1 2021 Pre Recorded Earnings Call, Apr 29, 2021 | 2021-04-29 | 00:00:00 | Q1 2021 | 2263067 |
| 1676352902 | 2386830 | 009777 | The J. M. Smucker Company | The J. M. Smucker Company, Q1 2022 Pre Recorded Earnings Call, Aug 26, 2021 | 2021-08-26 | 00:00:00 | Q1 2022 | 2386707 |
| 1684347080 | 2414450 | 005568 | The Kraft Heinz Company | The Kraft Heinz Company, Q3 2021 Pre Recorded Earnings Call, Oct 27, 2021 | 2021-10-27 | 00:00:00 | Q3 2021 | 2414127 |
| 1760204955 | 2463588 | 005071 | General Mills, Inc. | General Mills, Inc., Q2 2022 Pre Recorded Earnings Call, Dec 21, 2021 | 2021-12-21 | 00:00:00 | Q2 2022 | 2463491 |
| 1769743460 | 2484372 | 005597 | The Hershey Company | The Hershey Company, Q4 2021 Pre Recorded Earnings Call, Feb 03, 2022 | 2022-02-03 | 00:00:00 | Q4 2021 | 2484014 |
| 1773231639 | 2529766 | 005071 | General Mills, Inc. | General Mills, Inc., Q3 2022 Pre Recorded Earnings Call, Mar 23, 2022 | 2022-03-23 | 00:00:00 | Q3 2022 | 2529542 |
| 1783615448 | 2585609 | 009777 | The J. M. Smucker Company | The J. M. Smucker Company, Q4 2022 Pre Recorded Earnings Call, Jun 07, 2022 | 2022-06-07 | 00:00:00 | Q4 2022 | 2585216 |
| 1785257882 | 2593944 | 005071 | General Mills, Inc. | General Mills, Inc., Q4 2022 Pre Recorded Earnings Call, Jun 29, 2022 | 2022-06-29 | 00:00:00 | Q4 2022 | 2594158 |
| 1785263519 | 2594741 | 007343 | Micron Technology, Inc. | Micron Technology, Inc., Q3 2022 Earnings Call, Jun 30, 2022 | 2022-06-30 | 22:00:00 | Q3 2022 | 2594722 |
| 1789593261 | 2605876 | 005568 | The Kraft Heinz Company | The Kraft Heinz Company, Q2 2022 Pre Recorded Earnings Call, Jul 27, 2022 | 2022-07-27 | 00:00:00 | Q2 2022 | 2605225 |
| 1794347464 | 2636288 | 009777 | The J. M. Smucker Company | The J. M. Smucker Company, Q1 2023 Earnings Call, Aug 23, 2022 | 2022-08-23 | 13:00:00 | Q1 2023 | 2636362 |
| 1814136926 | 2675588 | 005597 | The Hershey Company | The Hershey Company, Q3 2022 Pre Recorded Earnings Call, Nov 04, 2022 | 2022-11-04 | 00:00:00 | Q3 2022 | 2675804 |
| 1813344622 | 2695273 | 009777 | The J. M. Smucker Company | The J. M. Smucker Company, Q2 2023 Pre Recorded Earnings Call, Nov 21, 2022 | 2022-11-21 | 00:00:00 | Q2 2023 | 2695262 |
| 1816852237 | 2711975 | 005071 | General Mills, Inc. | General Mills, Inc., Q2 2023 Pre Recorded Earnings Call, Dec 20, 2022 | 2022-12-20 | 00:00:00 | Q2 2023 | 2711908 |
| 1819409582 | 2712254 | 007343 | Micron Technology, Inc. | Micron Technology, Inc., Q1 2023 Earnings Call, Dec 21, 2022 | 2022-12-21 | 23:00:00 | Q1 2023 | 2712179 |
| 1835135595 | 2788177 | 005597 | The Hershey Company | The Hershey Company, Q1 2023 Pre Recorded Earnings Call, Apr 27, 2023 | 2023-04-27 | 00:00:00 | Q1 2023 | 2787851 |
| 1837865848 | 2831634 | 009777 | The J. M. Smucker Company | The J. M. Smucker Company, Q4 2023 Pre Recorded Earnings Call, Jun 06, 2023 | 2023-06-06 | 00:00:00 | Q4 2023 | 2831621 |
| 1840016849 | 2844367 | 005071 | General Mills, Inc. | General Mills, Inc., Q4 2023 Earnings Call, Jun 28, 2023 | 2023-06-28 | 13:00:00 | Q4 2023 | 2844356 |
| 1847526923 | 2865866 | 005568 | The Kraft Heinz Company | The Kraft Heinz Company, Q2 2023 Pre Recorded Earnings Call, Aug 02, 2023 | 2023-08-02 | 00:00:00 | Q2 2023 | 2865648 |
| 1853487426 | 2909750 | 005071 | General Mills, Inc. | General Mills, Inc., Q1 2024 Pre Recorded Earnings Call, Sep 20, 2023 | 2023-09-20 | 00:00:00 | Q1 2024 | 2909676 |
| 1859999685 | 2927227 | 005597 | The Hershey Company | The Hershey Company, Q3 2023 Pre Recorded Earnings Call, Oct 26, 2023 | 2023-10-26 | 00:00:00 | Q3 2023 | 2927018 |
| 1858513524 | 2932366 | 005568 | The Kraft Heinz Company | The Kraft Heinz Company, Q3 2023 Pre Recorded Earnings Call, Nov 01, 2023 | 2023-11-01 | 00:00:00 | Q3 2023 | 2932066 |
| 1862569135 | 2983969 | 005071 | General Mills, Inc. | General Mills, Inc., Q2 2024 Pre Recorded Earnings Call, Dec 20, 2023 | 2023-12-20 | 00:00:00 | Q2 2024 | 2983890 |
| 1867586167 | 3016030 | 005597 | The Hershey Company | The Hershey Company, Q4 2023 Earnings Call, Feb 08, 2024 | 2024-02-08 | 13:30:00 | Q4 2023 | 3016537 |
| 1868664945 | 3023746 | 005568 | The Kraft Heinz Company | The Kraft Heinz Company, Q4 2023 Pre Recorded Earnings Call, Feb 14, 2024 | 2024-02-14 | 00:00:00 | Q4 2023 | 3023342 |
| 1868920633 | 3034534 | 007881 | Newmont Corporation | Newmont Corporation, Q4 2023 Earnings Call, Feb 22, 2024 | 2024-02-22 | 21:00:00 | Q4 2023 | 3034195 |
| 1871010319 | 3040633 | 009777 | The J. M. Smucker Company | The J. M. Smucker Company, Q3 2024 Pre Recorded Earnings Call, Feb 27, 2024 | 2024-02-27 | 00:00:00 | Q3 2024 | 3040935 |
| 1877205594 | 3079352 | 003362 | Conagra Brands, Inc. | Conagra Brands, Inc., Q3 2024 Pre Recorded Earnings Call, Apr 04, 2024 | 2024-04-04 | 00:00:00 | Q3 2024 | 3079384 |
| 1877687769 | 3100936 | 004503 | ExxonMobil Holdings Corporation | Exxon Mobil Corporation, Q1 2024 Pre Recorded Earnings Call, Apr 28, 2024 | 2024-04-28 | 00:00:00 | Q1 2024 | 3098691 |
| 1877508245 | 3105110 | 005568 | The Kraft Heinz Company | The Kraft Heinz Company, Q1 2024 Pre Recorded Earnings Call, May 01, 2024 | 2024-05-01 | 00:00:00 | Q1 2024 | 3104749 |
| 1880022904 | 3109494 | 005597 | The Hershey Company | The Hershey Company, Q1 2024 Pre Recorded Earnings Call, May 03, 2024 | 2024-05-03 | 00:00:00 | Q1 2024 | 3109070 |
| 1881245025 | 3154540 | 009777 | The J. M. Smucker Company | The J. M. Smucker Company, Q4 2024 Pre Recorded Earnings Call, Jun 06, 2024 | 2024-06-06 | 00:00:00 | Q4 2024 | 3154216 |
| 1882254713 | 3170595 | 005071 | General Mills, Inc. | General Mills, Inc., Q4 2024 Pre Recorded Earnings Call, Jun 26, 2024 | 2024-06-26 | 00:00:00 | Q4 2024 | 3170395 |
| 1884463515 | 3184973 | 003362 | Conagra Brands, Inc. | Conagra Brands, Inc., Q4 2024 Pre Recorded Earnings Call, Jul 11, 2024 | 2024-07-11 | 00:00:00 | Q4 2024 | 3185002 |
| 1889428605 | 3207507 | 133768 | MetLife, Inc. | MetLife, Inc., Q2 2024 Pre Recorded Earnings Call, Aug 01, 2024 | 2024-08-01 | 00:00:00 | Q2 2024 | 3208364 |
| 1893364834 | 3276112 | 003362 | Conagra Brands, Inc. | Conagra Brands, Inc., Q1 2025 Pre Recorded Earnings Call, Oct 02, 2024 | 2024-10-02 | 00:00:00 | Q1 2025 | 3276084 |
| 1901425005 | 3286140 | 002136 | Verizon Communications Inc. | Verizon Communications Inc., Q3 2024 Pre Recorded Earnings Call, Oct 22, 2024 | 2024-10-22 | 00:00:00 | Q3 2024 | 3286293 |
| 1916087618 | 3367470 | 005597 | The Hershey Company | The Hershey Company, Q4 2024 Earnings Call, Feb 06, 2025 | 2025-02-06 | 13:30:00 | Q4 2024 | 3367606 |
| 1934325177 | 3405547 | 007343 | Micron Technology, Inc. | Micron Technology, Inc., Q2 2025 Earnings Call, Mar 20, 2025 | 2025-03-20 | 22:00:00 | Q2 2025 | 3405520 |
| 1938312227 | 3421791 | 002136 | Verizon Communications Inc. | Verizon Communications Inc., Q1 2025 Pre Recorded Earnings Call, Apr 22, 2025 | 2025-04-22 | 00:00:00 | Q1 2025 | 3421768 |
| 1941575314 | 3432165 | 005597 | The Hershey Company | The Hershey Company, Q1 2025 Pre Recorded Earnings Call, May 01, 2025 | 2025-05-01 | 00:00:00 | Q1 2025 | 3432389 |
| 1941543896 | 3436837 | 004503 | ExxonMobil Holdings Corporation | Exxon Mobil Corporation, Q1 2025 Pre Recorded Earnings Call, May 02, 2025 | 2025-05-02 | 00:00:00 | Q1 2025 | 3437402 |
| 1946103325 | 3491552 | 005071 | General Mills, Inc. | General Mills, Inc., Q4 2025 Pre Recorded Earnings Call, Jun 25, 2025 | 2025-06-25 | 00:00:00 | Q4 2025 | 3491384 |
| 1944666647 | 3491707 | 007343 | Micron Technology, Inc. | Micron Technology, Inc., Q3 2025 Earnings Call, Jun 25, 2025 | 2025-06-25 | 22:00:00 | Q3 2025 | 3491689 |
| 1947526497 | 3496130 | 003362 | Conagra Brands, Inc. | Conagra Brands, Inc., Q4 2025 Pre Recorded Earnings Call, Jul 10, 2025 | 2025-07-10 | 00:00:00 | Q4 2025 | 3496076 |
| 1951586257 | 3507830 | 005597 | The Hershey Company | The Hershey Company, Q2 2025 Earnings Call, Jul 30, 2025 | 2025-07-30 | 12:15:00 | Q2 2025 | 3508534 |
| 1959051081 | 3552227 | 007343 | Micron Technology, Inc. | Micron Technology, Inc., Q4 2025 Earnings Call, Sep 23, 2025 | 2025-09-23 | 22:00:00 | Q4 2025 | 3552240 |
| 1959051630 | 3555577 | 003362 | Conagra Brands, Inc. | Conagra Brands, Inc., Q1 2026 Pre Recorded Earnings Call, Oct 01, 2025 | 2025-10-01 | 00:00:00 | Q1 2026 | 3555514 |
| 1971299962 | 3622504 | 007343 | Micron Technology, Inc. | Micron Technology, Inc., Q1 2026 Earnings Call, Dec 17, 2025 | 2025-12-17 | 23:00:00 | Q1 2026 | 3622525 |
| 1970999015 | 3623349 | 003362 | Conagra Brands, Inc. | Conagra Brands, Inc., Q2 2026 Pre Recorded Earnings Call, Dec 19, 2025 | 2025-12-19 | 00:00:00 | Q2 2026 | 3623315 |
| 1989973680 | 3681544 | 007343 | Micron Technology, Inc. | Micron Technology, Inc., Q2 2026 Earnings Call, Mar 18, 2026 | 2026-03-18 | 22:00:00 | Q2 2026 | 3681979 |
| 2004549982 | 3757404 | 007343 | Micron Technology, Inc. | Micron Technology, Inc., Q3 2026 Earnings Call, Jun 24, 2026 | 2026-06-24 | 22:00:00 | Q3 2026 | 3757460 |
| 2010260979 | 3781298 | 005568 | The Kraft Heinz Company | The Kraft Heinz Company, Q2 2026 Pre Recorded Earnings Call, Aug 05, 2026 | 2026-08-05 | 00:00:00 | Q2 2026 | 3780939 |

</details>

## Calls written, by year x GICS sector

| year | 10 | 15 | 20 | 25 | 30 | 35 | 40 | 45 | 50 | 55 | 60 | NA | total |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2011 | 144 | 110 | 210 | 290 | 156 | 196 | 252 | 283 | 33 | 97 | 0 | 4 | 1775 |
| 2012 | 159 | 111 | 224 | 309 | 160 | 200 | 274 | 270 | 31 | 97 | 0 | 4 | 1839 |
| 2013 | 166 | 118 | 231 | 324 | 157 | 208 | 287 | 266 | 27 | 115 | 0 | 4 | 1903 |
| 2014 | 169 | 120 | 247 | 330 | 153 | 214 | 290 | 252 | 22 | 97 | 0 | 4 | 1898 |
| 2015 | 152 | 107 | 245 | 330 | 141 | 210 | 299 | 252 | 21 | 93 | 0 | 4 | 1854 |
| 2016 | 144 | 104 | 250 | 327 | 145 | 219 | 305 | 255 | 19 | 95 | 28 | 3 | 1894 |
| 2017 | 130 | 100 | 256 | 314 | 139 | 236 | 242 | 265 | 16 | 101 | 123 | 0 | 1922 |
| 2018 | 121 | 96 | 264 | 294 | 126 | 241 | 252 | 269 | 30 | 106 | 123 | 0 | 1922 |
| 2019 | 112 | 104 | 266 | 249 | 127 | 240 | 255 | 267 | 88 | 105 | 122 | 0 | 1935 |
| 2020 | 102 | 109 | 273 | 230 | 128 | 240 | 246 | 282 | 87 | 101 | 111 | 0 | 1909 |
| 2021 | 89 | 112 | 282 | 241 | 127 | 249 | 246 | 289 | 88 | 105 | 108 | 0 | 1936 |
| 2022 | 84 | 111 | 275 | 227 | 129 | 248 | 252 | 292 | 83 | 108 | 115 | 0 | 1924 |
| 2023 | 88 | 115 | 288 | 209 | 144 | 258 | 274 | 266 | 79 | 114 | 115 | 0 | 1950 |
| 2024 | 81 | 109 | 301 | 202 | 152 | 247 | 278 | 257 | 76 | 115 | 120 | 0 | 1938 |
| 2025 | 87 | 105 | 298 | 197 | 142 | 240 | 285 | 270 | 79 | 115 | 121 | 0 | 1939 |
| 2026 | 62 | 77 | 231 | 134 | 94 | 174 | 220 | 206 | 56 | 86 | 90 | 0 | 1430 |
| total | 1890 | 1708 | 4141 | 4207 | 2220 | 3620 | 4257 | 4241 | 835 | 1650 | 1176 | 23 | 29968 |

Firms (gvkeys): 780. Pairs: 654,447.

## Label classifiers (build-period labels only)

| Score | Labels | Ones | Labels not matched to a pair | OOF AUC |
|---|---|---|---|---|
| EVADE_CLF | 414 | 61 | 0 | 0.629 |
| NONANS_CLF | 414 | 57 | 0 | 0.674 |
