1. What is the sponsor's underlying business or investment objective?

Orchid AI wants to know whether earnings-call language carries tradable information that prices do not yet reflect. Specifically: do tone, hedging, evasiveness and sentiment shifts predict post-earnings drift and volatility beyond the earnings surprise and standard risk controls? If so, the end goal is a cost-aware long-short strategy, with honest evidence on decay, capacity and bias, that could feed into their investment research.

2. Where did the project stand when your team began?

The project started from scratch. We had the sponsor's research brief, but no dataset, code, features or models. Everything presented here was built by our team this semester: the data pipeline, the call-level dataset, the language features, the models and the evaluation framework.

3. What were your team's objectives for this semester?

- Build a reproducible WRDS data pipeline from scratch covering all historical S&P 500 members, with point-in-time index membership and identifier links.
- Engineer two families of language features: Loughran-McDonald dictionary measures and FinBERT sentence-level sentiment.
- Test whether each adds predictive power over a controls-only baseline, using leakage-safe time-series validation.
- Test whether any signal is sector-specific by running the same framework on IT, Healthcare and Communication Services.



5. What are your three most significant accomplishments or discoveries?

5.1. Accomplishment or finding

Starting from nothing, we built a production-grade research pipeline for the full S&P 500.

5.1. Supporting evidence

The pipeline covers 29,698 earnings calls from 2010–2026 across 11 sectors. It joins Capital IQ, CRSP, Compustat and I/B/E/S with date-validated links and point-in-time index membership. It also avoids common pitfalls: it uses unadjusted I/B/E/S data, includes delisting returns, and applies a 4 p.m. ET event-timing rule. FinBERT scored about 2.7M sentences.

5.1. Value to sponsor

The sponsor now has a reusable asset. Any new text feature, sector or horizon can be tested in days rather than weeks, on data the sponsor can trust.

5.2. Accomplishment or finding

We produced a credible, well-powered answer for the full S&P 500. Off-the-shelf language features do not add predictive power once surprise, size, momentum and other controls are included.

5.2. Supporting evidence

The controls-only model reached a test Rank IC of +0.047 for 2022–2024. Two independent selection methods (rank filter and forward selection) added zero LM or FinBERT features. Forcing all features in lowered performance: LM +0.043, FinBERT +0.032.

5.2. Value to sponsor

The sponsor avoids investing in a naive "sentiment score" product. The result also tells them where signal is not, which shapes where to look next.

5.3. Accomplishment or finding

FinBERT sentiment adds signal within sectors, especially against sector-relative returns.

5.3. Supporting evidence

- **IT, sector benchmark:** Model C reached Rank IC +0.092 against +0.073 for controls only.
- **Healthcare:** FinBERT beat the dictionary model in 20 of 24 settings (median +0.033 Rank IC) and beat controls in 21 of 24. The 2025 case reached +0.134, with paired 95% intervals above zero.
- **Recurring features:** sentence counts, sentiment dispersion and prepared-remarks sentiment change were selected again and again.

5.3 Value to sponsor

This gives the sponsor a concrete, testable hypothesis: sector-specific FinBERT models are the most promising route to a tradable signal.

6. What methods, data, and models produced these results?

- **Data:** WRDS (Capital IQ transcripts, CRSP, Compustat, I/B/E/S). Targets are cumulative abnormal returns (CAR) after the call, against the S&P 500 and sector indices.
- **Features:** 16 controls (SUE, size, book-to-market, momentum, volatility, analysts, call structure); 9 Loughran-McDonald dictionary features; 13 FinBERT features.
- **Models:** controls only (A), + dictionary (B), + FinBERT (C), plus "all features forced" variants, using Ridge, ElasticNet and gradient boosting.
- **Evaluation:** time-series CV with train-only selection, test years scored once. Metrics are Rank IC, OOS R², hit rate, quintile spreads and paired bootstrap intervals.

7. How reliable are your results?

Reliable in direction, not yet conclusive in size.

- **Full S&P 500 result:** robust, given about 26,000 labeled calls and agreement between two selection methods.
- **Sector results:** consistent in sign, with FinBERT beating the dictionary model in 20 of 24 Healthcare settings. Effect sizes are modest, though, and many paired intervals still include zero (182 of 200).
- **Small sectors:** some test windows are small; Comm Services has only 67 calls.
- **Remaining biases:** sector membership uses current GICS codes, and real-time transcript availability is unverified.

We report these limits openly. Our next-step proposals are built to resolve them.

8. What can the sponsor actually use today?

- **Data:** the end-to-end pipeline and the clean call-level modeling table (29,698 calls, documented columns, validation report).
- **Features:** a FinBERT feature extractor that runs on any transcript set.
- **Evaluation:** a reusable A/B/C model framework with standardized metrics and paired inference, so any new feature idea can be tested rigorously.
- **Reports:** sector reports (IT, Healthcare, Comm Services) and a sponsor-ready Healthcare brief.

What is not yet deployable is a live trading signal. That requires the cost-aware backtest and timing audit proposed for next semester.

9. What approaches failed or produced disappointing results?

- **Dictionary features:** they added little or nothing in most settings, likely because bag-of-words ignores context.
- **Forcing all features in:** this consistently hurt performance. Parsimony matters.
- **Tree models on sector samples:** gradient boosting and XGBoost overfit about 2,500-call training sets, giving negative Rank IC. Linear models won.
- **Pooling all sectors:** this diluted the language signal.
- **Communication Services:** results were inconclusive because the 67-call holdout cannot separate signal from noise.

Each result narrowed the search space and pointed toward sector-specific, context-aware features.

10. What were your most important blockers?

10.1. Blocker 1

**Building the dataset from scratch.** Linking four WRDS databases correctly is hard. Problems we had to handle include I/B/E/S split-adjustment rounding, multiple transcript versions per call, and identifier links that change over time. We built date-validated joins, used unadjusted I/B/E/S data and stored text in parquet. **Status: resolved.** The cost was a large share of the semester spent on infrastructure before modeling could start.

10.2. Blocker 2

**Label horizon and sample size.** CRSP originally ended in December 2024, which limited test windows, and single-sector holdouts are small. We extended the dataset to mid-2026 and used annual multi-setting tests. **Status: partially resolved.** More time and pooled sector designs would resolve it.

10.3. Blocker 3

**Compute and access.** FinBERT needs GPUs (about 90 minutes per sector on a T4), WRDS requires interactive Duo login, and audio features had no cloud credits. We used a GCP VM and focused on text. **Status: workable, but audio remains blocked.** Dedicated GPU credits and service-account data access would resolve it.

11. Which negative findings should future cohorts treat as established, and which deserve another investigation?

**Established, with credible evidence:**
- Loughran-McDonald dictionary features add no incremental value on the pooled S&P 500.
- Forcing all features in degrades performance.
- On sector-sized samples, regularized linear models outperform trees.

**Failed because of data or time, so worth revisiting:**
- Communication Services, whose 67-call holdout was too small.
- Evasiveness, Q&A relevance, and spoken-versus-press-release divergence, which were scoped but not yet built.
- Audio features, blocked by compute.
- Pooled models with sector interactions, which were never tested.

12. What would you do differently if you were starting the semester again?

- **Prespecify one primary test:** a fixed benchmark, horizon, contrast and metric, set before scoring the holdout.
- **Build one unified evaluation module first,** shared by all sectors. This avoids inconsistent baselines.
- **Select models on CV only.** Report the holdout once.
- **Use point-in-time sector membership from day one.**
- **Start the cost-aware backtest earlier,** so economic significance is measured alongside statistical significance.
- **Get a minimal end-to-end version running in the first weeks,** then refine it. A thin first version of data → features → model would have surfaced data issues sooner.

13. What are the three most promising research questions or development opportunities for the next cohort?

13.1. Proposed project

**Confirmatory, multi-sector FinBERT test.** Run a prespecified test across IT, Healthcare and other sectors on rolling out-of-sample windows, separating sentiment features from transcript-structure features such as sentence counts.

13.1. Why it matters

It would establish whether the sector-level lift is real and repeatable, and where it comes from.

13.1. Expected deliverable

A frozen research spec, plus a results table with paired confidence intervals per sector.

13.2. Proposed project

**Cost-aware long-short backtest with a timing audit.** Measure when transcripts actually arrive, then backtest net of costs with decay and capacity analysis.

13.2. Why it matters

It converts Rank IC into money, which is the sponsor's real question.

13.2. Expected deliverable

A tested strategy prototype with performance attribution and a risk memo.

13.3. Proposed project

**Next-generation features.** These include LLM-scored analyst-question evasiveness, Q&A relevance, and divergence between spoken remarks and the written release.

13.3. Why it matters

These were in the original scope, and context-aware measures are where the evidence points.

13.3. Expected deliverable

New validated features plugged into the existing A/B/C framework.

14. What evidence suggests these opportunities are worth pursuing?

- **Context beats word counts:** FinBERT beat dictionary features in 20 of 24 Healthcare settings and lifted IT sector-relative Rank IC from +0.073 to +0.092.
- **Stable features:** the same FinBERT features (sentence counts, sentiment dispersion, prepared-tone change) were repeatedly selected across sectors and years. That stability suggests real signal rather than noise.
- **Strongest single result:** the 2025 Healthcare case cleared zero under both inference methods.

Together, this points toward richer context-aware features, which motivates the evasiveness and divergence measures.

15. What would a successful next semester look like?

- **A confirmed signal:** a prespecified test confirming, or cleanly rejecting, the FinBERT lift in at least two sectors, with intervals excluding zero.
- **Economic value:** a cost-aware long-short backtest showing returns net of transaction costs, with decay and capacity analysis.
- **Wider coverage:** at least one new context-aware feature (evasiveness or press-release divergence) tested.
- **A sponsor decision:** a clear go/no-go recommendation on building a production signal.

16. What would the next cohort need to succeed?

- **Data access:** continued WRDS access, ideally non-interactive, plus a source of real transcript timestamps.
- **Compute:** GPU credits for FinBERT or LLM scoring, and possibly audio.
- **A handover:** the current pipeline, modeling table and documentation, with a short walkthrough.
- **Sponsor alignment:** agreement early in the semester on the primary benchmark, horizon and success metric.
- **Expertise:** occasional guidance on realistic transaction-cost and capacity assumptions from a practitioner.


18. How can the next cohort become productive quickly?

1. Define priorities early on
2. Request compute power (GPU)
3. Define data scope


19. Why should the sponsoring organization continue this project next semester?

The expensive groundwork is done. A clean, validated, reusable dataset and evaluation framework now exist, and the research question has narrowed from "does language matter?" to "how much, in which sectors, and is it tradable?". The sector-level FinBERT results are consistent enough to justify a confirmatory test. The marginal cost of the next semester is low relative to what has been built, and the potential payoff, a differentiated signal, is high.

20. What decisions or commitments do you need from the sponsor?

- **Approve priorities:** the next semester's research focus, namely the confirmatory sector test, the backtest and new features.
- **Fix the primary test:** the benchmark (market or sector), the horizon (e.g. CAR[+2,+20]) and the success metric.
- **Data:** confirm continued WRDS access, and if possible provide transcript timestamp data.
- **Compute:** a GPU or cloud budget.
- **Guidance:** assumptions on transaction costs, capacity and target holding period.
- **Mentoring:** a regular check-in with a practitioner.

21. What is the single most compelling reason to continue the project?

The project now has both rigorous infrastructure and a consistent lead: context-aware FinBERT sentiment improves return rankings within sectors where simple word counts fail. One more semester can turn that lead into a confirmed, cost-tested signal on top of a pipeline that already works.

23. What are your five most important presentation visuals?

One visual per section of the 15-minute talk.

23.1 Visual 1: Executive summary and sponsor objectives (2 minutes)

**Question-to-answer slide:** the sponsor's question on the left; three headline numbers on the right (29,698 calls built; S&P 500: language adds 0 over controls' +0.047 Rank IC; IT +0.073 → +0.092 with FinBERT).

**Single message:** Earnings-call language does predict returns, but only within sectors, not across the whole market.

23.2 Visual 2: Inherited work and this semester's approach (2 minutes)

**"Started from zero" pipeline diagram:** empty starting point → WRDS sources → identity links → targets → text features → A/B/C model framework.

**Single message:** We built the entire research foundation from scratch this semester, and the sponsor can trust it and reuse it.

23.3 Visual 3: Accomplishments, findings, and validation (5 minutes)

**Two-panel chart:** left, full S&P 500 test Rank IC for A, B, B*, C, C*; right, IT and Healthcare controls versus FinBERT (Healthcare 21 of 24 settings positive). Footer: train-only selection, paired bootstrap intervals.

**Single message:** Generic sentiment adds nothing market-wide, but FinBERT consistently improves return rankings within sectors.

23.4 Visual 4: Failures, blockers, and lessons learned (2 minutes)

**Three-column table: what failed → why → lesson** (dictionary features, forced features, trees, pooling, 67-call Comm Services holdout, data-build time).

**Single message:** Every failure ruled something out and pointed us toward sector-specific, context-aware features.

23.5 Visual 5: Next-semester opportunities and research plan (4 minutes)

**Roadmap: done versus next** (confirmatory test → cost-aware backtest → new features), with success criteria and sponsor asks beneath.

**Single message:** One more semester can turn a consistent lead into a confirmed, cost-tested trading signal.

24. What challenging questions might the sponsor ask?

- **"Are these ICs too small to trade?"** Modest ICs are typical for event signals; the backtest will measure net returns.
- **"Did you overfit?"** Healthcare selection was train-only; a prespecified confirmatory test comes next.
- **"Why sectors but not the full market?"** Language norms differ by sector, so pooling dilutes the signal.
- **"Is sentence count just call length?"** Possibly; separating counts from sentiment is our first proposed test.
- **"Any look-ahead bias?"** Event timing is controlled; sector labels and transcript timestamps are next.
- **"Why no backtest yet?"** We started from scratch, and a backtest is only as good as its data, so we built and validated the dataset first.
