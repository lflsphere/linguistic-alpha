"use strict";

const data = JSON.parse(document.getElementById("research-data").textContent);
const research = document.getElementById("research");
const common = "Five-model common (train-imputed)";
const intersection = "Five-model horizon intersection";
const models = ["A", "B", "B*", "C", "C*"];
const colors = ["#112d43", "#08766c", "#785aa0", "#2b73b2", "#c05f20"];
const years = [2024, 2025, 2026];
const benchmarks = ["S&P 500", "Healthcare"];
const horizons = [5, 20];
const schedules = ["Expanding", "Rolling 5Y"];
const representations = ["LM", "FinBERT"];
const matrixNames = ["LM × LM", "FinBERT × FinBERT", "LM × FinBERT", "LM × controls", "FinBERT × controls"];
const modelNames = Object.fromEntries(models.map((name, i) => [name, colors[i]]));
const displayYear = (year) => year === 0 ? "Development 2010–2023" : year === 2026 ? "Test 2026 (Partial)" : "Test " + year;
const car = (horizon) => "CAR[+2,+" + horizon + "]";
const finite = (value) => typeof value === "number" && Number.isFinite(value);
const num = (value) => finite(value) ? value.toFixed(4) : "—";
const pct = (value) => finite(value) ? (100 * value).toFixed(2) + "%" : "—";
const pp = (value) => finite(value) ? (100 * value).toFixed(2) + " pp" : "—";
const signed = (value) => finite(value) ? (value >= 0 ? "+" : "") + value.toFixed(4) : "—";
const esc = (text) => String(text ?? "—").replaceAll("&", "&amp;").replaceAll("<", "&lt;")
  .replaceAll(">", "&gt;").replaceAll('"', "&quot;").replaceAll("'", "&#39;");
const by = (table, filters = {}) => data[table].filter((row) => Object.entries(filters)
  .every(([key, value]) => row[key] === value));
const caption = (text) => "<p class='caption'>" + esc(text) + "</p>";
const heading = (text) => "<h3>" + esc(text) + "</h3>";
const direct = (label, value) => "<strong>" + esc(label) + "</strong>" + esc(value);

const lm = {
  tone_prepared: ["Prep tone", "Dictionary tone in prepared remarks."],
  tone_answers: ["Ans tone", "Dictionary tone in management answers."],
  hedging_prepared: ["Prep hedge", "Weak modal words per 1,000 prepared-remark words."],
  uncertainty_answers: ["Ans uncertainty", "Uncertainty words per 1,000 answer words."],
  tone_gap: ["Ans−prep LM", "Answer tone minus prepared tone."],
  tone_prepared_chg: ["Δ prep tone", "Prepared tone relative to recent earlier calls."],
  tone_answers_chg: ["Δ ans tone", "Answer tone relative to recent earlier calls."],
  hedging_answers_chg: ["Δ ans hedge", "Answer hedging relative to recent earlier calls."],
};
const control = {
  sue: ["SUE", "Standardized earnings surprise."],
  surprise_pct: ["Relative surprise", "Reported EPS surprise relative to the forecast EPS magnitude."],
  vol_pre: ["Pre-event vol", "Pre-call stock return volatility."],
  "car_-1_1": ["Early CAR", "Benchmark-adjusted returns on sessions −1 through +1."],
  mom: ["Momentum", "Historical stock return momentum."],
  bm: ["Book-to-market", "Book equity relative to market equity."],
  size: ["Size", "Firm market capitalization on the pipeline's feature scale."],
  dispersion: ["Dispersion", "Cross-analyst EPS forecast dispersion."],
  avg_answer_words: ["Mean ans words", "Mean word count across answer turns."],
  words_prepared: ["Prep words", "Word count in prepared remarks."],
  words_questions: ["Question words", "Word count in analyst questions."],
  words_answers: ["Answer words", "Word count in management answers."],
  n_questions: ["Questions", "Number of question turns."],
  n_answers: ["Answers", "Number of answer turns."],
  n_analysts: ["Analysts", "Number of analyst forecasts contributing to consensus."],
  prior_calls_observed: ["Prior calls", "Calls already observed for the same PERMNO."],
};
const fbLabels = ["Prep mean", "Ans mean", "Prep positive", "Prep negative", "Ans positive", "Ans negative",
  "Prep SD", "Ans SD", "Prep−ans FB", "Δ prep mean", "Δ ans mean", "Prep sentences", "Ans sentences"];
const names = {...Object.fromEntries(Object.entries(lm).map(([key, [label]]) => [key, label])),
  ...Object.fromEntries(Object.entries(control).map(([key, [label]]) => [key, label])),
  ...Object.fromEntries(Object.keys(data.definitions).map((key, index) => [key, fbLabels[index]]))};
const brief = (name) => names[name] || name;
const featureList = (value) => value ? value.split("|").map(brief).join(", ") : "Controls only";

function cell(value, className = "") {
  return {value, className};
}
function table(headers, rows) {
  const head = "<thead><tr>" + headers.map((name) => "<th scope='col'>" + esc(name) + "</th>").join("") + "</tr></thead>";
  if (rows.length === 0) return caption("No observations satisfy the selected display requirements.");
  const body = rows.map((row) => "<tr>" + row.map((value) => {
    const item = value && typeof value === "object" ? value : cell(value);
    return "<td class='" + item.className + "'>" + esc(item.value) + "</td>";
  }).join("") + "</tr>").join("");
  return "<div class='table-scroll'><table>" + head + "<tbody>" + body + "</tbody></table></div>";
}
const range = (x, a, b) => Math.min(Math.max(x, a), b);
const tint = (value) => {
  if (!finite(value)) return "#e5edf0";
  const white = [241, 245, 246];
  const end = value < 0 ? [112, 70, 148] : [9, 121, 110];
  return "rgb(" + end.map((color, i) => Math.round(white[i] + Math.abs(value) * (color - white[i]))).join(",") + ")";
};

function heatmap(records) {
  if (!records.length) return caption("No correlation matrix is available for this context.");
  const rowKeys = [...new Set(records.map((row) => row.row))];
  const colKeys = [...new Set(records.map((row) => row.column))];
  const values = new Map(records.map((row) => [row.row + "|" + row.column, row]));
  const header = "<thead><tr><th scope='col'>Factor</th>" + colKeys.map((column) =>
    "<th scope='col' title='" + esc(column) + "'>" + esc(brief(column)) + "</th>").join("") + "</tr></thead>";
  const body = rowKeys.map((row) => "<tr><th scope='row' title='" + esc(row) + "'>" + esc(brief(row)) + "</th>" +
    colKeys.map((column) => {
      const value = values.get(row + "|" + column);
      const rho = value.rank_corr;
      const fg = finite(rho) && Math.abs(rho) >= .72 ? "white" : "#17344c";
      return "<td style='background:" + tint(rho) + ";color:" + fg + "' title='" + esc(row + " × " + column +
        " | n=" + value.events + " | ρ=" + num(rho)) + "'>" + (finite(rho) ? rho.toFixed(2) : "—") + "</td>";
    }).join("") + "</tr>").join("");
  return "<div class='heat'><div class='table-scroll'><table>" + header + "<tbody>" + body + "</tbody></table></div>" +
    "<div class='key'>−1 <span class='gradient'></span> +1 <span> Spearman correlation</span></div></div>";
}

function svg(content, height, label) {
  return "<div class='chart-scroll'><svg viewBox='0 0 700 " + height +
    "' role='img' aria-label='" + esc(label) + "' xmlns='http://www.w3.org/2000/svg'>" +
    content + "</svg></div>";
}
function tick(x, y, text, anchor = "middle", size = 12, fill = "#49647a") {
  return "<text x='" + x + "' y='" + y + "' text-anchor='" + anchor +
    "' font-size='" + size + "' fill='" + fill + "'>" + esc(text) + "</text>";
}
function bars(groups, series, valueLabel = num, label = "Spearman Rank IC") {
  const width = 700, height = 202, left = 47, right = 12, top = 24, bottom = 39;
  const vals = groups.flatMap((group) => series.map((key) => group.values[key])).filter(finite);
  if (!vals.length) return caption("No valid estimates for these selections.");
  const min = Math.min(0, ...vals), max = Math.max(0, ...vals);
  const padding = Math.max(.014, (max - min) * .15);
  const lo = min - padding, hi = max + padding;
  const y = (value) => top + (hi - value) * (height - top - bottom) / (hi - lo);
  const legend = series.map((key, index) => {
    const x = left + index * Math.min(140, 570 / series.length);
    const color = modelNames[key] || colors[index];
    return "<rect x='" + x + "' y='3' width='9' height='9' rx='2' fill='" + color + "'/>" +
      tick(x + 14, 12, key, "start", 10);
  }).join("");
  let lines = "";
  for (const value of [min, 0, max].filter((one, idx, list) => list.indexOf(one) === idx)) {
    lines += "<line x1='" + left + "' x2='" + (width - right) + "' y1='" + y(value) + "' y2='" +
      y(value) + "' stroke='" + (value === 0 ? "#597280" : "#dce7eb") + "'/>" + tick(left - 5, y(value) + 4, valueLabel(value), "end", 10);
  }
  const slot = (width - left - right) / groups.length;
  const barWidth = Math.min(94, .94 * slot / series.length);
  const marks = groups.map((group, index) => {
    const center = left + (index + .5) * slot;
    return tick(center, height - 11, group.label, "middle", 11) + series.map((key, j) => {
      const value = group.values[key];
      if (!finite(value)) return "";
      const x = center + (j - (series.length - 1) / 2) * barWidth;
      const labelY = value >= 0 ? y(value) - 4 : y(value) + 12;
      return "<rect x='" + (x - .43 * barWidth) + "' y='" + Math.min(y(0), y(value)) +
        "' width='" + (.86 * barWidth) + "' height='" + Math.max(1, Math.abs(y(0) - y(value))) +
        "' fill='" + (modelNames[key] || colors[j]) + "'><title>" +
        esc(group.label + " · " + key + " " + valueLabel(value)) + "</title></rect>" +
        tick(x, labelY, valueLabel(value), "middle", Math.min(11, .28 * barWidth), "#17344c");
    }).join("");
  }).join("");
  return svg(legend + lines + marks, height, label);
}

function intervalPlot(rows) {
  const usable = rows.filter((row) => finite(row.point));
  if (!usable.length) return caption("No point estimate is available for this context.");
  const lo = Math.min(0, ...usable.flatMap((row) => [row.lower, row.point]).filter(finite)) - .015;
  const hi = Math.max(0, ...usable.flatMap((row) => [row.upper, row.point]).filter(finite)) + .015;
  const extent = 34 + 27 * usable.length;
  const x = (value) => 145 + 525 * (value - lo) / (hi - lo);
  let chart = "<line x1='" + x(0) + "' x2='" + x(0) + "' y1='6' y2='" + (extent - 15) +
    "' stroke='#a8bcc8' stroke-dasharray='4 3'/>" + tick(x(0), extent - 2, "0", "middle", 10);
  for (const [index, row] of usable.entries()) {
    const y = 15 + 27 * index;
    chart += tick(140, y + 4, row.contrast || brief(row.factor), "end");
    if (row.available) {
      chart += "<line x1='" + x(row.lower) + "' x2='" + x(row.upper) + "' y1='" + y + "' y2='" + y +
        "' stroke='#08766c' stroke-width='2.5'/>";
    }
    chart += "<circle cx='" + x(row.point) + "' cy='" + y + "' r='4' fill='#112d43'><title>" +
      esc(signed(row.point) + " [" + num(row.lower) + ", " + num(row.upper) + "]") + "</title></circle>";
  }
  return svg(chart, extent, "Paired point estimates and available 95% intervals");
}

function stabilityChart(rows) {
  const grouped = new Map();
  for (const row of rows) {
    if (!grouped.has(row.factor)) grouped.set(row.factor, []);
    grouped.get(row.factor).push(row);
  }
  const yearsShown = [...new Set(rows.map((row) => row.year))].filter((year) => year !== 0).sort();
  const all = rows.filter((row) => yearsShown.includes(row.year) && finite(row.rank_ic));
  if (all.length < 2 || yearsShown.length < 2) return caption("Insufficient valid annual observations.");
  const height = 174, left = 50, top = 16, right = 664;
  const lo = Math.min(0, ...all.map((row) => row.rank_ic)) - .02;
  const hi = Math.max(0, ...all.map((row) => row.rank_ic)) + .02;
  const x = (year) => left + (right - left) * yearsShown.indexOf(year) / (yearsShown.length - 1);
  const y = (value) => top + (hi - value) * (height - top - 35) / (hi - lo);
  let chart = "<line x1='" + left + "' x2='" + right + "' y1='" + y(0) + "' y2='" + y(0) +
    "' stroke='#a3bbc5' stroke-dasharray='4 3'/>" + tick(left - 7, y(0) + 4, "0", "end");
  chart += yearsShown.map((year) => tick(x(year), height - 7, String(year) + (year === 2026 ? " partial" : ""))).join("");
  [...grouped.entries()].forEach(([key, values], index) => {
    const line = values.filter((value) => yearsShown.includes(value.year) && finite(value.rank_ic));
    const color = colors[index % colors.length];
    chart += "<polyline fill='none' stroke='" + color + "' stroke-width='2' points='" +
      line.map((value) => x(value.year) + "," + y(value.rank_ic)).join(" ") + "'/>";
    chart += line.map((value) => "<circle cx='" + x(value.year) + "' cy='" + y(value.rank_ic) +
      "' r='4' fill='" + color + "'><title>" + esc(brief(key) + " · " + num(value.rank_ic)) + "</title></circle>").join("");
    chart += tick(left + index * 170, 11, brief(key), "start", 11, color);
  });
  return svg(chart, height, "Annual outer-test Rank IC");
}

const descr = {
  overview: "Predict post-earnings abnormal returns from observed Healthcare earnings-call information. The experiment separates financial controls, LM dictionary signals, and FinBERT contextual sentiment.",
  timing: "Annual refits use only labels mature strictly before the refit. Three expanding validation folds inside each training window enforce the same rule. S&P 500-adjusted and Healthcare-adjusted CAR sum stock-minus-benchmark daily returns.",
  limitation: "The signal is timestamped after the Day +1 close and before the Day +2 open. The first daily return begins at the previous close and includes overnight movement. The results describe event forecasts under the original daily-label convention.",
  pipeline: "Capital IQ & CRSP → timestamped event cohorts → mature historical training → frozen annual predictions → paired test evaluation",
};

function annualGroups(rows, field) {
  return [...new Set(rows.map((row) => row[field]))].map((key) => ({
    label: field === "horizon" ? car(key) : key,
    values: Object.fromEntries(rows.filter((row) => row[field] === key).map((row) => [row.model, row.rank_ic])),
  }));
}

function modelTable(rows, field = "model") {
  const best = new Map();
  for (const row of rows) {
    const context = [row.year, row.benchmark, row.horizon, row.window, row.sample].join("|");
    if (finite(row.rank_ic) && (!best.has(context) || row.rank_ic > best.get(context))) best.set(context, row.rank_ic);
  }
  return table([field === "model" ? "Model" : "Setting / model", "Events", "Issuers", "Rank IC", "Pearson IC", "Q5−Q1"],
    rows.map((row) => {
      const context = [row.year, row.benchmark, row.horizon, row.window, row.sample].join("|");
      const label = field === "model" ? row.model : car(row.horizon) + " · " + row.window + " · " + row.model;
      return [label, row.events, row.issuers, cell(num(row.rank_ic), best.get(context) === row.rank_ic ? "best" : ""),
        num(row.pearson), pp(row.q5_minus_q1)];
    }));
}

function factorTable(rows) {
  return table(["Factor", "Events", "Eligible", "Issuers", "Rank IC", "Pearson IC", "Q5−Q1"],
    rows.map((row) => [brief(row.factor), row.events, row.eligible_events, row.issuers,
      num(row.rank_ic), num(row.pearson), pp(row.q5_minus_q1)]));
}

function intervalTable(rows) {
  return table(["Contrast", "Method", "Events", "Issuers", "Point", "Lower 95%", "Upper 95%"],
    rows.map((row) => [row.contrast || brief(row.factor), row.method, row.events, row.issuers, signed(row.point),
      num(row.lower), num(row.upper)]));
}

function incremental(rows) {
  return table(["Contrast", "Rank IC gain", "Events", "Issuers"],
    rows.map((row) => [row.contrast, cell(signed(row.delta_rank_ic),
      row.delta_rank_ic > 0 ? "positive" : row.delta_rank_ic < 0 ? "negative" : ""), row.events, row.issuers]));
}

function quintiles(rows, key) {
  return table([key === "model" ? "Model" : "Factor", "Quintile", "Events", "Mean CAR", "Median CAR"],
    rows.map((row) => [key === "model" ? row.model : brief(row.factor), "Q" + row.quintile,
      row.n, pct(row.mean_car), pct(row.median_car)]));
}

function selection(rows) {
  return table(["Model", "Selection route", "Selected inputs", "Estimator", "CV Rank IC"],
    rows.map((row) => [row.model, row.route, cell(featureList(row.selection_order || row.features), "long"),
      row.estimator, num(row.inner_rank_ic)]));
}

function summaryData() {
  const wins = [], inc = [], windows = [], time = [];
  for (const year of years) {
    const p = by("annual_model_performance", {year, sample: common});
    const w = new Map(models.map((model) => [model, 0]));
    for (const benchmark of benchmarks) for (const horizon of horizons) for (const window of schedules) {
      const group = p.filter((row) => row.benchmark === benchmark && row.horizon === horizon && row.window === window);
      const best = Math.max(...group.map((row) => row.rank_ic));
      for (const row of group.filter((value) => value.rank_ic === best)) w.set(row.model, w.get(row.model) + 1);
    }
    wins.push([displayYear(year), ...models.map((name) => w.get(name))]);
    const annual = by("annual_increments", {year, sample: common});
    inc.push([displayYear(year), ...["B−A", "B*−A", "C−A", "C*−A"].map((key) => {
      const group = annual.filter((row) => row.contrast === key);
      return group.filter((row) => row.delta_rank_ic > 0).length + "/" + group.length;
    })]);
    const windowsRows = by("window_comparison", {year});
    const horizonRows = by("horizon_comparison", {year});
    windows.push([displayYear(year), windowsRows.filter((row) => row.rolling_minus_expanding > 0).length +
      "/" + windowsRows.length, horizonRows.filter((row) => row.long_minus_short > 0).length + "/" + horizonRows.length]);
    const caseRows = by("baseline_training_cohorts", {year});
    for (const schedule of schedules) {
      const sample = caseRows.filter((row) => row.window === schedule);
      const count = sample.map((row) => row.train_events);
      const min = Math.min(...count), max = Math.max(...count);
      const first = sample[0];
      time.push([displayYear(year), schedule, first.train_start + "–" + (year - 1),
        first.refit_cutoff.slice(0, 10) + " 16:00 ET", min === max ? min : min + "–" + max, first.train_issuers]);
    }
  }
  return {wins, inc, windows, time};
}
const summary = summaryData();

function insights() {
  const language = data.paired_inference.filter((row) => row.family === "Language" &&
    row.method === "Issuer cluster" && row.contrast !== "C−B" && row.available);
  const exclude = (rows) => rows.filter((row) => row.lower > 0 || row.upper < 0).length;
  const fb = language.filter((row) => row.contrast.startsWith("C"));
  const lm = language.filter((row) => row.contrast.startsWith("B"));
  const empty = by("finbert_selected_features", {model: "C"}).filter((row) => row.feature_count === 0).length;
  const selected = by("finbert_forward_paths").filter((row) => row.accepted);
  const frequency = Object.entries(selected.reduce((count, row) =>
    ({...count, [row.candidate]: (count[row.candidate] || 0) + 1}), {})).sort((a, b) => b[1] - a[1]);
  return heading("Language increments") + table(["Test year", "B−A > 0", "B*−A > 0", "C−A > 0", "C*−A > 0"], summary.inc) +
    caption("Count of positive increments across the eight annual combinations for each contrast.") +
    "<p>" + esc("Issuer-cluster intervals exclude zero in " + exclude(lm) + "/" + lm.length + " LM and " + exclude(fb) +
      "/" + fb.length + " FinBERT comparisons with available intervals. Shared disclosure content and the early return may already reflect some information in the language measures. Pointwise intervals and correlated tests limit the strength of any general claim.") + "</p>" +
    heading("Window and horizon dependence") +
    table(["Test year", "Rolling > Expanding", "Long > short"], summary.windows) +
    "<p>" + esc("Recency can matter when relationships drift. A longer return horizon may allow more time for information to be reflected in prices and can also capture other news. All horizon comparisons use shared events and separate targets.") + "</p>" +
    heading("Feature stability and inference") +
    "<p>" + esc("Model C selected zero language features in " + empty + "/24 settings. " +
      (frequency.length ? brief(frequency[0][0]) + " was the most frequently selected FinBERT input (" + frequency[0][1] + "/24). " : "") +
      "The correlation matrices and quarterly factor tables show overlap and changing signs by context. Selection frequency describes the fitted procedure. The short 2026 sample supports issuer intervals where coverage is sufficient, with too few independent calendar blocks for full time inference.") + "</p>" +
    caption("C−B compares independently chosen modeling procedures. Quintile curves show realized event CAR, with overlapping events and no trading-cost or portfolio simulation.");
}

function featureDictionary(dictionary) {
  return table(["Feature", "Definition"], Object.entries(dictionary).map(([field, description]) =>
    [field, cell(Array.isArray(description) ? description[1] : description, "long")]));
}

function reportTitle(title) {
  return "<h2>" + esc(title) + "</h2>";
}

function renderDashboard() {
  const nav = [
    ["overview", "Overview", "Research Overview"], ["design", "Design", "Experimental Design"],
    ["coverage", "Coverage", "Factor Universe and Coverage"], ["factor", "Factor IC", "Single-Factor Rank IC"],
    ["stability", "Stability", "Factor IC Stability"], ["raw-quintiles", "Raw Quintiles", "Raw Factor Quintiles"],
    ["correlations", "Correlations", "Correlation and Redundancy"], ["construction", "Models", "Model Construction and Selection"],
    ["model-results", "Results", "Annual Model Results"], ["increments", "Incremental IC", "Incremental Linguistic Alpha"],
    ["windows", "Windows", "Training Window Comparison"], ["horizons", "Horizons", "Prediction Horizon Comparison"],
    ["model-quintiles", "Prediction Quintiles", "Model Prediction Quintiles"], ["inference", "Inference", "Statistical Inference"],
    ["insights", "Insights", "Research Insights"],
  ];
  document.getElementById("navigation").innerHTML = nav.map(([id, title]) => "<a href='#" + id + "'>" + esc(title) + "</a>").join("");
  research.innerHTML = nav.map(([id, _, title]) => "<section id='" + id + "'>" + reportTitle(title) +
    "<div data-section='" + id + "'></div></section>").join("");
  const selectors = [
    ["year", "Test year", years, displayYear], ["benchmark", "Benchmark", benchmarks, String],
    ["horizon", "Prediction horizon", horizons, car], ["window", "Training schedule", schedules, String],
    ["model", "Model family", ["All", ...models], String],
    ["representation", "Linguistic representation", representations, String],
    ["matrix", "Correlation comparison", matrixNames, String],
    ["cohort", "Factor analysis period", ["Outer test", "Development"], String],
  ];
  research.querySelector("[data-section=overview]").innerHTML = "<p>" + esc(descr.overview) + "</p>" +
    "<p class='statement'>" + esc(descr.pipeline) + "</p><div class='controls'>" + selectors
      .map(([id, label, values, format]) => "<label for='select-" + id + "'>" + esc(label) +
        "<select id='select-" + id + "'>" + values.map((value) => "<option value='" + esc(value) + "'>" +
          esc(format(value)) + "</option>").join("") + "</select></label>").join("") + "</div>";
  const set = (id, content) => { research.querySelector("[data-section=" + id + "]").innerHTML = content; };
  const update = () => {
    const state = Object.fromEntries(selectors.map(([key]) => [key, document.getElementById("select-" + key).value]));
    const year = Number(state.year), horizon = Number(state.horizon), benchmark = state.benchmark, window = state.window;
    const context = {year, horizon, benchmark, window};
    const factorContext = {year: state.cohort === "Development" ? 0 : year, benchmark, horizon,
      representation: state.representation};
    const modelFilter = (rows) => state.model === "All" ? rows : rows.filter((row) => row.model === state.model);
    const factors = by("factor_ic", factorContext);
    const ct = by("finbert_native_coverage", context)[0];
    const annual = by("annual_model_performance", {...context, sample: common});
    set("design", "<p>" + esc(descr.timing) + "</p><p>" + esc(descr.limitation) + "</p>" +
      table(["Test year", "Schedule", "Training years", "Refit", "Calls", "Issuers"], summary.time));
    set("coverage", caption("The archived LM candidate set drops the exact duplicate hedging_answers. Feature coverage is measured before imputation.") +
      "<div class='summary-grid'>" + [[ct.events, "Matched model events"], [ct.transcript_covered, "FinBERT text covered"],
        [ct.features_complete, "All 13 FinBERT predictors observed"]].map(([v, label]) =>
          "<div>" + direct(v, label) + "</div>").join("") + "</div>" + coverageTable() +
      "<details><summary>Financial and structural controls</summary>" + featureDictionary(control) + "</details>" +
      "<details><summary>LM8 dictionary</summary>" + featureDictionary(lm) + "</details>" +
      "<details><summary>FinBERT13 dictionary</summary>" + featureDictionary(data.definitions) + "</details>");
    set("factor", caption(displayYear(factorContext.year) + " · " + benchmark + " · " + car(horizon) + " · " + state.representation) +
      factorTable([...factors].sort((a, b) => (b.rank_ic ?? -2) - (a.rank_ic ?? -2))) +
      intervalTable(by("factor_inference", {...factorContext, method: "Issuer cluster"})));
    const foci = state.representation === "LM" ? ["tone_answers", "tone_answers_chg"] : ["fb_sent_prepared", "fb_sent_answers"];
    set("stability", stabilityChart(by("factor_ic", {benchmark, horizon, representation: state.representation})
      .filter((row) => foci.includes(row.factor))) +
      table(["Quarter", "Factor", "Pairs", "Issuers", "Rank IC"], by("factor_stability", factorContext)
        .filter((row) => foci.includes(row.factor))
        .map((row) => [row.quarter, brief(row.factor), row.events, row.issuers, num(row.rank_ic)])));
    const examples = by("factor_quintiles", factorContext).filter((row) => foci.includes(row.factor));
    set("raw-quintiles", caption("Raw-factor sorts use observed language values. They describe event CAR by group.") +
      quintiles(examples, "factor") + "<details><summary>All supported factor sorts</summary>" +
      quintiles(by("factor_quintiles", factorContext), "factor") + "</details>");
    set("correlations", caption("Pairwise observed predictors. The values use a common −1 to +1 scale and require 20 pairs.") +
      heatmap(by("correlations", {year: factorContext.year, benchmark, horizon, block: state.matrix})));
    const specs = [...by("baseline_selected_features", context), ...by("finbert_selected_features", context)];
    const attempts = by("finbert_forward_paths", context);
    set("construction", modelDefinitions() + selection(modelFilter(specs)) +
      "<details><summary>FinBERT forward-selection paths</summary>" + pathTable(attempts) + "</details>" +
      "<details><summary>FinBERT estimator grid and coefficients</summary>" +
      table(["Candidate block", "Estimator", "CV Rank IC"], by("finbert_estimator_cv", context)
        .map((row) => [cell(featureList(row.features), "long"), row.estimator, num(row.mean_rank_ic)])) +
      table(["Model", "Predictor", "Coefficient / n", "Absolute share"], modelFilter(by("finbert_feature_importance", context))
        .map((row) => [row.model, brief(row.feature), signed(row.scaled_by_train_n), pct(row.absolute_share)])) + "</details>");
    set("model-results", caption(displayYear(year) + " · " + benchmark + " · " + car(horizon) + " · " + window) +
      modelTable(modelFilter(annual)) + bars([{label: window, values: Object.fromEntries(annual.map((row) =>
        [row.model, row.rank_ic]))}], state.model === "All" ? models : [state.model]) +
      "<details><summary>Original A/B/B* full cohort and fully observed FinBERT sensitivity</summary>" +
      modelTable(by("original_ab_performance", {...context, sample: "Full horizon"})) +
      modelTable(modelFilter(by("annual_model_performance", {...context, sample: "All 13 observed (sensitivity)"}))) + "</details>");
    const inc = by("annual_increments", {...context, sample: common});
    set("increments", incremental(inc) + bars([{label: car(horizon), values: Object.fromEntries(inc.map((row) =>
      [row.contrast, row.delta_rank_ic]))}], inc.map((row) => row.contrast), signed, "Incremental Rank IC"));
    const selectionModels = state.model === "All" ? models : [state.model];
    const full = modelFilter(by("annual_model_performance", {year, benchmark, horizon, sample: common}));
    const matched = modelFilter(by("annual_model_performance", {year, benchmark, window, sample: intersection}));
    set("windows", caption("Same calls and realized labels for both schedules.") +
      bars(annualGroups(full, "window"), selectionModels) + modelTable(full, "settings"));
    set("horizons", caption("Each model is scored on the same test calls across horizons, against its own realized target.") +
      bars(annualGroups(matched, "horizon"), selectionModels) + modelTable(matched, "settings"));
    const sorts = modelFilter(by("model_quintiles", {...context, sample: common}));
    set("model-quintiles", caption("Q1 is the lowest predicted-score group. Realized event CAR is expressed in percent.") +
      quintiles(sorts, "model") + heading("Q5−Q1 separation") +
      table(["Model", "Spread", "Profile"], modelFilter(annual).map((row) => [row.model, pp(row.q5_minus_q1), row.monotonicity])));
    const intervals = by("paired_inference", {...context, family: "Language"});
    set("inference", intervalPlot(intervals.filter((row) => row.method === "Issuer cluster")) + intervalTable(intervals) +
      caption("Pointwise 95% paired issuer and two-month moving-block intervals. Missing bounds indicate insufficient support. Partial 2026 has three nominal calendar blocks."));
    set("insights", insights() + appendices());
  };
  research.querySelectorAll("select").forEach((input) => input.addEventListener("change", update));
  update();
}

function coverageTable() {
  return table(["Year", "Universe", "Text", "Prepared", "Answers", "Cached"],
    data.source_coverage.filter((row) => row.sector === "Health Care").map((row) =>
      [row.year, row.eligible_universe, row.text_accessible, row.prepared_available, row.answers_available, row.validated_cached]));
}
function modelDefinitions() {
  return table(["Model", "Predictors", "Training-only selection"],
    [["A", "Financial and structural controls", "Estimator grid"],
     ["B", "Controls + selected LM8", "Greedy forward or screened combination, frozen as run"],
     ["B*", "Controls + all LM8", "Estimator grid"],
     ["C", "Controls + selected FinBERT13", "Greedy forward with >0.001 gain per step"],
     ["C*", "Controls + all FinBERT13", "Estimator grid"]]);
}
function pathTable(rows) {
  return table(["Step", "Candidate", "Prior inputs", "Estimator", "CV Rank IC", "Gain", "Added"],
    rows.map((row) => [row.step, brief(row.candidate), cell(featureList(row.prior_selection), "long"),
      row.estimator, num(row.inner_rank_ic), signed(row.gain), row.accepted ? "Yes" : ""]));
}
function appendices() {
  const tables = ["annual_model_performance", "annual_increments", "original_ab_performance", "window_comparison",
    "horizon_comparison", "paired_inference", "model_quintiles", "factor_ic", "factor_inference", "factor_stability",
    "factor_quintiles", "correlations", "feature_missingness", "finbert_native_coverage", "finbert_selected_features",
    "finbert_forward_paths", "finbert_estimator_cv", "finbert_feature_importance"];
  return heading("Reproducibility tables") + "<p class='inline-links'>" + tables.map((name) =>
    "<a href='" + name + ".csv'>" + esc(name.replaceAll("_", " ")) + "</a>").join("") + "</p>";
}

function printReport() {
  let number = 0;
  const pages = [];
  const page = (title, contents, style = "") => {
    number += 1;
    pages.push("<section class='" + style + "'><p class='print-head'>Healthcare Earnings Call Research · LM and FinBERT</p>" +
      reportTitle(title) + contents + "<footer class='page-foot'><span>Healthcare Linguistic Alpha · Annual Test Results</span><span>" +
      String(number).padStart(2, "0") + "</span></footer></section>");
  };
  page("Linguistic Alpha Evaluation", "<p class='subtitle'>Healthcare Earnings Calls · Unified Research Report</p>" +
    "<p>" + esc(descr.overview) + "</p><p class='statement'>" + esc(descr.pipeline) + "</p>" +
    heading("Best Rank IC by annual matched setting") +
    table(["Test year", "A", "B", "B*", "C", "C*"], summary.wins) +
    caption("Eight settings per year. Ties count for each tied model. Exact benchmark, schedule and target results follow.") +
    heading("Frequency of positive linguistic increments") +
    table(["Test year", "B−A", "B*−A", "C−A", "C*−A"], summary.inc) +
    caption("Two benchmarks, two horizons and two schedules each year. Test 2026 is partial through the June 30 market data cutoff."), "cover");
  page("Experimental Design", "<p>" + esc(descr.timing) + "</p>" +
    table(["Test year", "Schedule", "Training years", "Refit", "Calls", "Issuers"], summary.time) +
    heading("Timing and benchmark construction") + "<p>" + esc(descr.limitation) + "</p>" +
    "<p>Market CAR uses the S&P 500 price return. Sector CAR uses the validated Healthcare basket. Source-specific CRSP return handling, temporal availability, and the 16:00 ET Day 0 convention follow the window and horizon research.</p>" +
    heading("Matching and missingness") +
    "<p>Five model scores share the original eligible events and realized CAR within each annual setting. Training-fitted medians fill missing language values. Full FinBERT coverage and fully observed sensitivity tables are also available. Short and long models use their corresponding targets on the intersection for direct horizon comparisons.</p>");
  page("Factor Universe and Coverage", coverageTable() +
    caption("Counts represent the full research transcript universe, including calls outside the label-eligible partial 2026 test.") +
    heading("Language provenance") + "<p>" + esc("The ProsusAI/finbert checkpoint is pinned at " + data.model_revision +
      ". The archive contains " + data.source_sentence_rows.toLocaleString("en-US") +
      " sentence occurrences. The model uses the top label and confidence in the original IT aggregation.") + "</p>" +
    "<p>Prepared remarks and management answers come from separately tagged Capital IQ components. Analyst questions are excluded. LM8 drops the exact duplicate hedging_answers. The feature dictionary follows.</p>");
  page("Financial and Structural Controls", featureDictionary(control));
  page("Linguistic Feature Dictionary", heading("Loughran–McDonald") + featureDictionary(lm) +
    heading("ProsusAI FinBERT") + featureDictionary(data.definitions));
  for (const representation of representations) for (const year of years) {
    const all = by("factor_ic", {year, representation});
    const namesHere = [...new Set(all.map((row) => row.factor))];
    const f = (feature, benchmark, horizon) => all.find((row) => row.factor === feature &&
      row.benchmark === benchmark && row.horizon === horizon);
    const anchor = by("factor_inference", {year, representation, benchmark: "Healthcare", horizon: 20, method: "Issuer cluster"})
      .sort((a, b) => Math.abs(b.point) - Math.abs(a.point)).slice(0, 6);
    page("Single-Factor Rank IC · " + representation,
      caption(displayYear(year) + " · Observed language values only") +
      table(["Factor", "S&P 500 +5", "S&P 500 +20", "Healthcare +5", "Healthcare +20", "Healthcare +20 n"],
        namesHere.map((feature) => [brief(feature), ...benchmarks.flatMap((benchmark) => horizons.map((horizon) =>
          num(f(feature, benchmark, horizon)?.rank_ic))), f(feature, "Healthcare", 20)?.events])) +
      heading("Healthcare +20 · issuer-cluster 95% intervals") + intervalPlot(anchor) +
      caption("The plotted examples have the largest absolute point estimates in this panel. All estimates and their uncertainty are available in the factor appendix tables."));
  }
  page("Factor IC Stability", caption("Annual outer tests on Healthcare CAR[+2,+20]. Full quarterly histories are in the companion tables.") +
    heading("LM answer tone and tone change") + stabilityChart(by("factor_ic", {benchmark: "Healthcare", horizon: 20})
      .filter((row) => ["tone_answers", "tone_answers_chg"].includes(row.factor))) +
    heading("FinBERT prepared and answer sentiment") + stabilityChart(by("factor_ic", {benchmark: "Healthcare", horizon: 20})
      .filter((row) => ["fb_sent_prepared", "fb_sent_answers"].includes(row.factor))) +
    caption("2026 includes the fully observable first part of the year. Development and outer test estimates are reported separately."));
  const example = {year: 2025, benchmark: "Healthcare", horizon: 20};
  page("Raw Factor Quintiles", caption("Test 2025 · Healthcare CAR[+2,+20] · prespecified illustration by answer-tone representation.") +
    quintiles(by("factor_quintiles", example).filter((row) => ["tone_answers", "fb_sent_answers"].includes(row.factor)), "factor") +
    factorTable(by("factor_ic", example).filter((row) => ["tone_answers", "fb_sent_answers"].includes(row.factor))) +
    caption("The Q5−Q1 spread is a difference between mean realized event CARs. The organized factor_quintiles.csv appendix includes all eligible annual and development sorts."));
  for (const block of matrixNames) {
    page("Correlation and Redundancy", heading(block) +
      caption("Development 2010–2023 · Healthcare CAR[+2,+20] eligible events · Pairwise observed Spearman correlations.") +
      heatmap(by("correlations", {year: 0, benchmark: "Healthcare", horizon: 20, block})) +
      caption("Numbers displayed to two decimals. The companion matrix table preserves pairwise counts and full precision."));
  }
  const focus = {year: 2025, benchmark: "Healthcare", horizon: 20, window: "Expanding"};
  page("Model Construction and Selection", modelDefinitions() + caption("A/B/B* specifications and predictions are reused from the completed window and horizon analysis.") +
    "<p class='statement'>Training-fold transforms → candidate feature blocks → three-fold mean Rank IC → refit at annual cutoff → held-out predictions</p>" +
    selection([...by("baseline_selected_features", focus), ...by("finbert_selected_features", focus)]) +
    heading("Model C forward choices in this setting") +
    pathTable(by("finbert_forward_paths", focus).filter((row) => row.accepted)) +
    caption("The complete logs include all accepted and rejected proposals, the same four estimators used by A/B/B*, and the corresponding fold scores."));
  for (const benchmark of benchmarks) for (const year of years) {
    const rows = by("annual_model_performance", {benchmark, year, sample: common});
    const groups = horizons.flatMap((horizon) => annualGroups(rows.filter((row) => row.horizon === horizon), "window")
      .map((group) => ({...group, label: "+" + horizon + " · " + (group.label === "Expanding" ? "Expand" : "Roll")})));
    page(benchmark + "-adjusted Model Results", caption(displayYear(year) + " · Five-model common cohort") +
      bars(groups, models) + modelTable(rows, "setting") +
      caption("Outlined Rank IC marks the highest value in the same year, benchmark, horizon and training schedule. Ties are retained."));
  }
  for (const benchmark of benchmarks) for (const year of years) {
    const rows = by("annual_increments", {benchmark, year, sample: common});
    const grouping = horizons.flatMap((horizon) => schedules.map((window) => ({
      label: "+" + horizon + " · " + (window === "Expanding" ? "Expand" : "Roll"),
      values: Object.fromEntries(rows.filter((row) => row.horizon === horizon && row.window === window)
        .map((row) => [row.contrast, row.delta_rank_ic])),
    })));
    const contrasts = ["B−A", "B*−A", "C−A", "C*−A", "C−B"];
    const item = (horizon, window, contrast) => rows.find((row) => row.horizon === horizon &&
      row.window === window && row.contrast === contrast);
    page("Incremental Linguistic Alpha", caption(displayYear(year) + " · " + benchmark + " adjusted") +
      bars(grouping, contrasts.slice(0, 4), signed, "Incremental Spearman Rank IC") +
      table(["Setting", ...contrasts, "Events"], horizons.flatMap((horizon) => schedules.map((window) =>
        [car(horizon) + " · " + window, ...contrasts.map((contrast) => {
          const val = item(horizon, window, contrast).delta_rank_ic;
          return cell(signed(val), val > 0 ? "positive" : val < 0 ? "negative" : "");
        }), item(horizon, window, contrasts[0]).events]))) +
      caption("C−B compares complete procedures, including independently selected estimators. The four increments in the chart share the financial baseline A."));
  }
  for (const benchmark of benchmarks) {
    const short = by("annual_model_performance", {year: 2025, benchmark, horizon: 5, sample: common});
    const matched = by("annual_model_performance", {year: 2025, benchmark, window: "Expanding", sample: intersection});
    page("Training Window Comparison", caption(benchmark + " adjusted · Test 2025 · CAR[+2,+5]") +
      bars(annualGroups(short, "window"), models) +
      table(["Year", "Target", "Model", "Expanding", "Rolling 5Y", "Events"],
        by("window_comparison", {benchmark}).map((row) => [row.year, "+" + row.horizon, row.model,
          num(row.expanding_rank_ic), num(row.rolling_rank_ic), row.events])));
    page("Prediction Horizon Comparison", caption(benchmark + " adjusted · Test 2025 · Expanding · Matched events") +
      bars(annualGroups(matched, "horizon"), models) +
      table(["Year", "Schedule", "Model", "+2 to +5", "+2 to +20", "Events"],
        by("horizon_comparison", {benchmark}).map((row) => [row.year, row.window, row.model,
          num(row.short_rank_ic), num(row.long_rank_ic), row.events])));
  }
  for (const benchmark of benchmarks) {
    const cohort = {year: 2025, benchmark, horizon: 20, window: "Expanding", sample: common};
    const rows = by("model_quintiles", cohort);
    const groups = [1, 2, 3, 4, 5].map((n) => ({label: "Q" + n, values: Object.fromEntries(rows.filter((row) =>
      row.quintile === n).map((row) => [row.model, row.mean_car]))}));
    page("Model Prediction Quintiles", caption("Test 2025 · " + benchmark + " · CAR[+2,+20] · Expanding") +
      bars(groups, models, pct, "Realized mean CAR by prediction quintile") + quintiles(rows, "model") +
      caption("This event-level return sort is distinct from the raw factor quintiles. All year and model sorts are available in the appendix table."));
  }
  for (const year of years) {
    const rows = by("paired_inference", {...focus, year, family: "Language"});
    page("Statistical Inference", caption(displayYear(year) + " · Healthcare +20 · Expanding · paired increments") +
      intervalPlot(rows.filter((row) => row.method === "Issuer cluster")) + intervalTable(rows) +
      "<p>Pointwise 95% percentile intervals use 999 resamples of shared event indices. The issuer method clusters repeated calls. Overlapping two-month blocks address calendar clustering where independent periods are sufficient. Neither interval fully captures both dependence structures simultaneously.</p>" +
      caption("The experiments are retrospective. Year-specific estimates and intervals remain separate. A broad comparative grid limits confirmatory interpretation."));
  }
  page("Research Insights", insights());
  for (const year of years) for (const benchmark of benchmarks) {
    const rows = [...by("baseline_selected_features", {year, benchmark}), ...by("finbert_selected_features", {year, benchmark})]
      .sort((a, b) => a.horizon - b.horizon || schedules.indexOf(a.window) - schedules.indexOf(b.window) ||
        models.indexOf(a.model) - models.indexOf(b.model));
    page("Appendix · Fitted Specifications", caption(displayYear(year) + " · " + benchmark) +
      table(["Target", "Window", "Model", "Language inputs", "Estimator", "CV Rank IC"],
        rows.map((row) => ["+" + row.horizon, row.window, row.model, cell(featureList(row.features), "long"),
          row.estimator, num(row.inner_rank_ic)])), "appendix");
  }
  for (const benchmark of benchmarks) for (const horizon of horizons) {
    page("Appendix · Development Factor IC", caption(benchmark + " · " + car(horizon)) +
      factorTable(by("factor_ic", {year: 0, benchmark, horizon})), "appendix");
  }
  page("Appendix · Reproducibility", appendices() +
    heading("Model and transcript provenance") +
    "<p>Checkpoint: ProsusAI/finbert. The case-level feature and sentence Parquet archives remain in the ignored repository data directory. The sample, definitions, code reference, runtime and input digests are recorded in the local manifest. The report uses saved aggregate results.</p>" +
    heading("Complete appendix tables") +
    "<p>The companion dashboard and CSVs provide all candidate paths, quarterly factor histories, exact correlation matrices, annual factor and model quintiles, original A/B/B* cohorts and supported intervals. Each table identifies the benchmark, target and test cohort.</p>",
  "appendix");
  research.innerHTML = pages.join("");
}

if (document.body.classList.contains("print-report")) printReport();
else renderDashboard();
