"""Render validated, aggregate-only unified Healthcare research outputs."""

import argparse
import json
from pathlib import Path
import re
import subprocess
from tempfile import TemporaryDirectory

import pandas as pd

HERE = Path(__file__).resolve().parent
BASE = HERE.parent / "healthcare_window_horizon_comparison"
PDF = "Healthcare_Linguistic_Alpha_Unified_Research_Report.pdf"
EXPECTED_PAGES = 52
PDF_TABLE_BASE = (
    "https://github.com/lflsphere/linguistic-alpha/blob/"
    "research/healthcare-window-horizon-comparison/outputs/healthcare_unified_research/"
)
TABLES = (
    "annual_model_performance", "annual_increments", "window_comparison", "horizon_comparison",
    "paired_inference", "model_quintiles", "factor_ic", "factor_stability", "factor_quintiles",
    "factor_inference", "correlations", "feature_missingness", "finbert_native_coverage",
    "finbert_selected_features", "finbert_forward_paths", "finbert_estimator_cv",
    "finbert_feature_importance", "original_ab_performance",
)
BASE_TABLES = ("training_cohorts", "cv_folds", "selected_features", "feature_selection_paths",
               "estimator_cv", "feature_importance")


def records(frame):
    return json.loads(frame.to_json(orient="records", double_precision=15))


def read(directory, base):
    return pd.read_csv(directory / (base + ".csv"), float_precision="round_trip")


def assemble(result):
    if not (result / "aggregate_validation.json").exists():
        raise ValueError("The independent C/C* aggregate validation record is required.")
    manifest = json.loads((result / "finbert_source_snapshot.json").read_text(encoding="utf-8"))
    integrity = json.loads((result / "integrity.json").read_text(encoding="utf-8"))
    if integrity["new_model_fits"] != {"C": 24, "C*": 24}:
        raise ValueError("The C/C* result grid is incomplete.")
    t = {name: records(read(result, name)) for name in TABLES}
    t.update({"baseline_" + name: records(read(BASE, name)) for name in BASE_TABLES})
    t["source_coverage"] = manifest["coverage"]
    t["definitions"] = manifest["definitions"]
    t["model_revision"] = manifest["revision"]
    t["source_sentence_rows"] = manifest["sentence_rows"]
    t["runtime_validation"] = manifest["runtime_validation"]
    return t


def html(data, printable=False):
    markup = (HERE / "unified_dashboard.html.in").read_text(encoding="utf-8")
    if printable:
        markup = markup.replace("</head>", f'<base href="{PDF_TABLE_BASE}">\n</head>', 1)
    css = (HERE / "unified_report.css").read_text(encoding="utf-8")
    script = (HERE / "unified_report.js").read_text(encoding="utf-8")
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    payload = payload.replace("<", "\\u003c")
    return (markup.replace("{{CSS}}", css).replace("{{JS}}", script).replace("{{DATA}}", payload)
            .replace("{{CLASS}}", "print-report" if printable else "dashboard"))


def chrome_path(path):
    candidates = (path, Path("C:/Program Files/Google/Chrome/Application/chrome.exe"),
                  Path("C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe"))
    for item in candidates:
        if item and item.is_file():
            return item
    raise FileNotFoundError("Supply the path to a local Chromium browser using --chrome.")


def print_pdf(data, output, chromium, work):
    printable = work / "print.html"
    printable.write_text(html(data, printable=True), encoding="utf-8")
    args = [str(chromium), "--headless", "--no-sandbox", "--disable-gpu", "--disable-extensions",
            "--disable-background-networking", "--disable-component-update", "--no-first-run",
            "--no-pdf-header-footer", f"--user-data-dir={work / 'browser'}",
            f"--print-to-pdf={output}", printable.as_uri()]
    result = subprocess.run(args, timeout=120, capture_output=True,
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if result.returncode:
        raise RuntimeError(f"Local PDF rendering failed ({result.returncode}): "
                           f"{result.stderr.decode('utf-8', errors='replace')[-800:]}")
    if not output.is_file() or output.stat().st_size < 50000:
        raise ValueError("No substantial PDF output.")
    content = output.read_bytes()
    page_count = len(re.findall(rb"/Type\s*/Page\b", content))
    if page_count != EXPECTED_PAGES:
        raise ValueError(f"Produced {page_count} PDF pages, expected {EXPECTED_PAGES}. Inspect pagination.")
    # Keep fixed-width timestamps so Chromium's xref offsets stay valid.
    content = re.sub(rb"(/(?:CreationDate|ModDate) \(D:)\d{14}", rb"\g<1>20000101000000", content)
    output.write_bytes(content)


def chinese_summary(results, data):
    performance = pd.DataFrame(data["annual_model_performance"])
    full = performance.loc[performance["sample"].eq("Five-model common (train-imputed)")]
    increments = pd.DataFrame(data["annual_increments"])
    increments = increments.loc[increments["sample"].eq("Five-model common (train-imputed)")]
    counts = pd.DataFrame(data["finbert_native_coverage"])
    p = pd.DataFrame(data["paired_inference"])
    q = p.loc[p.family.eq("Language") & p.method.eq("Issuer cluster") & p.available
              & p.contrast.isin(["C−A", "C*−A"])]
    excluding = int((q.lower.gt(0) | q.upper.lt(0)).sum())
    headings = ["测试年", "本年测试场次（各设定范围）", "B−A 为正", "B*−A 为正",
                "C−A 为正", "C*−A 为正", "Rolling 领先", "长期限领先"]
    lines = ["# Healthcare 语言信号统一研究", "",
             "五个模型共享相同年度事件、各目标的已实现 CAR 及 A 的金融控制变量。"
             "A/B/B* 沿用已验证的历史设定与预测。年度结果分别报告。", "",
             "| " + " | ".join(headings) + " |", "|" + "|".join(["---"] * len(headings)) + "|"]
    for year in [2024, 2025, 2026]:
        current = full.loc[full.year.eq(year)]
        limits = f"{current.events.min()}–{current.events.max()}"
        positives = []
        for contrast in ["B−A", "B*−A", "C−A", "C*−A"]:
            x = increments.loc[increments.year.eq(year) & increments.contrast.eq(contrast)]
            positives.append(f"{x.delta_rank_ic.gt(0).sum()}/{len(x)}")
        extras = []
        for name, field in [("window_comparison", "rolling_minus_expanding"),
                            ("horizon_comparison", "long_minus_short")]:
            d = pd.DataFrame(data[name])
            z = d.loc[d.year.eq(year), field]
            extras.append(f"{z.gt(0).sum()}/{len(z)}")
        label = f"{year}（部分年度）" if year == 2026 else str(year)
        lines.append("| " + " | ".join([label, limits, *positives, *extras]) + " |")
    empty = sum(row["feature_count"] == 0 and row["model"] == "C" for row in data["finbert_selected_features"])
    lines.extend(["", f"FinBERT 的发行人聚类区间中，{excluding}/{len(q)} 个可用 95% 区间排除零。"
                       f"Model C 在 {empty}/24 个设定中保留了与 A 相同的特征组。2026 部分样本的两个月"
                       "时间块不足以同时评估时间依赖。多项比较的区间需按探索性证据解读。", "",
                  "各测试设定有 FinBERT 文本的场次范围为 "
                  f"{counts.transcript_covered.min()}–{counts.transcript_covered.max()}，"
                  "13 个特征均可观测的范围为 "
                  f"{counts.features_complete.min()}–{counts.features_complete.max()}。"
                  "原始因子季度稳定性、五分位收益和逐设定置信区间均保存在配套表格中。", "",
                  f"[英文研究报告]({PDF}) · [仪表板](dashboard.html) · "
                  "[年度模型结果](annual_model_performance.csv) · [统计推断](paired_inference.csv)", ""])
    (results / "summary_zh.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=HERE)
    parser.add_argument("--chrome", type=Path)
    args = parser.parse_args()
    results = args.results.resolve()
    data = assemble(results)
    with TemporaryDirectory(prefix="healthcare-unified-render-") as temp:
        scratch = Path(temp)
        target = scratch / PDF
        print_pdf(data, target, chrome_path(args.chrome), scratch)
        pdf = results / PDF
        pdf.write_bytes(target.read_bytes())
        (results / "dashboard.html").write_text(html(data), encoding="utf-8")
        chinese_summary(results, data)
    print(f"Wrote {pdf.name} and dashboard.html from validated saved tables.")


if __name__ == "__main__":
    main()
