"""
1) Build <yyyymmdd>_eci-llm-values.csv (project_id, truth, f_llm) from a results folder.
2) Draw fig1_bands_sorted.png: +/-DELTA band around each estimate vs. the true ECI,
   sorted by true ECI. Same figure as 20260825_eci-llm-eval.py.

Reads:   results/<run-name>/project-#_runs.csv  and  ground_truth.csv
Writes:  results/<run-name>/<yyyymmdd>_eci-llm-values.csv
         results/<run-name>/figures/fig1_bands_sorted.png
         results/<run-name>/figures/fig1_bands_summary.txt   (summary lines also printed)

  f_llm = mean of the parsed run midpoints for that building (failed runs skipped)
  band  = f_llm * (1 +/- DELTA); a building is "captured" if its true ECI is inside it

Usage (from CEE226-reference):
  python make_fig1_bands.py 20261022_15_no-background
"""

import re
import sys
import textwrap
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
from scipy import stats

# --------------------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------------------
DELTA = 0.20             # band half-width: 0.20 = +/-20% around the estimate
ALPHA = 0.05             # for the Wilson 95% CI
LABEL_MAX_N = 40         # up to this many buildings: one labelled row each (original look)
                         # more: compact plot, no per-building labels
LOG_X = "auto"           # "auto" = log axis when the largest ECI is >10x the smallest; True/False

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_ROOT = SCRIPT_DIR / "results"
GROUND_TRUTH = SCRIPT_DIR / "results" / "ground_truth.csv"

UNITS = r"kg CO$_2$e/m$^2$"
GREEN, RED, INK = "#2e8b6f", "#c0504d", "#333333"


# --------------------------------------------------------------------------------------
# 1) Values file
# --------------------------------------------------------------------------------------
def build_values(run_dir: Path) -> pd.DataFrame:
    files = sorted(run_dir.glob("project-*_runs.csv"),
                   key=lambda p: int(re.search(r"project-(\d+)_", p.name).group(1)))
    if not files:
        raise SystemExit(f"No project-*_runs.csv files in {run_dir}")
    truth = pd.read_csv(GROUND_TRUTH).set_index("Project Name")["ECI_true_kgCO2e_m2"]

    rows, no_runs, no_truth = [], [], []
    for f in files:
        name = f.name[: -len("_runs.csv")]                       # e.g. project-41
        runs = pd.read_csv(f)
        ok = runs[runs["parse_ok"].astype(str) == "True"]
        mids = pd.to_numeric(ok["mid"], errors="coerce").dropna()
        if mids.empty:
            no_runs.append(name)
        elif name not in truth.index:
            no_truth.append(name)
        else:
            rows.append({"project_id": "p" + name.split("-")[1],
                         "truth": float(truth[name]), "f_llm": float(mids.mean())})
    if no_runs:
        print(f"! left out, no usable runs: {no_runs}")
    if no_truth:
        print(f"! left out, not in ground_truth.csv: {no_truth}")
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# 2) Figure
# --------------------------------------------------------------------------------------
def coverage(truth, pred):
    lo, hi = pred * (1 - DELTA), pred * (1 + DELTA)
    hit = (truth >= lo) & (truth <= hi)
    k, n = int(hit.sum()), truth.size
    ci = stats.binomtest(k, n).proportion_ci(1 - ALPHA, method="wilson")
    return lo, hi, hit, k, n, (ci.low, ci.high)


def fig_bands(df: pd.DataFrame, out_png: Path):
    truth, pred = df["truth"].to_numpy(float), df["f_llm"].to_numpy(float)
    lo, hi, hit, k, n, ci = coverage(truth, pred)
    gap = (hi - truth) / truth                 # + = headroom above truth, - = truth above band
    order = np.argsort(truth)
    labelled = n <= LABEL_MAX_N
    log_x = (max(hi.max(), truth.max()) / min(lo.min(), truth.min()) > 10) if LOG_X == "auto" else LOG_X

    height = 5.2 if labelled else 9.0
    fig, ax = plt.subplots(figsize=(8.6, height))
    bar_h, tick_h, dot = (0.55, 0.28, 6) if labelled else (0.8, 0.45, 2.2)

    for row, i in enumerate(order):
        col = GREEN if hit[i] else RED
        ax.barh(row, hi[i] - lo[i], left=lo[i], height=bar_h, color=col, alpha=0.18,
                edgecolor=col if labelled else "none", linewidth=1.2)
        ax.plot([pred[i]] * 2, [row - tick_h, row + tick_h], color=col,
                linewidth=2.4 if labelled else 1.0)
        ax.plot(truth[i], row, "o", color=INK, markersize=dot, zorder=5)
        if labelled:
            ax.annotate(f"{gap[i]:+.0%}", (hi[i], row), xytext=(6, 0), textcoords="offset points",
                        ha="left", va="center", fontsize=8.5,
                        fontweight="bold" if gap[i] < 0 else "normal",
                        color=RED if gap[i] < 0 else "#555555", zorder=6)

    xmin = min(lo.min(), truth.min())
    if log_x:
        ax.set_xscale("log")
        x0, x1 = xmin * 0.9, max(hi.max(), truth.max()) * (1.6 if labelled else 1.15)
        ax.set_xlim(x0, x1)
        ticks = [t for t in (10, 20, 50, 100, 200, 500, 1000, 2000, 5000) if x0 <= t <= x1]
        ax.set_xticks(ticks, [f"{t:,}" for t in ticks])
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    else:
        xmax = max(hi.max(), truth.max())
        span = xmax - xmin
        ax.set_xlim(xmin - 0.03 * span, xmax + (0.13 if labelled else 0.03) * span)

    if labelled:
        ax.set_yticks(range(n))
        ax.set_yticklabels(df["project_id"].to_numpy()[order])
    else:
        ax.set_yticks([])
        ax.set_ylabel(f"{n} buildings, sorted by true ECI (low → high)")
    ax.set_ylim(-0.8, n - 0.2)
    ax.set_xlabel(f"ECI ({UNITS})" + (", log scale" if log_x else ""))
    ax.set_title(f"±{DELTA:.0%} band vs. truth, sorted by true ECI", loc="left")
    ax.spines[["top", "right", "left"]].set_visible(False)

    note = (f"{k}/{n} captured — coverage {k / n:.1%}, "
            f"Wilson 95% CI [{ci[0]:.0%}, {ci[1]:.0%}]")
    if labelled:
        legend = ("Labels: distance from band's upper edge to true ECI, as % of truth. "
                  "Positive = headroom; negative = truth above the edge.")
    else:
        below, above = int((truth < lo).sum()), int((truth > hi).sum())
        legend = (f"Dot = true ECI; tick = estimate; band = estimate ±{DELTA:.0%}. Red = truth outside "
                  f"the band ({below} below, {above} above).")
    y1, y2 = (-0.155, -0.225) if labelled else (-0.085, -0.115)
    ax.text(0.0, y1, note, transform=ax.transAxes, fontsize=9, color="#555")
    ax.text(0.0, y2, legend, transform=ax.transAxes, fontsize=8, color="#888")

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_summary(df: pd.DataFrame, run_name: str, out_txt: Path):
    """Per-building inside/outside table saved as .txt; the three totals are printed."""
    truth, pred = df["truth"].to_numpy(float), df["f_llm"].to_numpy(float)
    lo, hi, hit, k, n, ci = coverage(truth, pred)
    result = np.where(hit, "inside", np.where(truth > hi, "ABOVE", "BELOW"))
    table = pd.DataFrame({
        "project": df["project_id"].str.replace("p", "project-", regex=False),
        "truth": truth.round(1), "estimate": pred.round(1),
        "band_low": lo.round(1), "band_high": hi.round(1),
        "est/truth": (pred / truth).round(2), "result": result,
    })
    above = table.loc[table.result == "ABOVE", "project"].tolist()
    below = table.loc[table.result == "BELOW", "project"].tolist()

    lines = [
        f"Run: {run_name}",
        f"Band: estimate ±{DELTA:.0%}.  inside = true ECI within the band; "
        "ABOVE = truth above the band (model underestimates); "
        "BELOW = truth below the band (model overestimates).",
        "",
        f"Inside: {k}/{n} = {k / n:.1%}   Wilson 95% CI [{ci[0]:.1%}, {ci[1]:.1%}]",
        f"ABOVE:  {len(above)}/{n} = {len(above) / n:.1%}",
        f"BELOW:  {len(below)}/{n} = {len(below) / n:.1%}",
        "",
        textwrap.fill(f"ABOVE ({len(above)}): {', '.join(above) or '-'}", 100,
                      subsequent_indent="    "),
        textwrap.fill(f"BELOW ({len(below)}): {', '.join(below) or '-'}", 100,
                      subsequent_indent="    "),
        "",
        "Per building (ECI in kgCO2e/m2):",
        table.to_string(index=False),
    ]
    text = "\n".join(lines) + "\n"
    out_txt.write_text(text, encoding="utf-8")
    print("\n" + "\n".join(lines[3:6]) + "\n")      # Terminal: the three summary lines only


def main(run_name: str):
    run_dir = RESULTS_ROOT / run_name
    df = build_values(run_dir)
    values_path = run_dir / f"{datetime.now():%Y%m%d}_eci-llm-values.csv"
    df.to_csv(values_path, index=False)
    print(f"{len(df)} buildings. Saved: {values_path}")

    fig_path = run_dir / "figures" / "fig1_bands_sorted.png"
    fig_bands(df, fig_path)
    print(f"Saved: {fig_path}")

    txt_path = run_dir / "figures" / "fig1_bands_summary.txt"
    write_summary(df, run_name, txt_path)
    print(f"Saved: {txt_path}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python make_fig1_bands.py <run-name>")
    main(sys.argv[1])
