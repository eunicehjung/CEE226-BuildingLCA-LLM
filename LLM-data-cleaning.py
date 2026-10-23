"""
Post-process LLM ECI estimation runs from gemini_lca_project_runs.py.

Reads:
  GROUND_TRUTH_PATH                        (one row per project with the true ECI)
  <run folder>/project-#_runs.csv          (one file per project, N rows of runs)

Writes (into the run folder):
  YYYYMMDD_HHMM_results_<lastproject>_<ntests>.csv

Layout of the output file (same as 20260825_1248_results_project-11_10.csv):
  - one row per project: ground truth, mean of the per-run midpoints, SD, error, ratio
  - MEAN, MEDIAN, MIN, MAX, RANGE, SD (across projects) and RMSE summary rows

Usage:
  python LLM-data_cleaning.py                                   # newest folder in results/
  python LLM-data_cleaning.py 20261019_14_gemini-3_8-flash      # a folder name in results/
  python LLM-data_cleaning.py /full/path/to/run/folder
"""

import os
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------------------
# Settings - edit here
# --------------------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_ROOT = SCRIPT_DIR / "results"          # same RESULTS_ROOT as gemini_lca_project_runs.py

# Ground truth for all projects. CSV or Excel. If the file is not found here, the script
# also looks for ground_truth.csv inside the run folder (the old layout).
GROUND_TRUTH_PATH = SCRIPT_DIR / "results" / "ground_truth.csv"

# Column names in the ground-truth file. The ID column may hold "project-4" or just 4.
TRUTH_ID_COL = "Project Name"                  # e.g. "project_index" if it holds numbers
TRUTH_COL = "ECI_true_kgCO2e_m2"

# Only report projects that have run data (True), or every project in the ground truth (False).
ONLY_PROJECTS_WITH_RUNS = True

NAME_COL = "Project Name"
EST_COL = "LLM_mean_mid_kgCO2e_m2"


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------
def project_sort_key(name):
    """Sort project-2 before project-10 (numeric, not lexicographic)."""
    m = re.search(r"(\d+)", str(name))
    return (int(m.group(1)) if m else 10**9, str(name))


def to_project_name(value):
    """4, 4.0, '4', 'project-4', 'Project 4' -> 'project-4' (matches the runs files)."""
    m = re.search(r"(\d+)", str(value))
    return f"project-{int(m.group(1))}" if m else str(value).strip()


def resolve_run_folder(arg):
    """Full path, a folder name inside RESULTS_ROOT, or (no arg) the newest folder with runs."""
    if arg:
        p = Path(arg)
        if not p.is_absolute() and not p.exists():
            p = RESULTS_ROOT / arg
        if not p.is_dir():
            raise FileNotFoundError(f"Run folder not found: {p}")
        return p
    candidates = [d for d in RESULTS_ROOT.iterdir()
                  if d.is_dir() and any(d.glob("project-*_runs.csv"))] if RESULTS_ROOT.is_dir() else []
    if not candidates:
        raise FileNotFoundError(f"No run folders with project-*_runs.csv in {RESULTS_ROOT}")
    newest = max(candidates, key=lambda d: d.stat().st_mtime)
    print(f"Using newest run folder: {newest.name}")
    return newest


def load_ground_truth(run_folder):
    path = Path(GROUND_TRUTH_PATH)
    if not path.exists():
        path = run_folder / "ground_truth.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Ground truth not found at {GROUND_TRUTH_PATH} or {run_folder / 'ground_truth.csv'}. "
            "Set GROUND_TRUTH_PATH at the top of this script.")
    gt = pd.read_excel(path) if path.suffix.lower() in (".xlsx", ".xls") else pd.read_csv(path)
    missing_cols = [c for c in (TRUTH_ID_COL, TRUTH_COL) if c not in gt.columns]
    if missing_cols:
        raise ValueError(f"{path.name} is missing {missing_cols}. Found: {list(gt.columns)}. "
                         "Adjust TRUTH_ID_COL / TRUTH_COL at the top of this script.")
    gt = gt[[TRUTH_ID_COL, TRUTH_COL]].copy()
    gt[NAME_COL] = gt[TRUTH_ID_COL].map(to_project_name)
    gt[TRUTH_COL] = pd.to_numeric(gt[TRUTH_COL], errors="coerce")
    print(f"Ground truth: {path} ({len(gt)} projects)")
    return gt[[NAME_COL, TRUTH_COL]]


def load_runs(run_folder):
    """Return {project_name: DataFrame of runs} from project-*_runs.csv files."""
    runs = {}
    for path in sorted(run_folder.glob("project-*_runs.csv")):
        try:
            df = pd.read_csv(path)
        except Exception as e:
            print(f"  ! could not read {path.name}: {e}")
            continue
        if "project" in df.columns and df["project"].notna().any():
            name = str(df["project"].dropna().iloc[0])
        else:
            name = path.name[:-len("_runs.csv")]
        runs[to_project_name(name)] = df
    return runs


def summarize_project(df):
    """Mean and SD of per-run midpoints, using only runs with parse_ok == True."""
    if "parse_ok" in df.columns:
        ok = df[df["parse_ok"].astype(str).str.lower().isin(["true", "1"])]
    else:
        ok = df
    if "mid" in ok.columns:
        mids = pd.to_numeric(ok["mid"], errors="coerce")
    else:
        mids = (pd.to_numeric(ok.get("low"), errors="coerce")
                + pd.to_numeric(ok.get("high"), errors="coerce")) / 2
    mids = mids.dropna()
    return {
        "mean_mid": mids.mean() if len(mids) else np.nan,
        "std_mid": mids.std(ddof=1) if len(mids) > 1 else np.nan,
        "n_runs": len(df),
        "n_parsed": len(mids),
    }


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
def main(arg=None):
    run_folder = resolve_run_folder(arg)
    out = load_ground_truth(run_folder)

    runs = load_runs(run_folder)
    if not runs:
        raise FileNotFoundError(f"No project-*_runs.csv files in {run_folder}")
    stats = {name: summarize_project(df) for name, df in runs.items()}

    no_truth = sorted(set(stats) - set(out[NAME_COL]), key=project_sort_key)
    if no_truth:
        print(f"  ! runs without ground truth (excluded): {no_truth}")
    if ONLY_PROJECTS_WITH_RUNS:
        out = out[out[NAME_COL].isin(stats)]
    out = out.copy()

    out[EST_COL] = out[NAME_COL].map(lambda p: stats.get(p, {}).get("mean_mid", np.nan))
    out["LLM_sd_mid"] = out[NAME_COL].map(lambda p: stats.get(p, {}).get("std_mid", np.nan))

    # Report incomplete projects (failed or missing runs)
    incomplete = {p: (s["n_parsed"], s["n_runs"]) for p, s in stats.items()
                  if p in set(out[NAME_COL]) and s["n_parsed"] < s["n_runs"]}
    if incomplete:
        print("  ! projects with failed runs (parsed/total): "
              + ", ".join(f"{p} {a}/{b}" for p, (a, b) in
                          sorted(incomplete.items(), key=lambda kv: project_sort_key(kv[0]))))
    missing = out[out[EST_COL].isna()][NAME_COL].tolist()
    if missing:
        print(f"  ! no usable run data for: {missing}")

    # Most common parsed-run count, used for the filename
    counts = [stats[p]["n_parsed"] for p in out[NAME_COL] if p in stats]
    n_tests = int(pd.Series(counts).mode().max()) if counts else 0

    # ---- Per-project error columns ----
    out["abs_error"] = (out[EST_COL] - out[TRUTH_COL]).abs()
    out["ratio_est_over_true"] = out[EST_COL] / out[TRUTH_COL]
    out = out.sort_values(NAME_COL, key=lambda s: s.map(project_sort_key)).reset_index(drop=True)

    # ---- Summary rows ----
    truth_v = out[TRUTH_COL].astype(float)
    est_v = out[EST_COL].astype(float)
    paired = out[[TRUTH_COL, EST_COL]].dropna()
    rmse = float(np.sqrt(((paired[EST_COL] - paired[TRUTH_COL]) ** 2).mean())) if len(paired) else np.nan

    blank = {c: "" for c in out.columns}

    def row(label, **vals):
        r = dict(blank)
        r[NAME_COL] = label
        r.update(vals)
        return r

    ae, ratio = out["abs_error"], out["ratio_est_over_true"]
    summary = pd.DataFrame([
        dict(blank),
        row("MEAN (all projects)", **{TRUTH_COL: truth_v.mean(), EST_COL: est_v.mean(),
            "LLM_sd_mid": out["LLM_sd_mid"].mean(), "abs_error": ae.mean(),
            "ratio_est_over_true": ratio.mean()}),
        row("MEDIAN (all projects)", **{TRUTH_COL: truth_v.median(), EST_COL: est_v.median(),
            "LLM_sd_mid": out["LLM_sd_mid"].median(), "abs_error": ae.median(),
            "ratio_est_over_true": ratio.median()}),
        row("MIN (all projects)", **{TRUTH_COL: truth_v.min(), EST_COL: est_v.min(),
            "abs_error": ae.min(), "ratio_est_over_true": ratio.min()}),
        row("MAX (all projects)", **{TRUTH_COL: truth_v.max(), EST_COL: est_v.max(),
            "abs_error": ae.max(), "ratio_est_over_true": ratio.max()}),
        row("RANGE (max - min)", **{TRUTH_COL: truth_v.max() - truth_v.min(),
            EST_COL: est_v.max() - est_v.min(), "abs_error": ae.max() - ae.min(),
            "ratio_est_over_true": ratio.max() - ratio.min()}),
        # SD across projects: if the estimate SD is much smaller than the truth SD,
        # the model is not discriminating between buildings.
        row("SD (across projects)", **{TRUTH_COL: truth_v.std(ddof=1), EST_COL: est_v.std(ddof=1)}),
        row("RMSE (truth vs estimate)", **{EST_COL: rmse, "abs_error": rmse}),
    ])
    final = pd.concat([out, summary], ignore_index=True)

    for c in [TRUTH_COL, EST_COL, "LLM_sd_mid", "abs_error", "ratio_est_over_true"]:
        final[c] = pd.to_numeric(final[c], errors="coerce").round(2)

    # ---- Filename: date_time_results_lastproject_ntests.csv ----
    last_project = re.sub(r"[^A-Za-z0-9_-]+", "-", str(out[NAME_COL].iloc[-1]))
    out_path = run_folder / f"{datetime.now():%Y%m%d_%H%M}_results_{last_project}_{n_tests}.csv"
    final.to_csv(out_path, index=False)

    # ---- Console report ----
    print(f"\nProjects: {len(out)}   tests per project: {n_tests}")
    if len(out) <= 40:
        print(final.to_string(index=False))
    print(f"\nRMSE                 : {rmse:.2f} kgCO2e/m2")
    print(f"Mean absolute error  : {ae.mean():.2f} kgCO2e/m2")
    print(f"Mean ratio est/true  : {ratio.mean():.2f}")
    print(f"SD within project    : {out['LLM_sd_mid'].mean():.2f} (mean of per-project SDs)")
    print(f"SD across projects   : truth {truth_v.std(ddof=1):.2f} | estimate {est_v.std(ddof=1):.2f}")
    print(f"Range across projects: truth {truth_v.min():.2f}-{truth_v.max():.2f} "
          f"(span {truth_v.max() - truth_v.min():.2f}) | estimate {est_v.min():.2f}-{est_v.max():.2f} "
          f"(span {est_v.max() - est_v.min():.2f})")
    print(f"\nSaved: {out_path}")
    return out_path


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)