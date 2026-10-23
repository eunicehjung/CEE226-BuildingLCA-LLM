#!/usr/bin/env python3
"""
Gemini Batch API run: embodied-carbon (A-C) range estimation, one building per request,
N repeated runs per project. Standalone: settings, prompts and parser are all in this file.

Prompts, parser and CSV format were copied from gemini_lca_project_runs.py (text PDF mode), so
the two scripts send identical requests until you edit one of them. Edit the prompts in the
"Prompt" section below. Results go to results/<run-name>/project-#_runs.csv and raw/, the same
as gemini_lca_project_runs.py, so LLM-data-cleaning.py works unchanged.

Batch requests have their own quota and cost half the standard price. Google's target
turnaround is 24 hours (often much faster). The jobs run on Google's side, so the computer can
be off between submit and collect.

Steps
  python gemini_lca_batch.py submit  --run-name result-folder-name     # build, upload, start job(s)
  python gemini_lca_batch.py status  --run-name result-folder-name     # check progress
  python gemini_lca_batch.py collect --run-name result-folder-name     # write results to CSVs
  python gemini_lca_batch.py wait    --run-name result-folder-name     # check every 5 min, collect when done

  submit also takes --projects ("all", "4", "1-20", "4,10,25-30"), --runs (default 10),
  and --dry-run (write the request files only, upload nothing).

Install:  pip install google-genai pandas openpyxl PyPDF2 python-dotenv
Key:      GEMINI_API_KEY in the environment or a .env file
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
import warnings
from datetime import datetime
from pathlib import Path

# Hide pandas' notices about old optional speed-up packages (numexpr, bottleneck).
# They don't affect results. Must come before "import pandas".
warnings.filterwarnings("ignore", message=r"Pandas requires version", category=UserWarning)

import pandas as pd
import PyPDF2
from dotenv import load_dotenv

load_dotenv()

# --------------------------------------------------------------------------------------
# Model settings - edit here
# --------------------------------------------------------------------------------------
GEMINI_MODEL = "gemini-3.8-flash"
TEMPERATURE = 1.0
THINKING_LEVEL = "medium"  # "low" | "medium" | "high" | None (model default)
MAX_OUTPUT_TOKENS = 16384  # includes thinking tokens

# --------------------------------------------------------------------------------------
# Run settings - edit here
# --------------------------------------------------------------------------------------
RETRY_FAILED = False  # True -> submit also resends runs with parse_ok=False
ONTOLOGY_MODE = "benchmarks"  # label written to the CSV column ("none" when the tables are off)
GEOMETRY_MODE = "none"  # label written to the CSV column
MAX_FILE_MB = 1000  # split requests into files below this size (Google's limit is 2 GB)
POLL_MINUTES = 5  # for "wait"

# --------------------------------------------------------------------------------------
# Paths - all relative to the folder this script is in (CEE226-reference)
# --------------------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent

BASE_DIR = SCRIPT_DIR
GLOSSARY_PATH = BASE_DIR / "references" / "data_glossary_LLM.xlsx"
BACKGROUND_PATHS = [BASE_DIR / "references" / f"{i}.pdf" for i in range(1, 5)]  # 1.pdf to 4.pdf
PROCESS_PATH = BASE_DIR / "references" / "250524_CLF-LCA-Practice-Guide.txt"

PROJECT_DIR = SCRIPT_DIR
METADATA_PATH = PROJECT_DIR / "references" / "buildings_metadata_LLM.xlsx"
RESULTS_ROOT = PROJECT_DIR / "results"
BENCHMARK_PATH = BASE_DIR / "references" / "eci_benchmarks_A-C.md"

MAX_PDF_CHARS = 100_000
SYSTEM_PROMPT = "You are a building LCA expert."

CSV_COLUMNS = [
    "project", "run_idx", "low", "high", "mid", "parse_ok", "failure_type",
    "ontology_mode", "geometry_mode", "finish_reason", "output_tokens", "thought_tokens",
    "cached_tokens", "prompt_tokens", "latency_s", "model", "temperature",
    "thinking_setting", "timestamp",
]

THINKING_SETTING = f"level={THINKING_LEVEL}" if THINKING_LEVEL else "default"
DONE_STATES = {"JOB_STATE_SUCCEEDED", "JOB_STATE_PARTIALLY_SUCCEEDED",
               "JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED"}
OK_STATES = {"JOB_STATE_SUCCEEDED", "JOB_STATE_PARTIALLY_SUCCEEDED"}


# --------------------------------------------------------------------------------------
# Readers
# --------------------------------------------------------------------------------------
def read_excel_text(path: Path) -> str:
    try:
        return pd.read_excel(path).to_string(index=False)
    except Exception as e:
        print(f"Warning: could not read {path}: {e}")
        return ""


def read_pdf(path: Path, max_chars: int = MAX_PDF_CHARS) -> str:
    if not path.exists():
        print(f"Warning: PDF not found: {path}")
        return ""
    try:
        with open(path, "rb") as f:
            text = "\n".join((p.extract_text() or "") for p in PyPDF2.PdfReader(f).pages)
        return text[:max_chars] + "...[truncated]" if len(text) > max_chars else text
    except Exception as e:
        print(f"Warning: could not read {path}: {e}")
        return ""


def read_txt(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except Exception as e:
        print(f"Warning: could not read {path}: {e}")
        return ""


# --------------------------------------------------------------------------------------
# Prompt - edit here
#   build_cached_prefix: the shared part, the same for every request
#   build_building_prompt: the part that changes per building
#   To include the background papers, uncomment the two lines in build_cached_prefix
# --------------------------------------------------------------------------------------
def build_cached_prefix(background_text: str, process_text: str, glossary_text: str,
                        benchmark_text: str) -> list[str]:
    """Identical for every request."""
    return [
        "Forget the information from previous sessions and calculate everything from the beginning.",

        f"data_glossary_LLM contains feature names and descriptions for the data:\n\n{glossary_text}",

        "These journal papers contain background information about the early building LCA "
        f"process:\n\n{background_text}",

        # "This document contains a detailed practice guide on life cycle assessment of buildings "
        # f"focused on implementation. Use this guideline for LCA estimation:\n\n{process_text}",

        "The following benchmark tables give published embodied carbon intensity (ECI) "
        f"distributions. Use them as described in the notes below the tables:\n\n{benchmark_text}",

        "Reference the given feature information above. Estimate the embodied carbon value of the "
        "building below:\n\n",
    ]


def build_building_prompt(building_text: str) -> str:
    """Different for every project."""
    return (
        f"{building_text}\n\n"
        "Provide a plausible range in units of kgCO2e/m2 (a range, not a single value). For floor "
        "area, you need to use bldg_cfa. If cfa is not provided, use bldg_renovated_cfa + "
        "bldg_renovated_gfa. Include life cycle stages A-C. Briefly state your method and key "
        "assumptions.\n\n"
        "End your answer with exactly one final line in this format and nothing after it:\n"
        "RANGE_kgCO2e_per_m2: <low> - <high>\n"
        "(plain numbers only, no units or other text on that line)"
    )


def format_building(row: pd.Series) -> str:
    """One 'field: value' line per non-empty column. project_index is the ID, not a feature."""
    lines = []
    for col, val in row.items():
        if col == "project_index" or pd.isna(val):
            continue
        if isinstance(val, float) and val.is_integer():
            val = int(val)
        lines.append(f"{col}: {val}")
    return "\n".join(lines)


# --------------------------------------------------------------------------------------
# Response parsing
# --------------------------------------------------------------------------------------
_RANGE_RE = re.compile(
    r"RANGE_kgCO2e_per_m2\s*:\s*([0-9][0-9,]*\.?[0-9]*)\s*(?:-|\u2013|\u2014|to)\s*([0-9][0-9,]*\.?[0-9]*)",
    re.IGNORECASE,
)


def parse_range(text: str) -> tuple[float | None, float | None, str]:
    """Return (low, high, failure_type). Uses the LAST matching line; failure_type '' on success."""
    clean = text.replace("*", "").replace("`", "")
    matches = _RANGE_RE.findall(clean)
    if not matches:
        return None, None, "no_range"
    try:
        low, high = (float(x.replace(",", "").rstrip(".")) for x in matches[-1])
    except ValueError:
        return None, None, "no_range"
    if low <= 0 or high < low:
        return low, high, "invalid_range"
    return low, high, ""


# --------------------------------------------------------------------------------------
# CSV and project selection (same as gemini_lca_project_runs.py)
# --------------------------------------------------------------------------------------
def write_csv(path: Path, rows: list[dict]):
    rows = sorted(rows, key=lambda r: int(r["run_idx"]))
    with open(path, "w", newline="", encoding="utf-8") as f:  # default \r\n, as in the example
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        w.writeheader()
        w.writerows(rows)


def load_existing(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def parse_project_selection(spec: str, available: list[int]) -> list[int]:
    if spec == "all":
        return available
    chosen = set()
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if "-" in chunk:
            a, b = chunk.split("-")
            chosen.update(range(int(a), int(b) + 1))
        elif chunk:
            chosen.add(int(chunk))
    missing = sorted(chosen - set(available))
    if missing:
        print(f"Warning: project_index not in metadata, skipped: {missing}")
    return sorted(chosen & set(available))


# --------------------------------------------------------------------------------------
# Batch API
# --------------------------------------------------------------------------------------
def client():
    from google import genai
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise SystemExit("Set GEMINI_API_KEY (or GOOGLE_API_KEY) in your environment/.env")
    return genai.Client(api_key=key)


def run_dir(run_name: str) -> Path:
    return RESULTS_ROOT / run_name


def manifest_path(run_name: str) -> Path:
    return run_dir(run_name) / "batch" / "jobs.json"


def load_manifest(run_name: str) -> dict:
    p = manifest_path(run_name)
    return json.loads(p.read_text()) if p.exists() else {"jobs": []}


def save_manifest(run_name: str, m: dict):
    p = manifest_path(run_name)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(m, indent=2))


def build_prefix() -> list[str]:
    glossary_text = read_excel_text(GLOSSARY_PATH)
    process_text = read_txt(PROCESS_PATH)
    background_text = "\n".join(read_pdf(p) for p in BACKGROUND_PATHS)
    benchmark_text = read_txt(BENCHMARK_PATH) if BENCHMARK_PATH.exists() else ""
    prefix = build_cached_prefix(background_text, process_text, glossary_text,
                                 benchmark_text or "<<BENCHMARK FILE MISSING>>")
    # Stop instead of sending tables that are empty (only when the benchmark lines are switched on)
    if any("<<BENCHMARK FILE MISSING>>" in part for part in prefix):
        raise SystemExit(f"Benchmark lines are on, but {BENCHMARK_PATH} is missing or empty.")
    return prefix


def generation_config() -> dict:
    cfg = {"maxOutputTokens": MAX_OUTPUT_TOKENS}
    if TEMPERATURE is not None:
        cfg["temperature"] = TEMPERATURE
    if THINKING_LEVEL:
        from google.genai import types
        lvl = types.ThinkingConfig(thinking_level=THINKING_LEVEL).thinking_level
        cfg["thinkingConfig"] = {"thinkingLevel": getattr(lvl, "value", str(lvl))}
    return cfg


def make_request(prefix: list[str], building_prompt: str, gen_cfg: dict) -> dict:
    # Same shape the SDK sends for contents=[*prefix, building_prompt]: one user turn, many parts
    return {
        "contents": [{"role": "user",
                      "parts": [{"text": p} for p in [*prefix, building_prompt]]}],
        "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "generationConfig": gen_cfg,
    }


# --------------------------------------------------------------------------------------
# submit
# --------------------------------------------------------------------------------------
def runs_to_do(csv_path: Path, n_runs: int) -> list[int]:
    rows = load_existing(csv_path)
    done_ok = {int(r["run_idx"]) for r in rows if str(r["parse_ok"]) == "True"}
    done_any = {int(r["run_idx"]) for r in rows}
    done = done_ok if RETRY_FAILED else done_any
    return [i for i in range(n_runs) if i not in done]


def cmd_submit(args):
    m = load_manifest(args.run_name)
    pending = [j for j in m["jobs"] if not j.get("collected") and j.get("state") not in DONE_STATES - OK_STATES]
    if pending and not args.force:
        raise SystemExit(f"{len(pending)} job(s) in this run are not collected yet. Run 'status' or "
                         "'collect' first (or add --force to submit anyway).")

    out = run_dir(args.run_name)
    bdir = out / "batch"
    bdir.mkdir(parents=True, exist_ok=True)

    meta = pd.read_excel(METADATA_PATH)
    projects = parse_project_selection(args.projects, [int(x) for x in meta["project_index"]])
    meta = meta.set_index("project_index")

    prefix = build_prefix()
    gen_cfg = generation_config()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    files, fh, size, n_req, n_chars = [], None, 0, 0, 0
    for idx in projects:
        todo = runs_to_do(out / f"project-{idx}_runs.csv", args.runs)
        if not todo:
            continue
        req = make_request(prefix, build_building_prompt(format_building(meta.loc[idx])), gen_cfg)
        for i in todo:
            line = json.dumps({"key": f"project-{idx}__run{i}", "request": req}, ensure_ascii=False) + "\n"
            b = len(line.encode("utf-8"))
            if fh is None or size + b > MAX_FILE_MB * 1e6:
                if fh:
                    fh.close()
                files.append(bdir / f"requests_{stamp}_part{len(files) + 1}.jsonl")
                fh, size = open(files[-1], "w", encoding="utf-8"), 0
            fh.write(line)
            size += b
            n_req += 1
            n_chars += sum(len(p["text"]) for p in req["contents"][0]["parts"])
    if fh:
        fh.close()

    if n_req == 0:
        print("Nothing to submit: every selected run is already in the CSVs.")
        return
    print(f"{n_req} requests in {len(files)} file(s); about {n_chars / 4 / 1e6:.0f}M input tokens "
          f"(rough estimate). Model={GEMINI_MODEL}, thinking={THINKING_SETTING}")
    if args.dry_run:
        print(f"Dry run: files written to {bdir}, nothing uploaded.")
        return

    c = client()
    for f in files:
        print(f"Uploading {f.name} ({f.stat().st_size / 1e6:.0f} MB) ...")
        up = c.files.upload(file=str(f), config={"display_name": f.stem, "mime_type": "jsonl"})
        job = c.batches.create(model=GEMINI_MODEL, src=up.name,
                               config={"display_name": f"{args.run_name}_{f.stem}"})
        m["jobs"].append({"job": job.name, "requests_file": f.name, "uploaded_file": up.name,
                          "state": job.state.name, "submitted": datetime.now().isoformat(timespec="seconds"),
                          "collected": False})
        save_manifest(args.run_name, m)  # saved after every job, so nothing is lost
        print(f"  started {job.name}")
    print(f"Job list saved to {manifest_path(args.run_name)}")
    print(f"Check later with: python gemini_lca_batch.py status --run-name {args.run_name}")


# --------------------------------------------------------------------------------------
# status
# --------------------------------------------------------------------------------------
def refresh(c, m: dict):
    for j in m["jobs"]:
        if j.get("collected"):
            continue
        job = c.batches.get(name=j["job"])
        j["state"] = job.state.name
        stats = job.completion_stats
        if stats:
            j["stats"] = {k: v for k, v in stats.model_dump().items() if v is not None}
        if job.dest and job.dest.file_name:
            j["result_file"] = job.dest.file_name
        for k in ("create_time", "start_time", "end_time"):
            v = getattr(job, k, None)
            if v:
                j[k] = v.isoformat(timespec="seconds")


def cmd_status(args):
    m = load_manifest(args.run_name)
    if not m["jobs"]:
        raise SystemExit(f"No batch jobs recorded for run '{args.run_name}'.")
    c = client()
    refresh(c, m)
    save_manifest(args.run_name, m)
    for j in m["jobs"]:
        tag = "collected" if j.get("collected") else j["state"].replace("JOB_STATE_", "").lower()
        took = ""
        if j.get("create_time") and j.get("end_time"):
            t = lambda k: datetime.fromisoformat(j[k])
            took = f"  total {(t('end_time') - t('create_time')).total_seconds() / 60:.0f} min"
            if j.get("start_time"):
                took += f" (queued {(t('start_time') - t('create_time')).total_seconds() / 60:.0f} min)"
        print(f"{j['requests_file']}: {tag}{took}  {j.get('stats', '')}")


# --------------------------------------------------------------------------------------
# collect
# --------------------------------------------------------------------------------------
def response_text(resp: dict) -> tuple[str, str]:
    cands = resp.get("candidates") or []
    if not cands:
        return "", ""
    c0 = cands[0]
    parts = (c0.get("content") or {}).get("parts") or []
    text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
    fr = c0.get("finishReason") or c0.get("finish_reason") or ""
    return text, f"FinishReason.{fr}" if fr else ""


def usage(resp: dict, *keys) -> int:
    u = resp.get("usageMetadata") or resp.get("usage_metadata") or {}
    for k in keys:
        if u.get(k) is not None:
            return u[k]
    return 0


def to_row(key: str, item: dict) -> tuple[str, int, dict, str]:
    name, run = key.split("__run")
    i = int(run)
    text, fr = "", ""
    if "response" in item and item["response"]:
        resp = item["response"]
        text, fr = response_text(resp)
        low, high, ftype = parse_range(text)
        if ftype and "MAX_TOKENS" in fr:
            ftype = "truncated"
        tokens = dict(
            output_tokens=usage(resp, "candidatesTokenCount", "candidates_token_count"),
            thought_tokens=usage(resp, "thoughtsTokenCount", "thoughts_token_count"),
            cached_tokens=usage(resp, "cachedContentTokenCount", "cached_content_token_count"),
            prompt_tokens=usage(resp, "promptTokenCount", "prompt_token_count"))
    else:
        low = high = None
        ftype = "api_error"
        tokens = dict(output_tokens="", thought_tokens="", cached_tokens="", prompt_tokens="")
        print(f"  [{name} run {i}] batch error: {str(item.get('error'))[:150]}")
    ok = ftype == ""
    row = {
        "project": name, "run_idx": i,
        "low": low if low is not None else "", "high": high if high is not None else "",
        "mid": (low + high) / 2 if (ok and low is not None) else "",
        "parse_ok": ok, "failure_type": ftype,
        "ontology_mode": ONTOLOGY_MODE, "geometry_mode": GEOMETRY_MODE,
        "finish_reason": fr, **tokens,
        "latency_s": "",  # not measured in batch mode
        "model": GEMINI_MODEL, "temperature": TEMPERATURE, "thinking_setting": THINKING_SETTING,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    return name, i, row, text


def collect_job(c, j: dict, out: Path) -> tuple[int, int]:
    data = c.files.download(file=j["result_file"])
    local = out / "batch" / f"results_{j['requests_file']}"
    local.write_bytes(data)  # keep the raw batch output

    by_project: dict[str, dict[int, dict]] = {}
    raw_dir = out / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    n, n_fail = 0, 0
    for line in data.decode("utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        name, i, row, text = to_row(item["key"], item)
        if text:
            (raw_dir / f"{name}_run{i}.md").write_text(text, encoding="utf-8")
        by_project.setdefault(name, {})[i] = row
        n += 1
        n_fail += not row["parse_ok"]

    for name, new in by_project.items():
        csv_path = out / f"{name}_runs.csv"
        rows = [r for r in load_existing(csv_path) if int(r["run_idx"]) not in new]
        write_csv(csv_path, rows + list(new.values()))
    return n, n_fail


def cmd_collect(args) -> bool:
    m = load_manifest(args.run_name)
    if not m["jobs"]:
        raise SystemExit(f"No batch jobs recorded for run '{args.run_name}'.")
    c = client()
    refresh(c, m)
    out = run_dir(args.run_name)
    all_done = True
    for j in m["jobs"]:
        if j.get("collected"):
            continue
        state = j["state"]
        if state in OK_STATES and j.get("result_file"):
            n, n_fail = collect_job(c, j, out)
            j["collected"] = True
            print(f"{j['requests_file']}: {n} results written, {n_fail} failed to parse")
        elif state in DONE_STATES:
            j["collected"] = True  # nothing to collect; the runs stay missing
            print(f"{j['requests_file']}: ended as {state}. Run 'submit' again to resend missing runs.")
        else:
            all_done = False
            print(f"{j['requests_file']}: still {state.replace('JOB_STATE_', '').lower()}")
        save_manifest(args.run_name, m)
    if all_done:
        print(f"All jobs collected. Results: {out}")
    return all_done


def cmd_wait(args):
    while not cmd_collect(args):
        print(f"Checking again in {POLL_MINUTES} min ...")
        time.sleep(POLL_MINUTES * 60)


# --------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["submit", "status", "collect", "wait"])
    ap.add_argument("--run-name", required=True, help="Results folder name under results/")
    ap.add_argument("--projects", default="all", help='submit only: "all", "4", "1-20", or "4,10,25-30"')
    ap.add_argument("--runs", type=int, default=10, help="submit only: runs per project (default 10)")
    ap.add_argument("--dry-run", action="store_true", help="submit only: write the request files, upload nothing")
    ap.add_argument("--force", action="store_true", help="submit even if earlier jobs are not collected")
    args = ap.parse_args()
    {"submit": cmd_submit, "status": cmd_status, "collect": cmd_collect, "wait": cmd_wait}[args.step](args)


if __name__ == "__main__":
    main()
