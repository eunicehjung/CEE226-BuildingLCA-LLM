# LLM-Based Embodied Carbon for Early-Stage Building LCA

This project uses Gemini to test the A–C embodied carbon intensity (ECI, kgCO₂e/m²) of 292 buildings from early-stage metadata. Each building is estimated 10 times, and the estimates are compared against ground-truth LCA results.

The requests are sent through the Gemini Batch API: all 2,920 calls are submitted as one job, which Google processes on its servers, at half the standard price.

For step-by-step instructions written for non-programmers, see **[Gemini_ECI_run_guide.pdf](Gemini_ECI_run_guide.pdf)**.

## Files

| File | Purpose |
|---|---|
| `gemini_lca_batch.py` | Builds the prompts, submits the batch job, and collects the results |
| `LLM-data-cleaning.py` | Summarizes a run: mean estimate per building, error, RMSE |
| `make_fig1_bands.py` | figure |
| `ground_truth.csv` | True ECI per building |
| `references/` | Building metadata, data glossary, LCA practice guide, background papers |
| `Gemini_ECI_run_guide.pdf` | Full run guide |

## Setup

Run this once per computer:

```bash
conda create -n gemini python=3.12 -y
conda activate gemini
pip install google-genai pandas openpyxl PyPDF2 python-dotenv matplotlib scipy
```

Create a file named `.env` in the project folder containing your own key:

```
GEMINI_API_KEY=your-key-here
```

**Never commit `.env`.** Each user needs their own key.

## Run

Activate the environment, go to the project folder, and pick a run name (`yyyymmdd_HH_description`):

```bash
conda activate gemini
cd CEE226-reference

python gemini_lca_batch.py submit  --run-name result-folder-name   # 1. submit
python gemini_lca_batch.py status  --run-name result-folder-name   # 2. check progress
python gemini_lca_batch.py collect --run-name result-folder-name   # 3. collect when succeeded
python LLM-data-cleaning.py result-folder-name                     # 4. summary table
python make_fig1_bands.py   result-folder-name                     # 5. band figure
```

To redo failed runs (step 3), set `RETRY_FAILED = True` in `gemini_lca_batch.py`, then submit and collect again with the same run name.

Results are written to `results/<run-name>/`. Model settings and prompts are at the top of `gemini_lca_batch.py`.

## Notes

- `results/` and `.env` should not be uploaded. Add both to `.gitignore`.
- Always use the same run name for submit, status, and collect. A new name resubmits, and is billed for, all 2,920 requests.
