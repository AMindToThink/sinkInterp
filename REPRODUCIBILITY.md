# Reproducibility — sinkInterp

Interpretability of attention "sink" values and head matrix norms. (The README notes it
was made public early so it could be cloned onto a vast.ai machine for more experiments.)

## Environment
- Python **3.11.13** (recorded in `vastai_extra_command.txt`, which also captures the exact
  vast.ai environment that was used).
- Fully pinned dependencies (exact `==` versions) are in **`requirements.txt`**. Recreate:
  ```bash
  uv venv --python 3.11 && uv pip install -r requirements.txt
  # or: python3.11 -m venv .venv && .venv/bin/pip install -r requirements.txt
  ```

## Models & data (all auto-downloaded from HuggingFace on first run)
- Models: **`gpt2`** and **`openai/gpt-oss-20b`** (large — needs a substantial GPU), loaded
  via `from_pretrained(...)`.
- Dataset: **`wikitext`** via `datasets.load_dataset("wikitext", ...)`, cached under
  `~/.cache/huggingface`. No manual download required.

## Entry points
- `src/` analysis scripts: `findHeadMatrixNorms.py`,
  `evenOddLayerMatrixNormsAnalysis.py`, `plotSinkValues.py`,
  `ablationPerplexity/headAblationLoop.py`, and others.
- Precomputed outputs live in `src/head_norms*.json`, `plots/`, and `output.png`.
- Tests: `pytest` (see `pytest.ini`, `tests/`).

---
> Reproducibility note: added 2026-07-02 during a machine migration.
