# Reproducing sinkInterp

This repo studies attention "sink" values in `openai/gpt-oss-20b` (a GPT-OSS mixture-of-experts
model with a learned per-head attention "sink" parameter) — correlating sink magnitude with
head weight-matrix norms and with perplexity under head ablation. Everything below was
determined by reading `requirements.txt`, the scripts in `src/`, and `tests/`; items that
could not be verified from the repo are marked `TODO (Matthew)`.

## 1. Environment

- The committed `.venv` (`pyvenv.cfg`) was built with **`uv 0.6.3`** on **Python 3.10.12**.
- `vastai_extra_command.txt` is a separate captured package listing from the vast.ai GPU
  machine, headed `Using Python 3.11.13 environment at: /workspace/sinkInterp/.venv`. It is
  **not identical** to `requirements.txt` — e.g. it additionally has `triton==3.4.0` and
  `kernels==0.9.0`, and slightly different `transformers`/`numpy`/`fsspec`/`networkx`/
  `filelock` versions and a different `triton-kernels` git commit pin. This suggests GPU runs
  on vast.ai needed `triton` + `kernels` installed on top of `requirements.txt` (likely for
  gpt-oss-20b's MXFP4 kernels), but the exact command that produced this file is not recorded.
  `TODO (Matthew): confirm whether `triton`/`kernels` must be installed manually for GPU runs,
  and with which exact versions.`

Recreate the environment with `uv` (per user's global convention):

```bash
cd sinkInterp
uv venv --python 3.10   # or 3.11 — TODO (Matthew): confirm which Python was authoritative
uv pip install -r requirements.txt
# GPU-only scripts may additionally need (unverified, see note above):
# uv add triton kernels
```

Plain `pip` also works: `pip install -r requirements.txt`. Note `requirements.txt` includes a
git dependency (`triton-kernels @ git+https://github.com/triton-lang/triton.git@<sha>#subdirectory=python/triton_kernels`),
so `git` must be installed and able to reach GitHub at install time.

## 2. Data

- **`wikitext` / `wikitext-2-raw-v1`, `test` split** — the only real dataset used, pulled via
  `datasets.load_dataset("wikitext", "wikitext-2-raw-v1", split="test")` in
  `src/ablationPerplexity/headAblation.py` and `src/ablationPerplexity/headAblationLoop.py`.
  Both scripts then filter empty strings and take the first 50 texts. Auto-downloads from the
  Hugging Face Hub on first run and caches under `~/.cache/huggingface` — no manual steps.
- `src/perplexityMeasure.py` and `tests/utils/test_perplexity.py` use small hardcoded strings
  (Lorem ipsum, "Happy Birthday to you!", the US Constitution preamble) — not a dataset file.
- No other datasets, CSVs, or download URLs were found in the code.

## 3. Running / entry points

All scripts are `# %%`-delimited (VS Code / Jupyter "interactive Python" cells) but also run
top-to-bottom as plain scripts. **They open relative paths (e.g. `"head_norms.json"`), so run
each from the directory it lives in** (`src/` or `src/ablationPerplexity/`).

Analysis pipeline, in the order the data flows:

```bash
cd src

# 1. Quick look at raw sink values per layer (loads gpt-oss-20b on CPU)
python plotSinkValues.py

# 2. Compute per-head weight-matrix norms (Frobenius/spectral/expected-quadratic-form)
#    for openai/gpt-oss-20b -> writes head_norms.json
python findHeadMatrixNorms.py
# NOTE: the file's own comment says "Change to 120b for larger test" to produce
# head_norms_120b.json (already committed) — but no 120b variant is committed as a
# separate script/model id. TODO (Matthew): confirm the exact HF model id used
# (e.g. openai/gpt-oss-120b) and whether it needs a code edit or a CLI arg.

# 3. Correlate sink values against those norms -> sink_vs_matrix_norms.png, sink_vs_norms_data.json
python findHeadMatrixNormsAnalysis.py

# 4. Same, split by even/odd layer index, with significance tests
#    -> even_odd_sink_vs_matrix_norms.png, even_odd_sink_vs_norms_data.json
python evenOddLayerMatrixNormsAnalysis.py

# 5. Correlate the 20B vs 120B head-norm JSONs against each other
python size_matrix_correlation.py

cd ablationPerplexity

# 6. Sweep perplexity impact of "ablating" (sink -> ~inf) every (layer, head) pair on GPU
#    -> head_ablation_results.json (resumable; skips already-tested pairs)
python headAblationLoop.py

# 7. Plot sink value vs. mean perplexity from the ablation sweep
python analysis.py
```

Misc / debug scripts (not part of the main pipeline):
- `src/ablationPerplexity/headAblation.py` — one-off perplexity check on wikitext with gpt-oss-20b.
- `src/perplexityMeasure.py` — sanity-checks `evaluate`'s perplexity metric against `google/gemma-2-2b`.
- `src/runOSS20Debug.py` — drops into `pdb` before `model.generate(...)`; interactive only.

Tests:
```bash
pytest                       # uses pytest.ini (testpaths=tests)
# or: uv run pytest
```
`tests/utils/test_grouped_query_attention.py` loads `openai/gpt-oss-20b` in bf16 (CPU-capable
but slow). `tests/utils/test_perplexity.py` is marked `@pytest.mark.gpu`/`@pytest.mark.model`
and hardcodes `.to("cuda")` plus the gated `google/gemma-2-2b` model — see §4.

## 4. External dependencies

- **Hugging Face Hub models**, downloaded via `from_pretrained(...)`, no code changes needed:
  - `openai/gpt-oss-20b` — used throughout `src/` and in `tests/utils/test_grouped_query_attention.py`.
  - `google/gemma-2-2b` — used in `src/perplexityMeasure.py` and `tests/utils/test_perplexity.py`.
    Gemma models are gated on Hugging Face: you must accept the license on the model page and
    authenticate (`huggingface-cli login` or an `HF_TOKEN` env var) before this will download.
  - `head_norms_120b.json` in `src/` implies a "120B" gpt-oss model was also used at some
    point; no exact HF model id for it is recorded in any committed script.
    `TODO (Matthew): record the exact model id (and command) used to generate head_norms_120b.json.`
- **No `wandb` usage, no `os.environ`/`.env` reads, and no API keys were found anywhere in the
  codebase** (checked via grep over all `*.py`). The only credential-like requirement is the HF
  auth needed for the gated `google/gemma-2-2b` model above.
- **No sibling/local repos are imported** — all imports resolve to PyPI/HF packages or files
  within this repo.

## 5. Hardware

- No VRAM/GPU-model figure is stated anywhere in the repo.
- Scripts that **require CUDA** (hardcoded `.to("cuda")`, will error without a GPU):
  `src/ablationPerplexity/headAblationLoop.py`, `src/ablationPerplexity/headAblation.py`,
  and `tests/utils/test_perplexity.py` (marked `@pytest.mark.gpu`).
- `src/plotSinkValues.py` explicitly loads the model on **CPU** (`device_map='cpu'`).
- `src/findHeadMatrixNorms.py`, `src/findHeadMatrixNormsAnalysis.py`,
  `src/evenOddLayerMatrixNormsAnalysis.py` load the model with no explicit device, so they run
  wherever `from_pretrained` defaults land (CPU unless HF/accelerate auto-places it).
- `vastai_extra_command.txt` confirms a vast.ai rented GPU instance was used for at least some
  runs, but not which GPU/VRAM size.
  `TODO (Matthew): note the GPU type/VRAM used on vast.ai for the ablation sweep and any other
  GPU-only scripts, since openai/gpt-oss-20b is large (~20B/MoE) and CPU/GPU memory needs are
  not otherwise documented here.`
