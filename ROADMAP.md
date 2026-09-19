# EvalForge — LLM Output Evaluation Framework
## Claude Code Execution Roadmap

---

## What This Project Is

EvalForge is a standalone framework for evaluating LLM-generated text outputs. It implements:
- A **golden dataset** pipeline with human-annotated ground truth
- **Multi-dimensional scorers** (faithfulness, relevance, coherence, conciseness)
- An **LLM-as-judge** pipeline with calibration and bias detection
- A **FastAPI** serving layer with batch scoring and a metric dashboard
- **CI-gated quality gates** — PRs fail if judge agreement drops below threshold

Demo dataset: CI Brain failure summaries (real LLM output you already have).
This makes EvalForge self-referential — it evaluates its sibling project, which is a strong interview story.

---

## Rules for Claude Code

- Phase-driven, not date-driven. Complete each phase fully before moving to the next.
- Every subphase must be completable in ≤ 20 minutes.
- Each subphase has: implement → test → self-review → fix → re-test → mark complete.
- Self-correct up to 2 times. If still broken after 2 attempts, stop and write to HUMAN_GUIDE.md.
- Prefer simpler solutions. Do not over-engineer.
- Update HUMAN_GUIDE.md at the end of every phase. Mark sections needing human input clearly.
- Never fabricate human annotations, test results, or metric numbers.
- Commit after every phase with message format: `phase-X: <one line description>`
- Do not commit broken code. Run all tests before every commit.
- Git identity: Soumya Padhi <soumyaswarup07@gmail.com>

---

## Project Structure (Build Toward This)

```
evalforge/
├── src/evalforge/
│   ├── config.py          paths, model names, thresholds
│   ├── dataset.py         golden dataset loader and validator
│   ├── scorers/
│   │   ├── rule_based.py  length, keyword overlap, format checks
│   │   ├── embedding.py   semantic similarity via sentence-transformers
│   │   └── llm_judge.py   Gemini-as-judge with prompt templates
│   ├── calibration.py     judge vs human agreement, Cohen's Kappa
│   ├── bias.py            position bias, verbosity bias detection
│   ├── evaluate.py        run full eval pipeline, write report
│   └── api.py             FastAPI: /evaluate, /batch, /report, /health
├── data/
│   ├── golden/            annotated JSONL files (human labels)
│   └── raw/               unannotated LLM outputs to evaluate
├── reports/               JSON eval reports + figures
├── tests/                 pytest suite
├── .github/workflows/     CI: lint + tests + eval gate on every push
├── Dockerfile
├── render.yaml
├── pyproject.toml
├── HUMAN_GUIDE.md
└── ROADMAP.md
```

---

## Phase 0 — Repo Skeleton and Golden Dataset

**Goal:** Repo is set up. Golden dataset exists with 20 annotated examples. All data contracts are tested.

### Subphase 0.1 — Repo Init
- Create the full directory structure above (empty files with docstrings).
- Write `pyproject.toml` with dependencies: `fastapi`, `uvicorn`, `sentence-transformers`, `google-generativeai`, `scikit-learn`, `pytest`, `ruff`, `httpx`.
- Write `config.py`: hardcode all paths, model name (`gemini-1.5-flash`), and thresholds as constants. No magic strings anywhere else.
- Write `.gitignore`: exclude `.env`, `data/raw/`, `reports/`, `__pycache__`, `.venv`.
- **Verify:** `pip install -e ".[dev]"` completes without error. `python -c "import evalforge"` works.

### Subphase 0.2 — Golden Dataset Creation
- In `data/golden/`, create `ci_brain_summaries.jsonl` — 20 examples pulled from CI Brain's actual failure summaries.
- Each record schema:
  ```json
  {
    "id": "cb_001",
    "input": "<the test failure cluster description passed to the LLM>",
    "output": "<the LLM-generated summary>",
    "human_label": {
      "faithfulness": 3,
      "relevance": 3,
      "coherence": 3,
      "conciseness": 2
    },
    "notes": "optional human annotation note"
  }
  ```
- Scores are 1–3 (1=poor, 2=acceptable, 3=good). Keep scale small — 3-point is easier to annotate consistently and reduces noise.
- **HUMAN CHECKPOINT:** Claude Code cannot fabricate these annotations. Write 20 real CI Brain failure summaries into `data/raw/ci_brain_raw.jsonl` first, then pause and write to HUMAN_GUIDE.md:
  > "Phase 0.2 needs your annotations. Open data/raw/ci_brain_raw.jsonl, read each entry, and fill data/golden/ci_brain_summaries.jsonl with your human_label scores. Use the 1–3 scale in the schema above. Do all 20 before resuming."
- Do not proceed past 0.2 until HUMAN_GUIDE.md is updated and the raw file is written.

### Subphase 0.3 — Dataset Validator
- Write `dataset.py` with a `GoldenDataset` class:
  - `load(path) -> List[dict]` — loads and validates JSONL.
  - `validate(record)` — checks schema, score ranges (1–3), no empty strings.
  - `split(train_ratio=0.8)` — deterministic 80/20 split for calibration vs test.
- Write tests in `tests/test_dataset.py`:
  - Schema validation rejects a record missing `human_label`.
  - Score out of range (0 or 4) raises `ValueError`.
  - Split is deterministic — same seed gives same split every time.
  - Dataset loads all 20 records with correct types.
- **Verify:** `pytest tests/test_dataset.py -v` — all pass.

### Phase 0 Exit Gate
- [ ] Repo structure matches the tree above.
- [ ] `pip install -e ".[dev]"` works clean.
- [ ] `data/raw/ci_brain_raw.jsonl` written with 20 raw entries.
- [ ] HUMAN_GUIDE.md updated with annotation instructions.
- [ ] `pytest tests/test_dataset.py` — all pass.
- [ ] Commit: `phase-0: repo skeleton and golden dataset pipeline`

---

## Phase 1 — Rule-Based and Embedding Scorers

**Goal:** Two scorer types are implemented and tested. Each returns a float score in [0, 1].

### Subphase 1.1 — Rule-Based Scorer
- Write `scorers/rule_based.py`:
  - `length_score(output, min_words=10, max_words=80) -> float` — 1.0 if within range, linearly decays outside.
  - `keyword_overlap(output, input, top_n=5) -> float` — TF-IDF top-N keywords from input, fraction present in output.
  - `format_score(output) -> float` — checks no garbled tokens, no repeated phrases (3+ word n-gram repeats), no truncation marker like `...` or `[TRUNCATED]`.
- All scorers return float in [0.0, 1.0]. No exceptions — catch and return 0.0 on bad input.
- Write `tests/test_rule_based.py`:
  - `length_score` returns 1.0 for a 20-word output, < 0.5 for a 3-word output.
  - `keyword_overlap` returns > 0.5 when output contains most input keywords.
  - `format_score` returns 0.0 for an output with 5 repeated trigrams.
- **Verify:** `pytest tests/test_rule_based.py -v` — all pass.

### Subphase 1.2 — Embedding Scorer
- Write `scorers/embedding.py`:
  - Load `all-MiniLM-L6-v2` from `sentence-transformers` once at module level (cache it — don't reload per call).
  - `semantic_similarity(output, reference) -> float` — cosine similarity between embeddings.
  - `relevance_to_input(output, input) -> float` — cosine similarity between output and input embeddings.
- **Important:** The model download happens on first import. In `config.py`, set `EMBEDDING_MODEL = "all-MiniLM-L6-v2"`. Do not hardcode the model name in `embedding.py`.
- Write `tests/test_embedding.py`:
  - Two identical strings return similarity > 0.99.
  - Two unrelated strings ("the cat sat" vs "quarterly revenue declined") return < 0.3.
  - A summary of a text returns relevance > 0.6 to that text.
- **Verify:** `pytest tests/test_embedding.py -v` — all pass. Note: first run downloads ~80MB model, this is expected.

### Subphase 1.3 — Scorer Registry
- Write `scorers/__init__.py` with a `score_all(output, input, reference=None) -> dict` function:
  - Runs all rule-based scorers and embedding scorers.
  - Returns: `{"length": float, "keyword_overlap": float, "format": float, "semantic_sim": float, "relevance": float}`.
  - If `reference` is None, skip `semantic_sim`.
- Write `tests/test_scorer_registry.py`:
  - `score_all` returns all expected keys.
  - All values are floats in [0.0, 1.0].
  - Passing None as reference returns dict without `semantic_sim` key.
- **Verify:** `pytest tests/test_scorer_registry.py -v` — all pass.

### Phase 1 Exit Gate
- [ ] All scorers return floats in [0.0, 1.0] — no exceptions propagate.
- [ ] Embedding model loads once, not per call.
- [ ] `pytest tests/` — all pass.
- [ ] Commit: `phase-1: rule-based and embedding scorers`

---

## Phase 2 — LLM-as-Judge Pipeline

**Goal:** Gemini scores each output on the same 4 dimensions as the golden labels. Prompt templates are versioned. Outputs are parsed reliably.

### Subphase 2.1 — Judge Prompt Templates
- Write `scorers/llm_judge.py`.
- Define prompt templates as module-level constants (not in config.py — they are implementation detail):
  ```python
  JUDGE_PROMPT_V1 = """You are evaluating an AI-generated summary of a software test failure.

  INPUT (what was given to the AI):
  {input}

  OUTPUT (what the AI generated):
  {output}

  Rate the output on each dimension using ONLY the integer 1, 2, or 3:
  - faithfulness: does the output contain only information from the input? (1=hallucinated facts, 2=mostly faithful, 3=fully faithful)
  - relevance: does the output address what the input describes? (1=off-topic, 2=partially relevant, 3=fully relevant)
  - coherence: is the output grammatically correct and easy to read? (1=hard to read, 2=readable, 3=clear and well-written)
  - conciseness: is the output appropriately brief without losing key information? (1=too long/repetitive, 2=acceptable, 3=concise)

  Respond ONLY with valid JSON. No explanation. No markdown. Example:
  {"faithfulness": 3, "relevance": 2, "coherence": 3, "conciseness": 2}
  """
  ```
- **Why this prompt design matters:** Asking for integers 1–3 matches the golden label scale exactly. This is deliberate — it makes Kappa calculation direct with no remapping. Document this in a comment.

### Subphase 2.2 — Judge Caller with Retry
- Implement `judge(input, output, prompt_template=JUDGE_PROMPT_V1, max_retries=3) -> dict`:
  - Calls Gemini API with the filled prompt.
  - Parses JSON response. If parsing fails, retry up to `max_retries` times.
  - On all retries exhausted, return `{"faithfulness": None, "relevance": None, "coherence": None, "conciseness": None}` — never raise, never crash the pipeline.
  - Log each retry attempt with reason.
- Read API key from environment variable `GEMINI_API_KEY`. If not set, raise `EnvironmentError` at import time with a clear message.
- Write `tests/test_llm_judge.py`:
  - A mock that returns valid JSON → `judge()` returns correctly parsed dict.
  - A mock that returns invalid JSON twice then valid → retries correctly, returns parsed dict.
  - A mock that always returns invalid JSON → returns all-None dict after `max_retries`.
  - Missing API key → `EnvironmentError` on import.
- **Verify:** `pytest tests/test_llm_judge.py -v` — all pass (mocked, no real API calls in tests).

### Subphase 2.3 — Batch Judge Runner
- Implement `run_judge_on_dataset(dataset: List[dict], prompt_template=JUDGE_PROMPT_V1) -> List[dict]`:
  - For each record, calls `judge(record["input"], record["output"])`.
  - Appends `judge_label` to each record (same schema as `human_label`).
  - Adds `judge_prompt_version: "v1"` to each record for traceability.
  - Writes output to `reports/judge_outputs.jsonl`.
  - Skips records where judge returned all-None (logs a warning, does not crash).
- Write `tests/test_batch_runner.py`:
  - With 3 mocked records and a mock judge, all 3 get `judge_label` appended.
  - An all-None response is skipped and logged, remaining records are processed.
  - Output file is written with correct record count.
- **Verify:** `pytest tests/test_batch_runner.py -v` — all pass.

### Phase 2 Exit Gate
- [ ] Judge never raises — all failures return None gracefully.
- [ ] Prompt version is tracked in every output record.
- [ ] All tests pass with mocked API — zero real API calls in the test suite.
- [ ] `pytest tests/` — all pass.
- [ ] Commit: `phase-2: LLM-as-judge pipeline with retry and batch runner`

---

## Phase 3 — Calibration and Bias Detection

**Goal:** Measure how well the judge agrees with humans. Detect known judge biases. Report quantified numbers — these become resume bullets.

### Subphase 3.1 — Cohen's Kappa Agreement
- Write `calibration.py`:
  - `compute_kappa(human_labels: List[int], judge_labels: List[int]) -> float`:
    - Use `sklearn.metrics.cohen_kappa_score`.
    - Filter out None judge labels before computing.
    - Return kappa per dimension and weighted average.
  - `agreement_report(dataset_with_judge_labels: List[dict]) -> dict`:
    - Extracts human and judge scores per dimension.
    - Returns: `{"faithfulness": kappa, "relevance": kappa, "coherence": kappa, "conciseness": kappa, "weighted_avg": kappa}`.
    - Saves to `reports/calibration_report.json`.
- Kappa interpretation to document in comments: < 0.2 = poor, 0.2–0.4 = fair, 0.4–0.6 = moderate, 0.6–0.8 = substantial, > 0.8 = near-perfect.
- Write `tests/test_calibration.py`:
  - Identical human and judge lists → kappa = 1.0.
  - Completely random disagreement → kappa ≈ 0.0 (allow ±0.2 tolerance).
  - None values in judge labels are correctly filtered before computation.
- **Verify:** `pytest tests/test_calibration.py -v` — all pass.

### Subphase 3.2 — Position Bias Detection
- Write `bias.py`:
  - **What position bias is:** When asked to compare two outputs (A vs B), LLMs tend to prefer whichever appears first. We detect this by presenting the same pair in both orders and checking if the judge flips.
  - `detect_position_bias(pairs: List[tuple], judge_fn) -> dict`:
    - For each (output_A, output_B) pair, call judge twice: once with A first, once with B first.
    - A flip = judge prefers A when A is first but B when B is first.
    - Returns: `{"total_pairs": int, "flips": int, "flip_rate": float}`.
    - A flip rate > 0.3 is considered significant — document this threshold in config.py.
  - **Important:** This requires 10 pairs minimum to be meaningful. Write a guard that raises `ValueError` if fewer than 10 pairs are provided.
- Write `tests/test_bias.py`:
  - A mock judge that always flips → flip rate = 1.0.
  - A mock judge that never flips → flip rate = 0.0.
  - Fewer than 10 pairs → `ValueError`.
- **Verify:** `pytest tests/test_bias.py -v` — all pass.

### Subphase 3.3 — Verbosity Bias Detection
- Add to `bias.py`:
  - **What verbosity bias is:** LLMs tend to rate longer outputs higher regardless of quality.
  - `detect_verbosity_bias(dataset_with_judge_labels: List[dict]) -> dict`:
    - Computes Pearson correlation between output word count and judge's average score.
    - Returns: `{"pearson_r": float, "p_value": float, "significant": bool}`.
    - Uses `scipy.stats.pearsonr`.
    - `significant = True` if `p_value < 0.05 and pearson_r > 0.3`.
  - Add `scipy` to dependencies in `pyproject.toml`.
- Write `tests/test_verbosity_bias.py`:
  - Synthetic dataset where longer outputs always get score 3 and short get score 1 → high pearson_r, significant = True.
  - Dataset with random length/score relationship → significant = False.
- **Verify:** `pytest tests/test_verbosity_bias.py -v` — all pass.

### Subphase 3.4 — Full Evaluation Report
- Write `evaluate.py`:
  - `run_full_eval(golden_dataset_path, output_path="reports/eval_report.json")`:
    1. Load golden dataset.
    2. Run all rule-based and embedding scorers on each record.
    3. Run judge on each record.
    4. Compute Kappa agreement.
    5. Run verbosity bias detection.
    6. Write single JSON report with all metrics.
  - Report schema:
    ```json
    {
      "dataset_size": 20,
      "scorer_averages": {"length": 0.82, "keyword_overlap": 0.71, ...},
      "calibration": {"faithfulness": 0.65, "weighted_avg": 0.61},
      "bias": {"verbosity": {"pearson_r": 0.21, "significant": false}},
      "judge_coverage": 0.93
    }
    ```
- Write `tests/test_evaluate.py`:
  - With mocked dataset, judge, and scorers → report is written with correct schema.
  - `judge_coverage` = fraction of records where judge did not return all-None.
- **Verify:** `pytest tests/test_evaluate.py -v` — all pass.
- **HUMAN CHECKPOINT:** After this subphase, run `python -m evalforge.evaluate` with your real annotated golden dataset and real Gemini API key. Read the calibration report. If weighted Kappa < 0.4, write to HUMAN_GUIDE.md — the prompt needs revision before proceeding to Phase 4.

### Phase 3 Exit Gate
- [ ] Calibration report written with Kappa per dimension.
- [ ] Position bias detector works with mocked judge.
- [ ] Verbosity bias detector computes correct correlation.
- [ ] Full eval pipeline runs end-to-end with mocked data.
- [ ] HUMAN_GUIDE.md updated: weighted Kappa from real run recorded.
- [ ] `pytest tests/` — all pass.
- [ ] Commit: `phase-3: calibration and bias detection`

---

## Phase 4 — FastAPI Service

**Goal:** EvalForge is served as an API. Single and batch endpoints. Health and report endpoints. Schema-validated in and out.

### Subphase 4.1 — API Skeleton and Health
- Write `api.py`:
  - Import and initialise: load embedding model, load golden dataset, load latest eval report if exists.
  - `GET /` → redirect to `/docs`.
  - `GET /health` → `{"status": "ok", "dataset_size": int, "embedding_model": str, "judge_model": str}`.
  - `GET /report` → returns latest `reports/eval_report.json` if it exists, else `{"error": "no report found"}`.
- Write `tests/test_api.py` using `httpx.AsyncClient`:
  - `GET /health` → 200, correct schema.
  - `GET /report` with no report file → returns error dict, not 500.
- **Verify:** `pytest tests/test_api.py -v` — all pass.

### Subphase 4.2 — Single Evaluation Endpoint
- Add `POST /evaluate` to `api.py`:
  - Request schema:
    ```json
    {"input": "string", "output": "string", "reference": "string (optional)"}
    ```
  - Response schema:
    ```json
    {
      "scores": {"length": 0.9, "keyword_overlap": 0.7, "semantic_sim": 0.85, ...},
      "judge": {"faithfulness": 3, "relevance": 2, "coherence": 3, "conciseness": 2},
      "judge_prompt_version": "v1"
    }
    ```
  - If `GEMINI_API_KEY` is not set, `judge` field returns `null` — do not crash.
- Add to `tests/test_api.py`:
  - Valid input → 200, correct response schema.
  - Missing `output` field → 422 validation error.
  - No API key set → 200 with `judge: null`, scores still returned.
- **Verify:** `pytest tests/test_api.py -v` — all pass.

### Subphase 4.3 — Batch Endpoint
- Add `POST /evaluate/batch` to `api.py`:
  - Request: `{"items": [{"input": str, "output": str}, ...]}` — max 50 items.
  - Response: `{"results": [...], "total": int, "failed": int}`.
  - Items where judge fails still return rule-based scores.
  - Enforce max 50 with a 422 if exceeded.
- Add to `tests/test_api.py`:
  - Batch of 3 → response has 3 results.
  - Batch of 51 → 422.
  - Batch where one item has empty output → that item's scores are 0.0, not a crash.
- **Verify:** `pytest tests/test_api.py -v` — all pass.

### Phase 4 Exit Gate
- [ ] API starts with `uvicorn evalforge.api:app --reload`.
- [ ] `/docs` renders all endpoints with correct schemas.
- [ ] No endpoint returns 500 under any valid input.
- [ ] `pytest tests/test_api.py -v` — all pass.
- [ ] Commit: `phase-4: FastAPI evaluation service`

---

## Phase 5 — Docker, CI, and Deployment

**Goal:** Containerised. CI runs on every push. Deployed live on Render. A CI quality gate blocks merges if judge agreement drops.

### Subphase 5.1 — Dockerfile
- Write `Dockerfile`:
  - Base: `python:3.11-slim`.
  - Install dependencies: `pip install -e .` — do not commit `.venv` or install dev deps in image.
  - Download embedding model at build time: `RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"` — bake it into the image so startup is instant.
  - Do NOT train or run the judge at build time — it needs the API key at runtime.
  - Start command: `uvicorn evalforge.api:app --host 0.0.0.0 --port $PORT`.
- Write `.dockerignore`: exclude `data/raw/`, `reports/`, `.env`, `.venv`, `tests/`.
- **Verify:** `docker build -t evalforge .` completes. `docker run -p 8000:8000 evalforge` starts. `curl localhost:8000/health` returns 200.

### Subphase 5.2 — GitHub Actions CI
- Write `.github/workflows/ci.yml`:
  - Trigger: push and pull_request on `main`.
  - Jobs:
    1. **lint** — `ruff check src/ tests/`
    2. **test** — `pytest tests/ -v` (no API key needed — all judge calls are mocked)
    3. **eval-gate** — runs only on push to main:
       - Runs `python -m evalforge.evaluate` with the golden dataset.
       - Reads `reports/eval_report.json`.
       - Fails the job if `calibration.weighted_avg < 0.4`.
       - This is the CI quality gate — it enforces judge quality on every deploy.
  - `GEMINI_API_KEY` is a GitHub Actions secret used only in the eval-gate job.
- **Verify:** Push to main. All 3 jobs green in Actions tab.

### Subphase 5.3 — Render Deployment
- Write `render.yaml`:
  ```yaml
  services:
    - type: web
      name: evalforge
      runtime: docker
      buildCommand: ""
      startCommand: uvicorn evalforge.api:app --host 0.0.0.0 --port $PORT
      envVars:
        - key: GEMINI_API_KEY
          sync: false
  ```
- Set `GEMINI_API_KEY` in Render dashboard environment variables.
- **Verify:** Live URL returns 200 on `/health`. `POST /evaluate` with a real CI Brain summary returns scores and a non-null judge response.
- **HUMAN CHECKPOINT:** Write the live URL to HUMAN_GUIDE.md.

### Phase 5 Exit Gate
- [ ] `docker build` and `docker run` succeed locally.
- [ ] All 3 CI jobs green on push to main.
- [ ] Live Render URL is in HUMAN_GUIDE.md.
- [ ] `POST /evaluate` on live URL returns real judge scores.
- [ ] Commit: `phase-5: Docker, CI quality gate, and Render deployment`

---

## Phase 6 — README, Resume Bullets, and Interview Defence

**Goal:** Project is fully documented. Resume bullets are written. You can explain every decision without notes.

### Subphase 6.1 — README
Write `README.md` covering:
1. **What EvalForge does** — one paragraph, no jargon.
2. **Results** — table with actual Kappa scores from your real run. Do not fabricate.
3. **Why these design decisions:**
   - 1–3 scale (matches human annotation ergonomics, reduces label noise vs 1–5)
   - LLM-as-judge instead of human-only (scalable; human labels are ground truth for calibration only)
   - `class_weight` equivalent: why we track `judge_coverage` (None responses are not random — they signal hard cases)
   - Bias detection (a judge that prefers verbose outputs silently inflates scores)
4. **Quickstart** — clone, install, run, hit `/docs`.
5. **API reference** — endpoint table.
6. **Honest limitations** — position bias is detected but not corrected; 20-example golden set is small.

### Subphase 6.2 — Resume Bullets
Write 4 resume bullets in this exact format (same as Contrail/CI Brain):

```
- Engineered a [WHAT] that [DID WHAT], achieving [METRIC] on [DATASET SIZE] examples.
- Designed a [WHAT] detecting [WHAT BIAS], measuring [METRIC] via [METHOD].
- Built [N] tests covering [WHAT]; CI [QUALITY GATE] blocks deploys if [CONDITION].
- Deployed a [WHAT] with [ENDPOINTS], [INFRA DETAIL], and [OBSERVABILITY DETAIL].
```

Fill in with your real numbers after Phase 3 completes. Do not write placeholder numbers.

### Subphase 6.3 — Interview Checkpoint Questions
Write answers to these in HUMAN_GUIDE.md. You must be able to answer all without notes before any interview:

1. What is Cohen's Kappa and why is it better than raw percentage agreement?
2. What is position bias in LLM judges? How did you detect it?
3. Why did you use a 1–3 scale instead of 1–5?
4. What does your CI quality gate check? What happens if it fails?
5. Why is `judge_coverage` tracked? What does a low coverage signal?
6. What is the difference between faithfulness and relevance in your scoring dimensions?
7. Why was `all-MiniLM-L6-v2` chosen over a larger embedding model?
8. What would a production upgrade of this system look like? (Answer: third split for threshold search, online annotation tool, multiple judge models for ensemble, drift monitoring on score distributions)

### Phase 6 Exit Gate
- [ ] README is accurate — all metrics match `reports/eval_report.json`.
- [ ] Resume bullets contain real Kappa numbers and dataset size.
- [ ] All 8 interview questions answered in HUMAN_GUIDE.md.
- [ ] `pytest tests/` — all pass.
- [ ] Final commit: `phase-6: README, resume bullets, interview defence pack`

---

## HUMAN_GUIDE.md — What Claude Code Will Write Here

Claude Code updates this file at the end of every phase. It contains:
- Phase completion status
- Any human actions required (annotations, API keys, manual verification steps)
- Live URL once deployed
- Real metric numbers from actual runs
- Escalation notes if Claude Code hit 2 self-correction attempts and stopped

Claude Code never removes content from HUMAN_GUIDE.md — only appends.

---

## Final Resume Entry (fill after Phase 3)

```latex
\resumeSubheading
{EvalForge — LLM Output Evaluation Framework}{2026}
{Python, FastAPI, Gemini API, sentence-transformers, Docker, GitHub Actions, Render}{\href{https://github.com/Som0111/evalforge}{\faGithub\ GitHub}}

\resumeItemListStart
\resumeItem{Engineered a \textbf{multi-dimensional LLM-as-judge pipeline} scoring outputs on faithfulness, relevance, coherence, and conciseness, achieving \textbf{Cohen's Kappa of [X]} against human annotations on \textbf{20 golden examples}.}
\resumeItem{Designed a \textbf{judge bias detection system} measuring position bias (flip rate across swapped pairs) and verbosity bias (Pearson correlation between output length and judge score), quantifying systematic scoring errors.}
\resumeItem{Built \textbf{[N] tests} covering scorer contracts, judge retry logic, and API schema validation; a \textbf{CI quality gate} blocks deploys to Render if weighted judge-human agreement drops below threshold.}
\resumeItem{Deployed a \textbf{Dockerised FastAPI service} with single and batch evaluation endpoints, embedding model baked into the image for instant startup, and live on Render as a \textbf{demo on CI Brain's failure summaries}.}
\resumeItemListEnd
```

Replace `[X]` and `[N]` with real numbers after Phase 3.

---

*End of ROADMAP.md*
