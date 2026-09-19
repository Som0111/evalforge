# EvalForge — Human Guide

Claude Code appends to this file at the end of every phase. Never removed, only appended.

---

## Phase 0.2 — Annotations needed (YOU)

Dataset size is **20** (not 30): the Gemini free tier capped generation at 20 requests/day.

Phase 0.2 needs your annotations. Open `data/raw/ci_brain_raw.jsonl`, read each entry (`input` = prompt sent to the LLM, `output` = its summary), and write `data/golden/ci_brain_summaries.jsonl` — one JSON line per entry, same `id`/`input`/`output`, plus:

```json
"human_label": {"faithfulness": 3, "relevance": 3, "coherence": 3, "conciseness": 2},
"notes": "optional"
```

Scale is 1–3 (1 = poor, 2 = acceptable, 3 = good). Do all 20 before resuming. Drop the `_source` field.

- faithfulness: does the summary stick to the evidence shown (no invented causes)? Saying "evidence too thin" counts as faithful.
- relevance: does it answer the prompt (root-cause hypothesis, 2–3 sentences)?
- coherence: is it clear and well-structured?
- conciseness: does it respect the 2–3 sentence ask?

## Phase 1 — Scorers (done, 23 tests pass)

- `scorers/rule_based.py`: `length_score`, `keyword_overlap`, `format_score`. Bad input returns 0.0.
- `scorers/embedding.py`: `semantic_similarity`, `relevance_to_input`. Model loads lazily once (cached), not at import, so `import evalforge` stays cheap. A model load/download failure raises on purpose: swallowing it would turn every score into a silent 0.0.
- `scorers/__init__.py`: `score_all(output, input, reference=None)`.
- Measured on the 20 real CI Brain pairs (no human labels involved): `relevance` ranges 0.28 to 0.69 (mean 0.53); `keyword_overlap` 0.00 to 1.00 (mean 0.43). The roadmap's ">0.6 for a summary" only held on plain-prose test text. Expect real CI summaries to score lower.
- Known weakness: TF-IDF keywords from CI prompts are identifier-heavy (`test_dicttoolz`, `testcustommapping`), so `keyword_overlap` rewards verbatim test names.

## Scoring page (YOU)

Open `data/raw/score_ci_brain.html` in a browser. Score all 20, click Export JSONL, and save the download as `data/golden/ci_brain_summaries.jsonl` (progress is kept in the browser if you close the tab).

## Phase 2 — LLM judge (done, 39 tests pass, all mocked)

- `scorers/llm_judge.py`: `JUDGE_PROMPT_V1`, `judge()` (retry, never raises), `run_judge_on_dataset()` (resumable, appends to `reports/judge_outputs.jsonl`).
- Deviations from the roadmap, both forced:
  - SDK is `google-genai`, not `google-generativeai` (the latter is end-of-life). `pyproject.toml` updated.
  - `gemini-1.5-flash` no longer exists. `JUDGE_MODEL` is `gemini-3.1-flash-lite`: `gemini-2.5-flash` is closed to new users, and the summaries were written by `gemini-3.5-flash`, so a different model avoids self-preference bias.
- The roadmap prompt's JSON example had unescaped braces that would crash `str.format`; they are doubled in the template.
- A 429 (quota) returns all-None immediately instead of retrying. Re-running `run_judge_on_dataset` skips records already judged and retries the rest.
- Smoke check on 2 real records (2 API calls): judge gave 3/3/3/3 where you gave 2/3/3/2. The judge looks more lenient than you; Phase 3 will quantify it.
- Caveat: the judge prompt's dimension definitions are the roadmap's, not the rubric shown on the scoring page, so some disagreement is rubric mismatch, not judge error.

## Phase 3 — Calibration and bias (code done, 55 tests pass) — HUMAN DECISION NEEDED

Real run: `python -m evalforge.evaluate`, 20 records, judge `gemini-3.1-flash-lite`, prompt v1, judge_coverage 1.0.
Report: `reports/eval_report.json`, `reports/calibration_report.json`.

| dimension | quadratic-weighted kappa | exact agreement |
|---|---|---|
| faithfulness | 0.07 | 50% |
| relevance | 0.00 | 85% |
| coherence | 0.00 | 70% |
| conciseness | 0.00 | 20% |
| **weighted avg** | **0.016** | |

**This fails the 0.4 gate. The roadmap says the prompt needs revision before Phase 4.**

Why (from the label distributions):
- The judge answered 3 for relevance, coherence and conciseness on all 20 records. A constant rater has kappa 0 no matter what, so those three numbers say "judge does not discriminate", not "judge is randomly wrong".
- Conciseness is the real miss: you gave 1 to six outputs (71-104 words) and 2 to ten; the judge gave 3 to all. The prompt's conciseness definition never mentions the "2-3 sentences" ask.
- Faithfulness is the only dimension where the judge varied (9x 2, 11x 3). It agreed with you on 10 of 20; when it disagreed it was more lenient (9 of 10 cases).
- Your own labels are skewed (relevance 17/20 are 3, coherence 14/20), so even a good judge gets a noisy kappa on this dimension. Treat the relevance/coherence kappas as low-information either way.
- Verbosity bias: r = 0.04, p = 0.87, not significant (n = 20).
- Position bias: detector is built and unit-tested with a mock judge only. There is no real pairwise judge prompt, so no real flip rate exists yet.

Options: (a) write `JUDGE_PROMPT_V2` with your rubric (state the 2-3 sentence limit, ask for stricter grading, keep it 1-3), rerun, compare; (b) accept the number and report it honestly with the caveats above. Caution for (a): tuning the prompt on these same 20 labels overfits; the 4-record test split from `GoldenDataset.split` is too small to check it. Do not put a Kappa on the resume until this is settled.

## Phase 3 — Prompt v2 result: gate still FAILED

`JUDGE_PROMPT_V2` (your rubric wording, sentence-count anchors taken from the "2-3 sentences" ask, "be strict, choose the lower grade") run on the same 20 records, same judge model (`gemini-3.1-flash-lite`). v1 files are kept (`judge_outputs.jsonl`, `*_v1.backup.*`); v2 results are in `judge_outputs_v2.jsonl`, `calibration_report_v2.json`, `eval_report_v2.json`.

| dimension | kappa v1 | kappa v2 | agreement v1 -> v2 |
|---|---|---|---|
| faithfulness | 0.07 | 0.04 | 50% -> 45% |
| relevance | 0.00 | 0.00 | 85% -> 85% |
| coherence | 0.00 | 0.00 | 70% -> 70% |
| conciseness | 0.00 | 0.18 | 20% -> 30% |
| **weighted avg** | **0.02** | **0.05** | |

- The judge still gives 3 to all 20 on relevance and coherence, so those kappas are 0 by construction.
- Conciseness improved a little (5x 2, 15x 3) but is still lenient: you scored 16 of 20 at 1 or 2.
- Faithfulness got slightly worse. Verbosity r = -0.34 (p = 0.15, not significant, n = 20).
- Conclusion: rewording the prompt did not fix this. The likely limit is the small judge model not following the strictness instruction, and/or a real gap between its notion of faithfulness and yours. Not yet tested: a stronger judge model.
- Ideas, in order of cost: (1) same v2 prompt on a stronger model (e.g. `gemini-3.6-flash` or `gemini-3.1-pro-preview`), one-line change in `config.JUDGE_MODEL`; (2) measure conciseness deterministically (sentence count) instead of asking an LLM; (3) few-shot examples, which must come from outside these 20 or the kappa is inflated.
- Do not put a Kappa on the resume from these runs.

## Phase 3 — FINAL FINDING (accepted, decision by the project owner)

**Weighted Cohen's kappa between the LLM judge and human labels is about 0.05. This is below the 0.4 gate and was accepted as the honest result rather than tuned further.**

Two judge configurations on the same 20 human-labelled CI Brain summaries, prompt v2:

| Judge model | Records judged | Weighted kappa |
|---|---|---|
| `gemini-3.1-flash-lite` | 20 of 20 | 0.05 |
| `gemini-3.6-flash` | 12 of 20 (quota stopped the run) | 0.04 |

Drivers:
- **Systematic leniency on conciseness.** Humans scored 16 of 20 outputs 1 or 2 against the "2-3 sentences" ask. The judge scored most 3.
- **Zero discrimination on relevance and coherence.** The judge gave 3 to every record on both, so those kappas are 0 by construction (a constant rater has no agreement beyond chance).
- Faithfulness was the only dimension where the judge varied, and it agreed with humans no better than chance (kappa 0.04 and -0.05).
- Human labels are themselves skewed (17/20 relevance = 3), so per-dimension kappas on n = 20 are noisy.

What this does and does not support:
- It does NOT support any claim that the judge agrees with humans. Do not put a Kappa figure on the resume as a success metric.
- It supports: a working calibration pipeline that measured, and exposed, a lenient judge. Prompt v1 -> v2 and model lite -> 3.6-flash did not change that.
- Verbosity bias: not significant (v1 r = 0.04; v2 r = -0.34, p = 0.15, n = 20). Position bias: detector unit-tested with a mock only, no real measurement.
- Untried: a deterministic sentence-count check for conciseness; few-shot examples from outside the 20 labelled records.

Files: v1 `reports/judge_outputs.jsonl` + `calibration_report.json`; v2 lite `*_v2.*`; v2 3.6-flash `*_v2_gemini-3.6-flash.*`.

## Phase 4 — FastAPI service (done, 73 tests pass)

Run: `uvicorn evalforge.api:app --reload` (set `GEMINI_API_KEY` to enable the judge; without it `judge` is `null` and scores still return).
- `GET /health`, `GET /report` (newest `reports/eval_report*.json`, backups excluded), `POST /evaluate`, `POST /evaluate/batch` (1-50 items, 422 outside that).
- The judge uses prompt `v2`. Given the ~0.05 kappa above, treat the `judge` field as an unvalidated, lenient signal.
- Empty `input`/`output` scores 0.0 and is never sent to the judge. `failed` in the batch response counts judge attempts that returned nothing.
- Before deploying (Phase 5): the endpoints are unauthenticated and each judged request spends Gemini quota, so add an API key or rate limit first. Batch judging is sequential and can be slow. The embedding model loads on the first scored request, not at startup.
- Tests use Starlette's `TestClient` rather than `httpx.AsyncClient` (same httpx underneath, no async plugin needed).

## Phase 5 — Docker, CI, Render (files done and verified locally; deploy steps are YOURS)

Auth: `POST /evaluate` and `/evaluate/batch` need an `X-API-Key` header equal to the server's `EVALFORGE_API_KEY`. Missing/wrong key -> 401. Server with no key configured -> 503 (fails closed). `/health` and `/report` stay open.

Verified locally: `docker build -t evalforge .` succeeds (image 2.4 GB), the container runs as a non-root user, `/health` -> 200, unauthenticated `/evaluate` -> 401, keyed `/evaluate` -> 200. 76 tests pass.
NOT verified (needs your accounts): the GitHub Actions run, the Render deploy, and a live judged `/evaluate` call.

**The CI eval-gate WILL FAIL on the first push to main.** It runs the real judge and fails if weighted kappa < 0.4 (`MIN_WEIGHTED_KAPPA`); the accepted result is about 0.05. That is the gate doing its job, not a bug. It also refuses a verdict when judge coverage < 0.9 (the free-tier quota cut our own runs to 60-80%), so it can fail for that reason too. The lint and test jobs need no API key. Options: leave it red, lower the threshold (honest only if you say why), or make the job non-blocking. That is your call.

Steps for you:
1. `git init`, commit, create a GitHub repo, push to `main`. Add repo secret `GEMINI_API_KEY` (only the eval-gate job uses it).
2. In Render: New > Blueprint from the repo (`render.yaml`). Set `GEMINI_API_KEY`; Render generates `EVALFORGE_API_KEY` (read it in the dashboard, send it as `X-API-Key`).
3. `render.yaml` uses `plan: starter`. The free 512 MB tier will very likely run out of memory (torch + embedding model).
4. Write the live URL here: ______
5. Test: `curl -X POST <url>/evaluate -H "X-API-Key: <key>" -H "content-type: application/json" -d '{"input": "...", "output": "..."}'` and check `judge` is non-null.
- Port 8000 on this machine was already taken by another service, so local container tests used 8765.
