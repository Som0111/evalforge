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
