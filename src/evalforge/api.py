"""FastAPI service: /evaluate, /evaluate/batch, /report, /health (Phase 4)."""
import json
import logging
import os
import secrets

from fastapi import Depends, FastAPI, HTTPException, Security
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

from evalforge.config import (
    EMBEDDING_MODEL,
    GOLDEN_DATASET_PATH,
    JUDGE_MODEL,
    REPORTS_DIR,
)
from evalforge.dataset import GoldenDataset
from evalforge.scorers import grouped_scores, score_all

log = logging.getLogger(__name__)

MAX_BATCH = 50
API_PROMPT_VERSION = "v2"  # the stricter prompt from Phase 3; see HUMAN_GUIDE.md for how well it agrees with humans

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_api_key(key: str | None = Security(_api_key_header)) -> None:
    """Guards the endpoints that can spend judge quota. Fails closed if no server key is configured."""
    expected = os.environ.get("EVALFORGE_API_KEY")
    if not expected:
        raise HTTPException(503, "EVALFORGE_API_KEY is not configured on the server")
    if not key or not secrets.compare_digest(key.encode(), expected.encode()):
        raise HTTPException(401, "missing or invalid X-API-Key")


app = FastAPI(
    title="EvalForge",
    description=(
        "Scores LLM output three ways (rule-based, embedding similarity, LLM-as-judge) "
        "and reports how well the judge agrees with human labels. "
        "`/evaluate` and `/evaluate/batch` require an `X-API-Key` header — set `EVALFORGE_API_KEY` "
        "on the server and send the same value as the header. Try `/docs` below, or `/demo` for a "
        "no-code form."
    ),
)

_EXAMPLE_INPUT = "Nine tests failed in dicttoolz merge."
_EXAMPLE_OUTPUT = "The merge function likely drops keys when input dicts overlap."


class EvaluateRequest(BaseModel):
    input: str = Field(description="The text given to the LLM (e.g. a prompt or source document).")
    output: str = Field(description="The LLM-generated text to score.")
    reference: str | None = Field(
        default=None,
        description="Optional known-good text. When given, adds a `semantic_sim` score (embedding similarity to `output`).",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [{"input": _EXAMPLE_INPUT, "output": _EXAMPLE_OUTPUT}],
        }
    }


class Metadata(BaseModel):
    judge_prompt_version: str = Field(description="Which judge prompt template was used.")
    judge_model: str = Field(description="Which LLM acted as judge.")
    judge_failed: bool = Field(
        description="True if the judge was attempted but returned no usable scores (e.g. quota, unparseable output). "
        "False both when the judge succeeded and when it was skipped (no `GEMINI_API_KEY` set)."
    )


class EvaluateResponse(BaseModel):
    rule_based: dict[str, float] = Field(description="length, keyword_overlap, format — each in [0.0, 1.0].")
    semantic: dict[str, float] = Field(
        description="relevance (output vs input), and semantic_sim (output vs reference) if a reference was given."
    )
    judge: dict[str, int] | None = Field(
        description="faithfulness, relevance, coherence, conciseness, each 1-3. "
        "`null` if the judge is disabled or failed — see `metadata.judge_failed`."
    )
    metadata: Metadata

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "rule_based": {"length": 0.92, "keyword_overlap": 0.6, "format": 1.0},
                    "semantic": {"relevance": 0.81},
                    "judge": {"faithfulness": 3, "relevance": 2, "coherence": 3, "conciseness": 2},
                    "metadata": {"judge_prompt_version": "v2", "judge_model": "gemini-3.6-flash", "judge_failed": False},
                }
            ]
        }
    }


class BatchItem(BaseModel):
    input: str = Field(description="The text given to the LLM.")
    output: str = Field(description="The LLM-generated text to score.")


class BatchRequest(BaseModel):
    items: list[BatchItem] = Field(min_length=1, max_length=MAX_BATCH, description=f"1 to {MAX_BATCH} items.")

    model_config = {
        "json_schema_extra": {
            "examples": [{"items": [{"input": _EXAMPLE_INPUT, "output": _EXAMPLE_OUTPUT}]}],
        }
    }


class BatchResponse(BaseModel):
    results: list[EvaluateResponse] = Field(description="One result per input item, same order.")
    total: int = Field(description="Number of items submitted.")
    failed: int = Field(description="Items where the judge was attempted but returned no scores.")


def _judge_enabled() -> bool:
    return bool(os.environ.get("GEMINI_API_KEY"))


def _run_judge(input: str, output: str) -> dict | None:
    """Judge scores, or None if the judge produced nothing. Only call when _judge_enabled()."""
    from evalforge.scorers import (
        llm_judge,  # import raises without GEMINI_API_KEY, so it is lazy
    )

    label = llm_judge.judge(input, output, llm_judge.PROMPTS[API_PROMPT_VERSION])
    return None if all(v is None for v in label.values()) else label


def _evaluate(input: str, output: str, reference: str | None = None) -> tuple[EvaluateResponse, bool]:
    """Returns (response, judge_failed). Empty text scores 0.0 and is never sent to the judge."""
    judge, failed = None, False
    if _judge_enabled() and input.strip() and output.strip():
        judge = _run_judge(input, output)
        failed = judge is None
    scores = grouped_scores(score_all(output, input, reference))
    response = EvaluateResponse(
        rule_based=scores["rule_based"],
        semantic=scores["semantic"],
        judge=judge,
        metadata=Metadata(
            judge_prompt_version=API_PROMPT_VERSION,
            judge_model=JUDGE_MODEL,
            judge_failed=failed,
        ),
    )
    return response, failed


def _dataset_size() -> int:
    try:
        return len(GoldenDataset.load(GOLDEN_DATASET_PATH))
    except (OSError, ValueError):  # missing file, or a golden file that fails validation
        return 0


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse("/docs")


@app.get(
    "/health",
    summary="Service status",
    description="No auth required. Reports whether the service is up and which models it's configured with.",
)
def health() -> dict:
    return {
        "status": "ok",
        "dataset_size": _dataset_size(),
        "embedding_model": EMBEDDING_MODEL,
        "judge_model": JUDGE_MODEL,
    }


@app.get(
    "/report",
    summary="Latest dataset-level evaluation report",
    description="No auth required. The most recently written `reports/eval_report*.json` "
    "(judge-vs-human calibration, bias checks, per-scorer averages), or an `{\"error\": ...}` dict if none exists yet.",
)
def report() -> dict:
    reports = [p for p in REPORTS_DIR.glob("eval_report*.json") if ".backup." not in p.name]
    if not reports:
        return {"error": "no report found"}
    try:
        return json.loads(max(reports, key=lambda p: p.stat().st_mtime).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"error": "report could not be read"}


@app.post(
    "/evaluate",
    dependencies=[Depends(require_api_key)],
    summary="Score one input/output pair",
    description="Requires `X-API-Key`. Runs rule-based and embedding scorers, plus the LLM judge if "
    "`GEMINI_API_KEY` is configured on the server.",
)
def evaluate(req: EvaluateRequest) -> EvaluateResponse:
    return _evaluate(req.input, req.output, req.reference)[0]


@app.post(
    "/evaluate/batch",
    dependencies=[Depends(require_api_key)],
    summary="Score up to 50 input/output pairs",
    description=f"Requires `X-API-Key`. Same scoring as `/evaluate`, run sequentially over up to {MAX_BATCH} items.",
)
def evaluate_batch(req: BatchRequest) -> BatchResponse:
    # Sequential: each judge call is a network round trip, so a full batch of 50 can take a while.
    outcomes = [_evaluate(item.input, item.output) for item in req.items]
    return BatchResponse(
        results=[r for r, _ in outcomes],
        total=len(outcomes),
        failed=sum(failed for _, failed in outcomes),
    )


_DEMO_HTML = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>EvalForge demo</title>
<style>
  body { font-family: system-ui, sans-serif; max-width: 640px; margin: 2rem auto; padding: 0 1rem; }
  textarea, input { width: 100%; box-sizing: border-box; margin-bottom: 0.75rem; font-family: inherit; }
  textarea { height: 5rem; }
  pre { background: #f4f4f4; padding: 1rem; overflow-x: auto; white-space: pre-wrap; }
  button { padding: 0.5rem 1rem; }
  .error { color: #b00020; }
</style>
</head>
<body>
<h1>EvalForge demo</h1>
<p>Calls <code>POST /evaluate</code> directly from your browser. Needs the server's <code>X-API-Key</code>.</p>
<form id="f">
  <label>X-API-Key<input name="apiKey" required></label>
  <label>Input (given to the LLM)<textarea name="input" required>Nine tests failed in dicttoolz merge.</textarea></label>
  <label>Output (what the LLM wrote)<textarea name="output" required>The merge function likely drops keys when input dicts overlap.</textarea></label>
  <button type="submit">Evaluate</button>
</form>
<pre id="out">Result will appear here.</pre>
<script>
document.getElementById("f").addEventListener("submit", async (e) => {
  e.preventDefault();
  const f = new FormData(e.target);
  const out = document.getElementById("out");
  out.textContent = "Scoring...";
  out.className = "";
  try {
    const r = await fetch("/evaluate", {
      method: "POST",
      headers: { "Content-Type": "application/json", "X-API-Key": f.get("apiKey") },
      body: JSON.stringify({ input: f.get("input"), output: f.get("output") }),
    });
    const body = await r.json();
    out.textContent = JSON.stringify(body, null, 2);
    if (!r.ok) out.className = "error";
  } catch (err) {
    out.textContent = "Request failed: " + err;
    out.className = "error";
  }
});
</script>
</body>
</html>"""


@app.get("/demo", include_in_schema=False, response_class=HTMLResponse)
def demo() -> str:
    return _DEMO_HTML
