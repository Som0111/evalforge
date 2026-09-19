"""FastAPI service: /evaluate, /evaluate/batch, /report, /health (Phase 4)."""
import json
import logging
import os
import secrets

from fastapi import Depends, FastAPI, HTTPException, Security
from fastapi.responses import RedirectResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

from evalforge.config import (
    EMBEDDING_MODEL,
    GOLDEN_DATASET_PATH,
    JUDGE_MODEL,
    REPORTS_DIR,
)
from evalforge.dataset import GoldenDataset
from evalforge.scorers import score_all

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


app = FastAPI(title="EvalForge", description="Rule-based, embedding and LLM-judge scoring of LLM outputs.")


class EvaluateRequest(BaseModel):
    input: str
    output: str
    reference: str | None = None


class EvaluateResponse(BaseModel):
    scores: dict[str, float]
    judge: dict[str, int] | None
    judge_prompt_version: str
    judge_model: str


class BatchItem(BaseModel):
    input: str
    output: str


class BatchRequest(BaseModel):
    items: list[BatchItem] = Field(min_length=1, max_length=MAX_BATCH)


class BatchResponse(BaseModel):
    results: list[EvaluateResponse]
    total: int
    failed: int  # items where the judge was attempted but returned no scores


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
    response = EvaluateResponse(
        scores=score_all(output, input, reference),
        judge=judge,
        judge_prompt_version=API_PROMPT_VERSION,
        judge_model=JUDGE_MODEL,
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


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "dataset_size": _dataset_size(),
        "embedding_model": EMBEDDING_MODEL,
        "judge_model": JUDGE_MODEL,
    }


@app.get("/report")
def report() -> dict:
    """The most recently written eval report (backups excluded)."""
    reports = [p for p in REPORTS_DIR.glob("eval_report*.json") if ".backup." not in p.name]
    if not reports:
        return {"error": "no report found"}
    try:
        return json.loads(max(reports, key=lambda p: p.stat().st_mtime).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"error": "report could not be read"}


@app.post("/evaluate", dependencies=[Depends(require_api_key)])
def evaluate(req: EvaluateRequest) -> EvaluateResponse:
    return _evaluate(req.input, req.output, req.reference)[0]


@app.post("/evaluate/batch", dependencies=[Depends(require_api_key)])
def evaluate_batch(req: BatchRequest) -> BatchResponse:
    # Sequential: each judge call is a network round trip, so a full batch of 50 can take a while.
    outcomes = [_evaluate(item.input, item.output) for item in req.items]
    return BatchResponse(
        results=[r for r, _ in outcomes],
        total=len(outcomes),
        failed=sum(failed for _, failed in outcomes),
    )
