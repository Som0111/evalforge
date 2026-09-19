"""Gemini-as-judge with prompt templates, retry, and a resumable batch runner (Phase 2)."""
import json
import logging
import os
import re
import time

from google import genai
from google.genai import errors, types

from evalforge.config import (
    JUDGE_MODEL,
    JUDGE_OUTPUTS_PATH,
    LEGACY_JUDGE_MODEL,
    versioned_path,
)

_api_key = os.environ.get("GEMINI_API_KEY")
if not _api_key:
    raise OSError("GEMINI_API_KEY is not set. Export it before importing evalforge.scorers.llm_judge.")
_client = genai.Client(api_key=_api_key)

log = logging.getLogger(__name__)

DIMENSIONS = ("faithfulness", "relevance", "coherence", "conciseness")
RETRY_DELAY_S = 2.0  # multiplied by the attempt number for transient API errors

# The 1-3 scale is deliberately identical to the human golden labels, so Cohen's Kappa
# in Phase 3 compares judge and human directly with no remapping.
# Literal braces in the JSON example are doubled because the template goes through str.format.
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
{{"faithfulness": 3, "relevance": 2, "coherence": 3, "conciseness": 2}}
"""

# v2: v1 scored 0.016 weighted kappa because the judge gave 3 to almost everything. This version uses the
# human rubric wording, states the 2-3 sentence ask as concrete sentence-count anchors (taken from the ask
# itself, not from the labelled data), and tells the judge to be strict.
JUDGE_PROMPT_V2 = """You are a strict evaluator of an AI-generated summary of a software test failure.

INPUT (the full prompt that was given to the AI, including its instructions):
{input}

OUTPUT (what the AI generated):
{output}

Grade the output on each dimension using ONLY the integer 1, 2, or 3 (1 = poor, 2 = acceptable, 3 = good).

- faithfulness: sticks to the evidence shown in the input. No invented causes or details. Saying the evidence is too thin counts as faithful.
  3 = every claim is supported by the input; 2 = goes beyond the evidence in places; 1 = invents facts.
- relevance: answers the prompt with a root-cause hypothesis that is specific to this failure.
  3 = specific to this failure; 2 = generic or only partly on target; 1 = off-topic.
- coherence: clear and well structured.
  3 = clear and well organised; 2 = readable but awkward or cluttered; 1 = hard to follow.
- conciseness: respects the sentence limit the input asks for.
  3 = within the requested length with no filler; 2 = slightly over (up to about twice the limit) or has preamble or formatting; 1 = far over the limit, several paragraphs, or headings and bullet lists.

Be strict. Give 3 only when you cannot name a concrete flaw, and when torn between two grades choose the lower one. Do not reward length or a confident tone.

Respond ONLY with valid JSON. No explanation. No markdown. Example:
{{"faithfulness": 3, "relevance": 2, "coherence": 3, "conciseness": 2}}
"""

PROMPTS = {"v1": JUDGE_PROMPT_V1, "v2": JUDGE_PROMPT_V2}


def _empty() -> dict:
    return dict.fromkeys(DIMENSIONS)


def _generate(prompt: str) -> str:
    """The single network call. Tests replace this."""
    response = _client.models.generate_content(
        model=JUDGE_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0, response_mime_type="application/json"),
    )
    return response.text or ""


def _parse(text: str) -> dict:
    """Parse a judge reply into {dimension: 1|2|3}. Raises ValueError on anything else."""
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())  # tolerate a fenced reply
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("reply is not a JSON object")  # noqa: TRY004 - judge() retries on ValueError
    scores = {}
    for dim in DIMENSIONS:
        v = data.get(dim)
        if isinstance(v, bool) or v not in (1, 2, 3):
            raise ValueError(f"{dim}={v!r} is not 1, 2, or 3")
        scores[dim] = int(v)
    return scores


def judge(input: str, output: str, prompt_template: str = JUDGE_PROMPT_V1, max_retries: int = 3) -> dict:
    """Score one output. Never raises: on exhausted retries or a quota error, returns all-None."""
    prompt = prompt_template.format(input=input, output=output)
    for attempt in range(1, max_retries + 2):  # first try + max_retries retries
        try:
            return _parse(_generate(prompt))
        except (json.JSONDecodeError, ValueError) as e:
            log.warning("judge attempt %d/%d: unparseable reply (%s)", attempt, max_retries + 1, e)
        except errors.APIError as e:
            if e.code == 429:  # quota/rate limit: retrying within seconds will not help
                log.warning("judge: quota exhausted (%s); giving up on this record", e.message)
                return _empty()
            log.warning("judge attempt %d/%d: API error %s", attempt, max_retries + 1, e.code)
            time.sleep(RETRY_DELAY_S * attempt)
        except Exception as e:  # noqa: BLE001 - spec: the judge never crashes the pipeline
            log.warning("judge attempt %d/%d: %s: %s", attempt, max_retries + 1, type(e).__name__, e)
            time.sleep(RETRY_DELAY_S * attempt)
    log.warning("judge: retries exhausted")
    return _empty()


def judge_output_path(prompt_version: str, model: str | None = None):
    return versioned_path(JUDGE_OUTPUTS_PATH, prompt_version, model or JUDGE_MODEL)


def run_judge_on_dataset(dataset: list[dict], prompt_version: str = "v1", out_path=None) -> list[dict]:
    """Judge every record, appending judge_label + judge_prompt_version, and write JSONL.

    Resumable: the output file is appended to as we go, and ids already present are reused,
    so a run cut short by the quota picks up where it stopped. All-None results are
    skipped (warned, not written) so the next run retries them. Each (prompt version, judge model) has its own file.
    """
    template = PROMPTS[prompt_version]
    model = JUDGE_MODEL  # read at call time so a caller can override the module attribute
    out_path = out_path or judge_output_path(prompt_version, model)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = {}
    if out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            done = {r["id"]: r for r in map(json.loads, filter(str.strip, f)) if r.get("judge_prompt_version") == prompt_version and r.get("judge_model", LEGACY_JUDGE_MODEL) == model}

    results = []
    with open(out_path, "a", encoding="utf-8") as f:
        for record in dataset:
            if record["id"] in done:
                results.append(done[record["id"]])
                continue
            label = judge(record["input"], record["output"], template)
            if all(v is None for v in label.values()):
                log.warning("skipping %s: judge returned no scores", record["id"])
                continue
            out = {**record, "judge_label": label, "judge_prompt_version": prompt_version, "judge_model": model}
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
            f.flush()
            results.append(out)
    return results
