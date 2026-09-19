import importlib
import json

import pytest
from google.genai import errors

from evalforge.scorers import llm_judge

GOOD = {"faithfulness": 3, "relevance": 2, "coherence": 3, "conciseness": 2}
NONE = dict.fromkeys(GOOD)


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(llm_judge, "RETRY_DELAY_S", 0)


def fake_generate(monkeypatch, replies):
    """Replace the network call with a scripted sequence of replies (or exceptions)."""
    calls = []
    it = iter(replies)

    def _generate(prompt):
        calls.append(prompt)
        r = next(it)
        if isinstance(r, Exception):
            raise r
        return r

    monkeypatch.setattr(llm_judge, "_generate", _generate)
    return calls


def test_prompt_template_formats_with_json_example_braces():
    prompt = llm_judge.JUDGE_PROMPT_V1.format(input="IN {x}", output="OUT")
    assert "IN {x}" in prompt and '{"faithfulness": 3' in prompt


def test_valid_json_is_parsed(monkeypatch):
    fake_generate(monkeypatch, [json.dumps(GOOD)])
    assert llm_judge.judge("in", "out") == GOOD


def test_fenced_json_is_tolerated(monkeypatch):
    fake_generate(monkeypatch, ["```json\n" + json.dumps(GOOD) + "\n```"])
    assert llm_judge.judge("in", "out") == GOOD


def test_invalid_json_twice_then_valid_retries(monkeypatch):
    calls = fake_generate(monkeypatch, ["not json", "{broken", json.dumps(GOOD)])
    assert llm_judge.judge("in", "out") == GOOD
    assert len(calls) == 3


def test_always_invalid_returns_all_none_after_retries(monkeypatch):
    calls = fake_generate(monkeypatch, ["nope"] * 10)
    assert llm_judge.judge("in", "out", max_retries=3) == NONE
    assert len(calls) == 4  # first try + 3 retries


@pytest.mark.parametrize("bad", [
    {"faithfulness": 4, "relevance": 2, "coherence": 3, "conciseness": 2},
    {"faithfulness": "3", "relevance": 2, "coherence": 3, "conciseness": 2},
    {"faithfulness": True, "relevance": 2, "coherence": 3, "conciseness": 2},
    {"faithfulness": 3, "relevance": 2, "coherence": 3},
    [1, 2, 3],
])
def test_out_of_range_or_malformed_scores_are_rejected(monkeypatch, bad):
    fake_generate(monkeypatch, [json.dumps(bad)] * 2)
    assert llm_judge.judge("in", "out", max_retries=1) == NONE


def test_transient_api_error_is_retried(monkeypatch):
    fake_generate(monkeypatch, [RuntimeError("network"), json.dumps(GOOD)])
    assert llm_judge.judge("in", "out") == GOOD


def test_quota_error_gives_up_immediately_without_raising(monkeypatch):
    quota = errors.ClientError(429, {"error": {"message": "quota", "status": "RESOURCE_EXHAUSTED"}})
    calls = fake_generate(monkeypatch, [quota] * 10)
    assert llm_judge.judge("in", "out") == NONE
    assert len(calls) == 1


def test_missing_api_key_raises_on_import(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(EnvironmentError):
        importlib.reload(llm_judge)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-real")
    importlib.reload(llm_judge)


def test_v2_prompt_formats_and_carries_the_rubric():
    prompt = llm_judge.JUDGE_PROMPT_V2.format(input="IN {x}", output="OUT")
    assert "IN {x}" in prompt and '{"faithfulness": 3' in prompt
    assert "sentence limit" in prompt
    assert llm_judge.PROMPTS == {"v1": llm_judge.JUDGE_PROMPT_V1, "v2": llm_judge.JUDGE_PROMPT_V2}
