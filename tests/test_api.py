import json

import pytest
from fastapi.testclient import TestClient

from evalforge import api

DIMS = ("faithfulness", "relevance", "coherence", "conciseness")
GOOD = dict.fromkeys(DIMS, 3)
BODY = {"input": "Nine tests failed in dicttoolz merge.", "output": "The merge function likely drops keys."}


API_KEY = "test-api-key"


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("EVALFORGE_API_KEY", API_KEY)
    return TestClient(api.app, headers={"X-API-Key": API_KEY})


@pytest.fixture
def anon(monkeypatch):
    monkeypatch.setenv("EVALFORGE_API_KEY", API_KEY)
    return TestClient(api.app)


@pytest.fixture
def judge_ok(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-real")
    monkeypatch.setattr(api, "_run_judge", lambda i, o: GOOD)


@pytest.fixture
def no_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)


def test_health_schema(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert isinstance(body["dataset_size"], int)
    assert body["embedding_model"] and body["judge_model"]


def test_root_redirects_to_docs(client):
    r = client.get("/", follow_redirects=False)
    assert r.status_code in (302, 307) and r.headers["location"] == "/docs"


def test_report_with_no_file_returns_error_dict_not_500(client, monkeypatch, tmp_path):
    monkeypatch.setattr(api, "REPORTS_DIR", tmp_path)
    r = client.get("/report")
    assert r.status_code == 200 and r.json() == {"error": "no report found"}


def test_report_returns_latest_and_ignores_backups(client, monkeypatch, tmp_path):
    monkeypatch.setattr(api, "REPORTS_DIR", tmp_path)
    (tmp_path / "eval_report.json").write_text(json.dumps({"which": "old"}))
    (tmp_path / "eval_report_v2.json").write_text(json.dumps({"which": "new"}))
    (tmp_path / "eval_report_v1.backup.json").write_text(json.dumps({"which": "backup"}))
    import os
    os.utime(tmp_path / "eval_report.json", (1, 1))
    assert client.get("/report").json() == {"which": "new"}


def test_corrupt_report_returns_error_not_500(client, monkeypatch, tmp_path):
    monkeypatch.setattr(api, "REPORTS_DIR", tmp_path)
    (tmp_path / "eval_report.json").write_text("{not json")
    assert "error" in client.get("/report").json()


def test_evaluate_valid_input(client, judge_ok):
    r = client.post("/evaluate", json={**BODY, "reference": BODY["output"]})
    assert r.status_code == 200
    body = r.json()
    assert {"length", "keyword_overlap", "format"} <= set(body["rule_based"])
    assert {"relevance", "semantic_sim"} <= set(body["semantic"])
    assert all(0.0 <= v <= 1.0 for v in {**body["rule_based"], **body["semantic"]}.values())
    assert body["judge"] == GOOD
    assert body["metadata"]["judge_prompt_version"] == "v2"
    assert body["metadata"]["judge_failed"] is False


def test_evaluate_without_reference_omits_semantic_sim(client, judge_ok):
    assert "semantic_sim" not in client.post("/evaluate", json=BODY).json()["semantic"]


def test_evaluate_missing_output_is_422(client):
    assert client.post("/evaluate", json={"input": "x"}).status_code == 422


def test_evaluate_without_api_key_returns_null_judge_and_scores(client, no_key):
    r = client.post("/evaluate", json=BODY)
    assert r.status_code == 200
    assert r.json()["judge"] is None
    assert r.json()["metadata"]["judge_failed"] is False  # judge was skipped, not attempted
    assert r.json()["rule_based"]["length"] >= 0


def test_judge_failure_returns_scores_with_null_judge(client, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-real")
    monkeypatch.setattr(api, "_run_judge", lambda i, o: None)
    r = client.post("/evaluate", json=BODY)
    assert r.status_code == 200 and r.json()["judge"] is None
    assert r.json()["metadata"]["judge_failed"] is True


def test_batch_of_three(client, judge_ok):
    r = client.post("/evaluate/batch", json={"items": [BODY] * 3})
    assert r.status_code == 200
    body = r.json()
    assert len(body["results"]) == body["total"] == 3 and body["failed"] == 0


def test_batch_over_limit_and_empty_are_422(client):
    assert client.post("/evaluate/batch", json={"items": [BODY] * 51}).status_code == 422
    assert client.post("/evaluate/batch", json={"items": []}).status_code == 422


def test_batch_empty_output_scores_zero_and_skips_judge(client, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-real")
    calls = []
    monkeypatch.setattr(api, "_run_judge", lambda i, o: calls.append(o) or GOOD)
    items = [BODY, {"input": "some input", "output": ""}]
    body = client.post("/evaluate/batch", json={"items": items}).json()
    empty = body["results"][1]
    assert empty["judge"] is None
    assert empty["rule_based"]["length"] == empty["rule_based"]["format"] == 0.0
    assert empty["semantic"]["relevance"] == 0.0
    assert calls == [BODY["output"]]


def test_batch_counts_judge_failures_but_keeps_rule_scores(client, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-real")
    monkeypatch.setattr(api, "_run_judge", lambda i, o: None if "bad" in o else GOOD)
    items = [BODY, {"input": "in", "output": "bad output here"}]
    body = client.post("/evaluate/batch", json={"items": items}).json()
    assert body["failed"] == 1 and body["total"] == 2
    assert body["results"][1]["judge"] is None and "length" in body["results"][1]["rule_based"]


@pytest.mark.parametrize("path,body", [("/evaluate", BODY), ("/evaluate/batch", {"items": [BODY]})])
def test_protected_endpoints_reject_missing_and_wrong_key(anon, path, body):
    assert anon.post(path, json=body).status_code == 401
    assert anon.post(path, json=body, headers={"X-API-Key": "wrong"}).status_code == 401
    assert anon.post(path, json={}).status_code == 401  # auth is checked before validation


def test_correct_key_is_accepted(anon, no_key):
    assert anon.post("/evaluate", json=BODY, headers={"X-API-Key": API_KEY}).status_code == 200


def test_server_without_configured_key_fails_closed(monkeypatch):
    monkeypatch.delenv("EVALFORGE_API_KEY", raising=False)
    r = TestClient(api.app).post("/evaluate", json=BODY, headers={"X-API-Key": "anything"})
    assert r.status_code == 503


def test_health_and_report_stay_open(anon):
    assert anon.get("/health").status_code == 200
    assert anon.get("/report").status_code == 200


def test_demo_page_serves_html_without_auth(anon):
    r = anon.get("/demo")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert "/evaluate" in r.text


def test_openapi_schema_generates(client):
    r = client.get("/openapi.json")
    assert r.status_code == 200
    schema = r.json()
    assert {"/health", "/report", "/evaluate", "/evaluate/batch"} <= set(schema["paths"])
