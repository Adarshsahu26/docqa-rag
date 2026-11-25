"""The HTTP API, exercised in-process with FastAPI's TestClient (no network, no real model)."""
import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from tests.helpers import (
    OIL_DOC,
    PUMP_DOC,
    SAMPLE_DOCS,
    BrokenLLM,
    FakeLLM,
    make_docx,
    make_pdf,
    make_pipeline,
    make_settings,
)

OIL_QUESTION = "How often is the transformer oil sampled?"


def make_client(llm=None, **overrides) -> TestClient:
    """A client for a fresh app. Use it as a context manager so start-up (the lifespan) runs."""
    settings = make_settings(**overrides)
    return TestClient(create_app(settings=settings, pipeline=make_pipeline(llm, settings)))


def upload(client: TestClient, name: str, data: bytes):
    return client.post("/documents", files={"file": (name, data, "application/octet-stream")})


@pytest.fixture()
def client():
    with make_client() as c:
        yield c


# ---- health and docs ---------------------------------------------------------------------------
def test_health_reports_what_the_service_is_running_with(client):
    body = client.get("/health").json()
    assert body == {"status": "ok", "embedder": "hashing-4096", "llm": "fake-llm", "documents": 0, "chunks": 0}
    upload(client, "oil.txt", OIL_DOC)
    after = client.get("/health").json()
    assert (after["documents"], after["chunks"]) == (1, 1)


def test_the_root_redirects_to_the_interactive_docs(client):
    response = client.get("/", follow_redirects=False)
    assert response.status_code in (302, 307)
    assert response.headers["location"] == "/docs"


def test_every_response_carries_its_processing_time(client):
    assert float(client.get("/health").headers["x-process-time-ms"]) >= 0


# ---- uploading ---------------------------------------------------------------------------------
def test_uploading_a_text_file_creates_a_document(client):
    response = upload(client, "oil.txt", OIL_DOC)
    assert response.status_code == 201
    body = response.json()
    assert body["filename"] == "oil.txt"
    assert len(body["doc_id"]) == 12
    assert body["num_chunks"] >= 1


def test_uploading_the_same_bytes_again_returns_the_existing_document_with_200(client):
    first = upload(client, "oil.txt", OIL_DOC)
    second = upload(client, "copy.txt", OIL_DOC)
    assert second.status_code == 200
    assert second.json() == first.json()
    assert len(client.get("/documents").json()) == 1


def test_real_pdf_and_docx_files_can_be_uploaded_and_questioned(client):
    oil_pdf = make_pdf(["Valve notes.", "The transformer oil is sampled every 12 months."])
    assert upload(client, "oil.pdf", oil_pdf).status_code == 201
    assert upload(client, "pump.docx", make_docx(["Alarm P-101 means low suction pressure."])).status_code == 201

    source = client.post("/ask", json={"question": OIL_QUESTION}).json()["sources"][0]
    assert (source["filename"], source["page"]) == ("oil.pdf", 2)
    pump_answer = client.post("/ask", json={"question": "What does alarm P-101 mean?"}).json()
    assert pump_answer["sources"][0]["filename"] == "pump.docx"


@pytest.mark.parametrize("raw_name", ["..\\..\\evil.txt", "/etc/secrets/evil.txt", "C:\\Users\\me\\evil.txt"])
def test_only_the_file_name_is_kept_never_a_path(client, raw_name):
    assert upload(client, raw_name, OIL_DOC).json()["filename"] == "evil.txt"


def test_unsupported_file_types_get_415_and_a_hint(client):
    response = upload(client, "tool.exe", b"MZ")
    assert response.status_code == 415
    assert ".pdf" in response.json()["detail"]


@pytest.mark.parametrize(
    ("name", "data", "expected_text"),
    [
        ("broken.pdf", b"not a pdf", "PDF"),
        ("broken.docx", b"not a zip", "DOCX"),
        ("empty.txt", b"", "empty"),
        ("blank.txt", b"  \n \n", "readable text"),
    ],
)
def test_unreadable_files_get_422_with_a_reason(client, name, data, expected_text):
    response = upload(client, name, data)
    assert response.status_code == 422
    assert expected_text in response.json()["detail"]


def test_a_request_without_a_file_gets_422(client):
    assert client.post("/documents").status_code == 422


def test_files_over_the_limit_get_413():
    limit = 1024 * 1024
    with make_client(max_upload_mb=1) as client:
        assert upload(client, "ok.txt", b"a" * limit).status_code == 201  # exactly at the limit
        assert upload(client, "big.txt", b"b" * (limit + 1)).status_code == 413  # caught while reading
        assert upload(client, "huge.txt", b"c" * (limit * 3)).status_code == 413  # refused from Content-Length
        assert len(client.get("/documents").json()) == 1


# ---- listing and deleting ----------------------------------------------------------------------
def test_documents_can_be_listed_and_deleted(client):
    oil = upload(client, "oil.txt", OIL_DOC).json()
    upload(client, "pump.txt", PUMP_DOC)
    assert sorted(d["filename"] for d in client.get("/documents").json()) == ["oil.txt", "pump.txt"]

    assert client.delete(f"/documents/{oil['doc_id']}").status_code == 204
    assert [d["filename"] for d in client.get("/documents").json()] == ["pump.txt"]


def test_deleting_an_unknown_document_gets_404(client):
    response = client.delete("/documents/does-not-exist")
    assert response.status_code == 404
    assert "does-not-exist" in response.json()["detail"]


# ---- asking ------------------------------------------------------------------------------------
def test_asking_returns_an_answer_with_its_sources(client):
    upload(client, "oil.txt", OIL_DOC)
    response = client.post("/ask", json={"question": OIL_QUESTION})
    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Fake answer [1]."
    assert body["grounded"] is True
    assert body["answered_by"] == "fake-llm"
    source = body["sources"][0]
    assert source["filename"] == "oil.txt"
    assert "12 months" in source["text"]
    assert 0 < source["score"] <= 1
    assert set(source) == {"doc_id", "filename", "chunk_id", "page", "score", "text"}


def test_an_unanswerable_question_is_reported_as_not_grounded(client):
    upload(client, "oil.txt", OIL_DOC)
    body = client.post("/ask", json={"question": "What is the capital of Australia?"}).json()
    assert body["grounded"] is False
    assert body["sources"] == []
    assert body["answered_by"] is None


def test_asking_before_uploading_anything_is_not_an_error(client):
    response = client.post("/ask", json={"question": OIL_QUESTION})
    assert response.status_code == 200
    assert response.json()["grounded"] is False


def test_a_question_can_be_limited_to_one_document(client):
    upload(client, "oil.txt", OIL_DOC)
    pump = upload(client, "pump.txt", PUMP_DOC).json()
    body = client.post("/ask", json={"question": OIL_QUESTION, "doc_id": pump["doc_id"]}).json()
    assert body["grounded"] is False


def test_asking_about_an_unknown_document_gets_404(client):
    upload(client, "oil.txt", OIL_DOC)
    assert client.post("/ask", json={"question": OIL_QUESTION, "doc_id": "nope"}).status_code == 404


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"question": "hi"},  # shorter than 3 characters
        {"question": "x" * 2001},
        {"question": OIL_QUESTION, "top_k": 0},
        {"question": OIL_QUESTION, "top_k": 21},
        {"question": OIL_QUESTION, "top_k": "many"},
    ],
)
def test_invalid_questions_get_422(client, payload):
    assert client.post("/ask", json=payload).status_code == 422


def test_model_failures_become_a_502_that_leaks_nothing():
    with make_client(BrokenLLM()) as client:
        upload(client, "oil.txt", OIL_DOC)
        response = client.post("/ask", json={"question": OIL_QUESTION})
    assert response.status_code == 502
    assert "sk-123456" not in response.text
    assert "language model" in response.json()["detail"]


# ---- search ------------------------------------------------------------------------------------
def test_search_shows_the_raw_ranking_without_calling_the_model():
    llm = FakeLLM()
    with make_client(llm) as client:
        upload(client, "oil.txt", OIL_DOC)
        upload(client, "pump.txt", PUMP_DOC)
        results = client.post("/search", json={"query": "oil sampling", "top_k": 5}).json()["results"]
    assert results[0]["filename"] == "oil.txt"
    assert [r["score"] for r in results] == sorted((r["score"] for r in results), reverse=True)
    assert llm.calls == []


def test_the_bundled_sample_documents_answer_the_demo_questions(client):
    for path in sorted(SAMPLE_DOCS.glob("*.txt")):
        assert upload(client, path.name, path.read_bytes()).status_code == 201
    expected = {
        "What does alarm code P-101 mean?": "pump_station_alarm_codes.txt",
        "What is the contracted peak demand limit?": "energy_management_guidelines.txt",
        "How long are alarm and event logs retained?": "remote_monitoring_data_policy.txt",
    }
    for question, filename in expected.items():
        assert client.post("/ask", json={"question": question}).json()["sources"][0]["filename"] == filename


# ---- authentication ----------------------------------------------------------------------------
PROTECTED = [
    ("get", "/documents", {}),
    ("post", "/ask", {"json": {"question": OIL_QUESTION}}),
    ("post", "/search", {"json": {"query": "oil"}}),
    ("delete", "/documents/abc", {}),
]


@pytest.mark.parametrize(("method", "path", "kwargs"), PROTECTED)
def test_protected_routes_need_the_api_key_when_one_is_configured(method, path, kwargs):
    with make_client(api_key="s3cret") as client:
        assert getattr(client, method)(path, **kwargs).status_code == 401
        wrong = getattr(client, method)(path, headers={"X-API-Key": "wrong"}, **kwargs)
        assert wrong.status_code == 401
        right = getattr(client, method)(path, headers={"X-API-Key": "s3cret"}, **kwargs)
        assert right.status_code != 401


def test_uploading_needs_the_api_key_too():
    with make_client(api_key="s3cret") as client:
        assert upload(client, "oil.txt", OIL_DOC).status_code == 401
        response = client.post(
            "/documents", files={"file": ("oil.txt", OIL_DOC)}, headers={"X-API-Key": "s3cret"}
        )
        assert response.status_code == 201


def test_health_and_docs_stay_open_so_monitors_and_browsers_still_work():
    with make_client(api_key="s3cret") as client:
        assert client.get("/health").status_code == 200
        assert client.get("/docs").status_code == 200
        assert client.get("/openapi.json").status_code == 200


def test_without_a_configured_key_no_header_is_needed(client):
    assert client.get("/documents").status_code == 200
