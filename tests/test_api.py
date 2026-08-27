from fastapi.testclient import TestClient


def test_health_reports_local_providers(client: TestClient) -> None:
    response = client.get("/v1/health")

    assert response.status_code == 200
    assert response.json()["providers"] == {"llm": "extractive", "embeddings": "hash"}
    assert response.headers["x-request-id"]


def test_ingest_and_ask_returns_grounded_sources(client: TestClient) -> None:
    content = (
        "Playback anomaly detection runs in scheduled Glue jobs. "
        "The jobs compare metrics against baselines and create a ticket."
    )
    ingest = client.post(
        "/v1/documents",
        files={"file": ("analytics.md", content, "text/markdown")},
    )

    assert ingest.status_code == 201
    document_id = ingest.json()["document_id"]
    response = client.post(
        "/v1/ask",
        json={"question": "How does anomaly detection work?", "document_ids": [document_id]},
    )

    assert response.status_code == 200
    payload = response.json()
    assert "Glue" in payload["answer"]
    assert "[1]" in payload["answer"]
    assert payload["sources"][0]["filename"] == "analytics.md"
    assert payload["latency_ms"] >= 0


def test_rejects_unsupported_file(client: TestClient) -> None:
    response = client.post(
        "/v1/documents",
        files={"file": ("image.png", b"not an image", "image/png")},
    )

    assert response.status_code == 415


def test_question_validation(client: TestClient) -> None:
    response = client.post("/v1/ask", json={"question": "?"})
    assert response.status_code == 422

