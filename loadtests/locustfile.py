import random

from locust import HttpUser, between, task


QUESTIONS = [
    "How are playback anomalies detected?",
    "What should I check when consumer lag rises?",
    "Which service handles historical analysis?",
    "What is the feature rollout order?",
    "How does caching reduce backend load?",
]


class MiniSiaUser(HttpUser):
    wait_time = between(0.5, 2.0)

    @task(9)
    def ask_question(self) -> None:
        with self.client.post(
            "/v1/ask",
            json={"question": random.choice(QUESTIONS), "top_k": 4},
            name="POST /v1/ask",
            catch_response=True,
            timeout=60,
        ) as response:
            if response.status_code != 200:
                response.failure(f"unexpected status {response.status_code}")
                return
            payload = response.json()
            if "answer" not in payload or "sources" not in payload:
                response.failure("response contract missing answer or sources")

    @task(1)
    def health(self) -> None:
        self.client.get("/v1/health", name="GET /v1/health", timeout=5)

