# Architecture and design decisions

## Request path

1. Ingestion validates a file, extracts pages, creates overlapping chunks, and batches
   embedding requests.
2. The store persists metadata, text, and embeddings atomically while maintaining an
   FTS5 index.
3. A question is embedded once. Semantic and lexical candidates are independently
   scored and normalized, then combined with a configurable weight.
4. The answer provider receives only the top chunks, with explicit source numbers and
   an instruction boundary that treats documents as untrusted data.
5. Invalid citation numbers are removed and the API returns source metadata separately.

## Why no orchestration framework?

The core workflow is deliberately implemented with small interfaces. This makes the
retrieval math, error boundaries, provider calls, and tests visible in an interview.
LangChain or LlamaIndex can be added later without changing the API contract.

## Scaling path

| Current choice | Scale-out replacement | Trigger |
| --- | --- | --- |
| SQLite + JSON vectors | PostgreSQL/pgvector or managed vector DB | Corpus no longer fits one node |
| In-process ingestion | Queue + idempotent workers | Large uploads or ingestion spikes |
| Local file parsing | Object storage + malware scanning | Multi-instance/public deployment |
| Single API service | Autoscaled stateless service | Sustained concurrency |
| Process metrics | Prometheus/OpenTelemetry collector | Multi-instance deployment |

## Reliability

- Content-derived document and chunk IDs make re-ingestion idempotent.
- SQLite writes are transactional and use WAL mode.
- Provider clients use bounded SDK retries and timeouts.
- Request IDs appear in response headers and structured logs.
- Quality and load tests are separate: evals guard correctness, Locust measures capacity.

## Security gaps before production

Add authentication, tenant IDs on every storage query, authorization checks, per-user
rate limits, encryption, retention controls, audit logs, upload scanning, and a durable
object store. Prompt-injection resistance is defense-in-depth, not a security boundary.

