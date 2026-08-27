# Live Event Operations Runbook

Before a Thursday Night Football event, the on-call engineer verifies ingestion lag,
Druid datasource freshness, dashboard health, and alert routing. A canary query must
return recent playback sessions before the event is declared ready.

If stream consumer lag rises, check partition hot spots, throttling, and recent
deployments. Scale consumers only after confirming downstream Druid capacity. Failed
events are retried with bounded exponential backoff and sent to a dead-letter path
after the retry budget is exhausted.

For a failed Glue anomaly job, inspect CloudWatch logs, input partitions, schema
changes, worker memory, and the job bookmark. Reprocess only the affected partition
and verify ticket deduplication before closing the incident.

