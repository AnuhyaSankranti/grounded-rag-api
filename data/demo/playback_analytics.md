# Playback Analytics Platform

The platform ingests near-real-time playback telemetry through partitioned streams.
Java consumers validate and aggregate events before writing optimized segments to
Apache Druid. Druid supports low-latency slice-and-dice queries during major live
events, while Athena is used for longer-retention historical analysis.

Playback anomaly detection runs as scheduled AWS Glue jobs. Each job compares key
playback metrics against metric-specific baselines. When a deviation crosses its
configured threshold, the automation opens a ticket for the owning engineering team
and includes the affected dimensions. This reduced manual investigation work by
approximately 60 percent.

The analytics API uses Redis as a read-through cache for frequently requested metric
metadata and computed responses. Cache hits avoid repeated downstream queries and
reduced backend API load by about 70 percent. CloudWatch alarms track consumer lag,
Glue failures, API errors, and end-to-end data freshness.

