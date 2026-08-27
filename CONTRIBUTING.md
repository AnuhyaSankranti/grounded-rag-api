# Contributing

1. Create a feature branch from `main`.
2. Install development dependencies with `pip install -e ".[dev]"`.
3. Add tests and at least one eval case for behavior changes.
4. Run `ruff check .`, `pytest`, and `mini-sia-eval --local --fail-under 0.70`.
5. Keep commits focused and describe measurable retrieval or latency changes.

Do not commit API keys, uploaded user documents, or generated database files.

