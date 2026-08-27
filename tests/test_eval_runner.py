import argparse
import asyncio

from mini_sia.eval.runner import run_eval


def test_local_eval_clears_quality_gate() -> None:
    args = argparse.Namespace(
        local=True,
        provider="openai",
        dataset=None,
        docs=None,
        output=None,
        fail_under=0.90,
    )

    report = asyncio.run(run_eval(args))

    assert report["case_count"] == 5
    assert report["quality_score"] >= 0.90
    assert report["metrics"]["citation_validity"] == 1.0

