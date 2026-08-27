from mini_sia.eval.metrics import answer_coverage, citation_validity
from mini_sia.models import AskResponse, Source


def test_answer_coverage_is_case_and_punctuation_insensitive() -> None:
    assert answer_coverage("AWS Glue compares BASELINES.", ["glue", "baselines"]) == 1.0


def test_citation_validity_rejects_out_of_range_numbers() -> None:
    response = AskResponse(
        answer="A grounded fact [1] and an invalid claim [3].",
        sources=[
            Source(
                citation=1,
                document_id="d1",
                filename="doc.md",
                chunk_id="c1",
                page=None,
                score=1.0,
                snippet="A grounded fact",
            )
        ],
        latency_ms=1.0,
    )
    assert citation_validity(response) == 0.5

