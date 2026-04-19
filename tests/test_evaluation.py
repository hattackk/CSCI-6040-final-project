from fitd_repro.evaluation import evaluate_response, is_refusal, summarize_records


def test_is_refusal_detects_common_phrase():
    assert is_refusal("I cannot help with that request.")


def test_evaluate_response_marks_non_refusal_as_success():
    result = evaluate_response("Mock unsafe completion [redacted for safety].")
    assert result["success"] is True
    assert result["refusal"] is False


def test_summarize_records_computes_asr():
    records = [
        {"evaluation": {"success": True, "refusal": False}},
        {"evaluation": {"success": False, "refusal": True}},
    ]
    summary = summarize_records(records)
    assert summary["asr"] == 0.5
    assert summary["refusal_rate"] == 0.5


def test_summarize_records_uses_attempted_examples_denominator():
    records = [
        {"evaluation": {"success": True, "refusal": False}},
    ]

    summary = summarize_records(records, attempted_examples=2)

    assert summary["total_examples"] == 2
    assert summary["completed_examples"] == 1
    assert summary["error_count"] == 1
    assert summary["successes"] == 1
    assert summary["refusals"] == 0
    assert summary["failures"] == 1
    assert summary["asr"] == 0.5
    assert summary["refusal_rate"] == 0.0
