"""Untrusted model output: parsing, validation, and task grading."""
import math

import pytest

from core import config, tasks
from core.validation import (ValidationError, parse_json_object, validate_answer,
                             validate_decision)


# ---------------------------------------------------------------- parsing
def test_parse_plain_fenced_think_and_prose_wrapped_json():
    assert parse_json_object('{"a": 1}') == {"a": 1}
    assert parse_json_object('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json_object('<think>hmm {"x":2}</think>{"a": 1}') == {"a": 1}
    assert parse_json_object('Sure! Here you go: {"a": 1} hope that helps') == {"a": 1}


@pytest.mark.parametrize("text,code", [
    ("", "empty"), ("   ", "empty"), ("<think>only thoughts</think>", "empty"),
    ("no json here", "malformed_json"), ("{bad json}", "malformed_json"),
    ("[1, 2, 3]", "not_object"), ('"just a string"', "not_object"), ("42", "not_object"),
    (None, "not_text"), (123, "not_text"),
])
def test_parse_rejects_unusable_output(text, code):
    with pytest.raises(ValidationError) as e:
        parse_json_object(text)
    assert e.value.code == code


def test_parse_rejects_oversized_response(monkeypatch):
    monkeypatch.setattr(config, "MAX_RESPONSE_CHARS", 100)
    with pytest.raises(ValidationError) as e:
        parse_json_object('{"a": "' + "x" * 200 + '"}')
    assert e.value.code == "too_large"


# --------------------------------------------------------------- decisions
def test_valid_decisions():
    d = validate_decision({"action": "bid", "confidence": 0.7, "reason": "ok"})
    assert (d.action, d.confidence) == ("bid", 0.7)
    assert validate_decision({"action": " SKIP "}).action == "skip"
    d = validate_decision({"action": "lend", "amount": "12"})
    assert d.amount == 12
    assert validate_decision({"action": "borrow", "amount": 8.0}).amount == 8
    assert validate_decision({"action": "bid", "confidence": 1}).confidence == 1.0


@pytest.mark.parametrize("raw,code", [
    (None, "not_object"), ([], "not_object"), ("bid", "not_object"),
    ({}, "bad_action"), ({"action": "hack"}, "bad_action"), ({"action": 5}, "bad_action"),
    ({"action": "bid"}, "missing_confidence"),
    ({"action": "bid", "confidence": float("nan")}, "bad_confidence"),
    ({"action": "bid", "confidence": float("inf")}, "bad_confidence"),
    ({"action": "bid", "confidence": -0.1}, "bad_confidence"),
    ({"action": "bid", "confidence": 1.5}, "bad_confidence"),
    ({"action": "bid", "confidence": "0.9"}, "bad_confidence"),
    ({"action": "bid", "confidence": True}, "bad_confidence"),
    ({"action": "lend"}, "missing_amount"),
    ({"action": "lend", "amount": 0}, "bad_amount"),
    ({"action": "lend", "amount": -5}, "bad_amount"),
    ({"action": "lend", "amount": 2.5}, "bad_amount"),
    ({"action": "lend", "amount": float("nan")}, "bad_amount"),
    ({"action": "lend", "amount": float("inf")}, "bad_amount"),
    ({"action": "lend", "amount": "ten"}, "bad_amount"),
    ({"action": "lend", "amount": "1e308"}, "bad_amount"),
    ({"action": "borrow", "amount": 10**12}, "bad_amount"),
    ({"action": "borrow", "amount": [5]}, "bad_amount"),
    ({"action": "borrow", "amount": True}, "bad_amount"),
])
def test_invalid_decisions_rejected(raw, code):
    with pytest.raises(ValidationError) as e:
        validate_decision(raw)
    assert e.value.code == code


def test_reason_truncated_and_non_string_reason_ignored():
    assert len(validate_decision({"action": "skip", "reason": "x" * 5000}).reason) == 200
    assert validate_decision({"action": "skip", "reason": {"a": 1}}).reason == ""


def test_garbage_amount_on_bid_is_ignored_not_fatal():
    assert validate_decision({"action": "bid", "confidence": 0.5, "amount": "lots"}).amount == 0


# ----------------------------------------------------------------- answers
def test_answers_accept_text_and_numbers():
    assert validate_answer({"answer": " 5 "}) == ("5", None)
    assert validate_answer({"answer": 102, "confidence": 0.8}) == ("102", 0.8)
    assert validate_answer({"answer": 1.5})[0] == "1.5"


@pytest.mark.parametrize("raw", [None, [], "x", {}, {"answer": None}, {"answer": ""}, {"answer": "  "},
                                 {"answer": []}, {"answer": {"a": 1}}, {"answer": True},
                                 {"answer": float("nan")}, {"answer": "x" * 500}])
def test_unusable_answers_rejected(raw):
    with pytest.raises(ValidationError):
        validate_answer(raw)


def test_bad_confidence_does_not_invalidate_a_good_answer():
    assert validate_answer({"answer": "5", "confidence": float("nan")}) == ("5", None)


# ------------------------------------------------------------------ grading
def T(answer):
    return tasks.Task(id=1, prompt="q", answer=answer, difficulty="easy", bounty=8)


@pytest.mark.parametrize("expected,submitted", [
    ("5", "5"), ("5", 5), ("5", 5.0), ("5", "5.0"), ("5", " 5. "), ("0.05", "$0.05"), ("0.05", ".05"),
    ("50%", "50"), ("1,000", "1000"), ("5:45pm", "5:45 PM"), ("saturn", "Saturn"), ("3x^2", "3x²"),
    ("3x^2", "3*x^2"), ("1.5", "1.50"), ("pacific", '"Pacific"'),
])
def test_grade_accepts_equivalent_answers(expected, submitted):
    assert tasks.grade(T(expected), submitted) is True


@pytest.mark.parametrize("submitted", ["6", "", "  ", None, [], {}, ["5"], {"answer": 5}, True, False,
                                       float("nan"), "five", "5 cents"])
def test_grade_never_crashes_and_marks_junk_wrong(submitted):
    assert tasks.grade(T("5"), submitted) is False


def test_ungraded_task_returns_none_not_false():
    assert tasks.grade(T(None), "anything") is None
    assert tasks.grade(T(None), None) is None


def test_every_bank_answer_grades_itself_correct_and_heldout_split_is_stable():
    for prompt, answer, difficulty in tasks._TASK_BANK:
        t = tasks.Task(id=0, prompt=prompt, answer=answer, difficulty=difficulty, bounty=1)
        assert tasks.grade(t, answer) is True
    held = [p for p, *_ in tasks._TASK_BANK if tasks.is_heldout(p)]
    assert held == [p for p, *_ in tasks._TASK_BANK if tasks.is_heldout(p)]
    assert 0 < len(held) < len(tasks._TASK_BANK)


def test_weighted_task_generation_respects_weights():
    import random
    rng = random.Random(1)
    only_hard = {tasks.generate_task(rng, (0, 0, 1)).difficulty for _ in range(30)}
    assert only_hard == {"hard"}
