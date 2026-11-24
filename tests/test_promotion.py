"""The promotion gate decides whether a retrained model replaces the served one."""
from app.services.promotion import should_promote


def m(qwk, g34=0.4):
    return {"qwk": qwk, "grade34_recall": g34}


def test_better_model_is_promoted():
    ok, reason = should_promote(m(0.62), m(0.59))
    assert ok and "improved" in reason


def test_small_drop_within_tolerance_is_promoted():
    ok, reason = should_promote(m(0.585), m(0.59), qwk_tolerance=0.01)
    assert ok and "within tolerance" in reason


def test_qwk_drop_beyond_tolerance_is_rejected():
    ok, reason = should_promote(m(0.55), m(0.59), qwk_tolerance=0.01)
    assert not ok and "QWK fell by 0.040" in reason


def test_recall_drop_beyond_tolerance_is_rejected_even_if_qwk_improves():
    ok, reason = should_promote(m(0.62, g34=0.30), m(0.59, g34=0.40), recall_tolerance=0.05)
    assert not ok and "Grade 3+4 recall fell by 0.100" in reason


def test_missing_recall_falls_back_to_qwk_only():
    ok, _ = should_promote(m(0.60, g34=None), m(0.59, g34=0.4))
    assert ok
