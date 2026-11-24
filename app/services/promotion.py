"""Promotion gate: should a retrained model replace the one being served?

Both models are scored on the same held-out evaluation set. The candidate is
promoted unless it is meaningfully worse on either of the two numbers that
matter most for screening:

- QWK (agreement with the true grade, penalising big misses more), and
- Grade 3+4 recall (catching sight-threatening disease).

Small drops within the tolerances are accepted: on a few hundred images the
metrics move by a point or two from noise alone.
"""
from app.config import PROMOTION_QWK_TOLERANCE, PROMOTION_RECALL_TOLERANCE


def should_promote(candidate: dict, current: dict,
                   qwk_tolerance: float = PROMOTION_QWK_TOLERANCE,
                   recall_tolerance: float = PROMOTION_RECALL_TOLERANCE) -> tuple[bool, str]:
    """Return ``(promote, reason)`` from two metric dicts (keys ``qwk``, ``grade34_recall``)."""
    d_qwk = candidate["qwk"] - current["qwk"]
    if d_qwk < -qwk_tolerance:
        return False, (f"QWK fell by {-d_qwk:.3f} ({current['qwk']:.3f} → "
                       f"{candidate['qwk']:.3f}), more than the {qwk_tolerance:.3f} allowed.")

    c34, p34 = candidate.get("grade34_recall"), current.get("grade34_recall")
    if c34 is not None and p34 is not None and c34 - p34 < -recall_tolerance:
        return False, (f"Grade 3+4 recall fell by {p34 - c34:.3f} ({p34:.3f} → {c34:.3f}), "
                       f"more than the {recall_tolerance:.3f} allowed.")

    recall_text = (f", Grade 3+4 recall {p34:.3f} → {c34:.3f}"
                   if c34 is not None and p34 is not None else "")
    verdict = "improved" if d_qwk > 0 else "held within tolerance"
    return True, f"QWK {verdict} ({current['qwk']:.3f} → {candidate['qwk']:.3f}{recall_text})."
