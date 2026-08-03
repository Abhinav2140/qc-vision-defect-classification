"""
severity.py — Turns raw model outputs into an operational pass/reject decision.

The model's severity head outputs a continuous 0-1 score, but a production
line needs a discrete action. This module applies:

  1. Per-defect-type thresholds (a 0.3-severity scratch might be cosmetic
     and acceptable; a 0.3-severity missing_component almost never is —
     thresholds are NOT the same across defect types).
  2. Confidence gating — if the classifier isn't confident, route to human
     review instead of auto-rejecting (avoids false-rejecting good parts
     on borderline frames).
  3. A severity band label (minor / major / critical) for the dashboard
     and for operators watching the line HMI.

Thresholds below are sane defaults — calibrate them against your own
QC team's accept/reject history before going live. `calibrate_thresholds()`
shows how to fit them from labeled outcome data.
"""

from dataclasses import dataclass
from constants import DEFECT_CLASSES


# Severity threshold above which a defect type triggers automatic rejection.
# Missing components and dimensional errors get near-zero tolerance because
# they're functional failures, not cosmetic ones.
DEFAULT_REJECT_THRESHOLDS = {
    "ok": 1.01,                     # never rejects
    "surface_scratch": 0.55,
    "dent": 0.45,
    "dimensional_error": 0.20,
    "missing_component": 0.10,
    "color_inconsistency": 0.50,
    "contamination": 0.30,
}

# Below this classifier confidence, don't trust the auto-decision — flag
# for human review instead of silently passing or silently rejecting.
CONFIDENCE_REVIEW_FLOOR = 0.65


@dataclass
class Decision:
    defect_type: str
    severity: float
    confidence: float
    band: str          # "none" | "minor" | "major" | "critical"
    action: str        # "pass" | "reject" | "human_review"


def severity_band(severity: float) -> str:
    if severity < 0.01:
        return "none"
    if severity < 0.35:
        return "minor"
    if severity < 0.70:
        return "major"
    return "critical"


def decide(defect_type: str, severity: float, confidence: float,
           thresholds: dict = None) -> Decision:
    thresholds = thresholds or DEFAULT_REJECT_THRESHOLDS
    band = severity_band(severity)

    if defect_type == "ok":
        return Decision(defect_type, severity, confidence, band, "pass")

    if confidence < CONFIDENCE_REVIEW_FLOOR:
        return Decision(defect_type, severity, confidence, band, "human_review")

    threshold = thresholds.get(defect_type, 0.5)
    action = "reject" if severity >= threshold else "pass"
    return Decision(defect_type, severity, confidence, band, action)


def calibrate_thresholds(history: "list[dict]") -> dict:
    """Fit per-class thresholds from historical (severity, human_verdict) pairs.

    `history` items look like:
        {"defect_type": "surface_scratch", "severity": 0.42, "human_verdict": "reject"}

    Finds, per class, the severity value that best separates human-verified
    accepts from rejects (simple 1D threshold sweep — swap in logistic
    regression or isotonic regression if you want a smoother boundary).
    """
    import pandas as pd
    df = pd.DataFrame(history)
    fitted = {}
    for defect_type, group in df.groupby("defect_type"):
        if defect_type == "ok" or group["human_verdict"].nunique() < 2:
            continue
        best_thr, best_acc = 0.5, 0.0
        for thr in [i / 100 for i in range(0, 101, 1)]:
            pred = group["severity"] >= thr
            actual = group["human_verdict"] == "reject"
            acc = (pred == actual).mean()
            if acc > best_acc:
                best_acc, best_thr = acc, thr
        fitted[defect_type] = best_thr
    return fitted
