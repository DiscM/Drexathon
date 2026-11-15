"""Calibrated blend: turn the committee's answers into P(overturn).

The chair's own probability is one Drex guess. The blend is a logistic
regression over every agent answer, the chair's answer and the train-set
rule prior. Its weights come from `calibrate.py`, which fits them on TRAIN
cases only and writes `calibration.json`. Without that file the chair's
probability is used unchanged, so nothing here changes results until a fit
exists.
"""
from __future__ import annotations

import json
import math
import os

import data

PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "calibration.json")
MISSING = 0.5  # value for a feature the run did not produce
_cache: dict = {"mtime": None, "model": None}


def features(answers: dict, chair: dict, fields: dict) -> dict[str, float]:
    """Flatten one committee result into named numbers in [0, 1].

    answers: the last round, {agent: {question: answer}}.
    """
    out: dict[str, float] = {}
    for agent, ans in answers.items():
        for q, v in ans.items():
            x = _value(v)
            if x is not None:
                out[f"{agent}.{q}"] = x
    for q in ("decision", "overturn", "needs_human"):
        x = _value(chair.get(q) or {})
        if x is not None:
            out[f"chair.{q}"] = x
    if data.available():
        out["prior.rule"] = float(data.rule_score(fields))
    return out


def _value(v: dict) -> float | None:
    if "noul" in v:
        return _clip01(float(v["noul"]))
    if "score" in v:
        top = len(v.get("legend") or []) - 1
        return _clip01(float(v["score"]) / top) if top > 0 else None
    if "choice" in v:
        probs = v.get("probabilities") or {}
        if "overturn" in probs:
            return _clip01(float(probs["overturn"]))
        if v["choice"] in ("overturn", "uphold"):
            return 1.0 if v["choice"] == "overturn" else 0.0
    return None  # other choices (e.g. denial_reason) are not numeric


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


def load() -> dict | None:
    """The fitted model, or None when there is no calibration.json."""
    try:
        mtime = os.path.getmtime(PATH)
    except OSError:
        return None
    if _cache["mtime"] != mtime:
        with open(PATH) as f:
            _cache["model"] = json.load(f)
        _cache["mtime"] = mtime
    return _cache["model"]


def score(model: dict, feats: dict[str, float]) -> float:
    z = model["bias"] + sum(w * feats.get(name, MISSING)
                            for name, w in model["weights"].items())
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))


def predict(feats: dict[str, float], p_chair: float) -> tuple[float, str]:
    """(P(overturn), method). method is "blend" or "chair"."""
    model = load()
    if model is None:
        return p_chair, "chair"
    return score(model, feats), "blend"


def fit(rows: list[dict[str, float]], labels: list[bool],
        l2: float = 0.001, steps: int = 6000, lr: float = 0.5) -> dict:
    """L2 logistic regression by gradient descent (no extra packages)."""
    names = sorted({k for r in rows for k in r})
    X = [[r.get(k, MISSING) for k in names] for r in rows]
    y = [1.0 if t else 0.0 for t in labels]
    n = len(X)
    w = [0.0] * len(names)
    b = 0.0
    for _ in range(steps):
        gw = [0.0] * len(names)
        gb = 0.0
        for xi, yi in zip(X, y):
            z = b + sum(wj * xj for wj, xj in zip(w, xi))
            err = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z)))) - yi
            gb += err
            for j, xj in enumerate(xi):
                gw[j] += err * xj
        b -= lr * gb / n
        w = [wj - lr * (g / n + l2 * wj) for wj, g in zip(w, gw)]
    return {"bias": b, "weights": dict(zip(names, w))}
