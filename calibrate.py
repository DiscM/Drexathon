"""Fit the calibrated blend (blend.py) on TRAIN cases.

    .venv/bin/python calibrate.py --n 300

It runs the committee on n train cases that have a leak-free summary,
keeps every run in results/calibration_runs.jsonl (so a re-run only pays
for new cases), reports cross-validated accuracy and AUC for the chair
alone and for the blend, and writes calibration.json.

The benchmark set is the test split, so these cases never overlap it.
Mock answers are random: a mock run tests the plumbing and never writes
calibration.json.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor

import blend
import committee
import data
import nace
from benchmark import _auc

RUNS = os.path.join("results", "calibration_runs.jsonl")


def _collect(n: int, workers: int, path: str) -> list[dict]:
    done: dict[str, dict] = {}
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                row = json.loads(line)
                done[row["case_id"]] = row
    cases = [c for c in data.train_cases() if c["case_summary"]]
    random.Random(7).shuffle(cases)
    cases = cases[:n]
    todo = [c for c in cases if c["case_id"] not in done]
    print(f"{len(cases)} train cases: {len(cases) - len(todo)} cached, "
          f"{len(todo)} to run")

    def one(c: dict) -> dict | None:
        try:
            res = committee.run(c)
        except nace.NaceError as e:
            print(f"  skipped {c['case_id']}: {e}")
            return None
        return {"case_id": c["case_id"], "label": bool(c["overturned"]),
                "p_chair": res["overturn_prob_chair"],
                "features": res["features"],
                "input_tokens": res["input_tokens"]}

    if todo:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a") as f, \
                ThreadPoolExecutor(max_workers=workers) as ex:
            for i, row in enumerate(ex.map(one, todo), 1):
                if row is None:
                    continue
                f.write(json.dumps(row) + "\n")
                f.flush()
                done[row["case_id"]] = row
                if i % 25 == 0 or i == len(todo):
                    print(f"  {i}/{len(todo)}")
    return [done[c["case_id"]] for c in cases if c["case_id"] in done]


def _report(name: str, scores: list[float], labels: list[bool]) -> None:
    acc = sum((s >= 0.5) == y for s, y in zip(scores, labels)) / len(labels)
    auc = _auc(scores, labels)
    print(f"  {name:<22} acc {acc:.3f}  AUC "
          + (f"{auc:.3f}" if auc is not None else "-"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--l2", type=float, default=0.001)
    args = ap.parse_args()

    nace.configure()
    mock = nace.is_mock()
    path = RUNS.replace(".jsonl", "_mock.jsonl") if mock else RUNS
    rows = _collect(args.n, args.workers, path)
    X = [r["features"] for r in rows]
    y = [r["label"] for r in rows]
    if len(set(y)) < 2 or len(rows) < 2 * args.folds:
        print("not enough cases to fit; raise --n")
        return 1

    # cross-validation: every case is scored by a model that never saw it
    cv = [0.0] * len(rows)
    for k in range(args.folds):
        test = [i for i in range(len(rows)) if i % args.folds == k]
        train = [i for i in range(len(rows)) if i % args.folds != k]
        m = blend.fit([X[i] for i in train], [y[i] for i in train], l2=args.l2)
        for i in test:
            cv[i] = blend.score(m, X[i])

    print(f"\n{len(rows)} train cases, {args.folds}-fold cross-validation:")
    _report("chair alone", [r["p_chair"] for r in rows], y)
    if all("prior.rule" in x for x in X):
        _report("rule prior alone", [x["prior.rule"] for x in X], y)
    _report("blend (held-out folds)", cv, y)

    model = blend.fit(X, y, l2=args.l2)
    top = sorted(model["weights"].items(), key=lambda kv: -abs(kv[1]))[:8]
    print("\nlargest weights:")
    for name, w in top:
        print(f"  {w:+.2f}  {name}")

    if mock:
        print("\nNOTE: mock mode - answers are random, these numbers are NOT "
              "real results. calibration.json was not written.")
        return 0
    model.update({"n_train": len(rows), "l2": args.l2,
                  "model": nace.configure()["model"],
                  "fitted": time.strftime("%Y-%m-%dT%H:%M:%S")})
    with open(blend.PATH, "w") as f:
        json.dump(model, f, indent=2)
    print(f"\nsaved: {blend.PATH}\nThe committee now uses the blend. Delete "
          "the file to go back to the chair's probability.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
