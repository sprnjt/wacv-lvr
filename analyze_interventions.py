#!/usr/bin/env python3
"""Mechanical MC scoring and paired summaries for E7/E8 exploratory controls."""
import argparse
import csv
from pathlib import Path
import ch


def main(model, raw_path, trials_path):
    trials = {t["trial_id"]: t for t in ch.read_jsonl(trials_path)}
    pool = ch.load_pool()
    rows = []
    for r in ch.read_jsonl(raw_path):
        if r["mode"] != "mc":
            continue
        t = trials[r["trial_id"]]
        s = ch.score_trial(t, "mc", r["raw"], pool)
        rows.append(dict(trial_id=t["trial_id"], exp=t["exp"], target=t["target"], N=t["N"], slot=t["slot"], **s))
    if not rows:
        raise RuntimeError("no MC rows to analyze")
    Path("results").mkdir(exist_ok=True)
    detail = Path(f"results/interventions_{model}_rows.csv")
    with detail.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    # Paired frozen baselines: E7 maps to E6 and E8 maps to E2.
    base = {(r["trial_id"], r["mode"]): r for r in csv.DictReader(open(f"scored_{model}.csv", encoding="utf-8"))}
    summary = []
    for exp, prefix in (("E7", "E6"), ("E8", "E2")):
        d = [r for r in rows if r["exp"] == exp]
        paired = []
        for r in d:
            old = r["trial_id"].replace(exp + "_", prefix + "_")
            old = old.replace("_P5", "_P4") if exp == "E7" else old
            b = base.get((old, "mc"))
            if b:
                paired.append((r["joint_ok"], int(b["joint_ok"])))
        summary.append(dict(model=model, exp=exp, n=len(d), joint_acc=sum(r["joint_ok"] for r in d) / len(d),
                            paired_n=len(paired), delta_vs_baseline=sum(x-y for x, y in paired) / len(paired)))
    with open(f"results/interventions_{model}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=summary[0].keys()); w.writeheader(); w.writerows(summary)
    print("wrote", detail, "and", f"results/interventions_{model}.csv")


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("model"); p.add_argument("--raw", required=True); p.add_argument("--trials", default="trials_interventions.jsonl")
    a = p.parse_args(); main(a.model, a.raw, a.trials)
