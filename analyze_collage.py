#!/usr/bin/env python3
"""Mechanical MC scoring and paired summaries for the E9 collage control (exploratory, post-freeze).

  python3 analyze_collage.py MODEL --raw raw_MODEL_collage.jsonl [--trials trials_collage.jsonl]
  -> results/collage_MODEL.csv, long format: model, table, condition, metric, n, value, lo, hi, note

Works on partial raw files (missing conditions/targets are skipped). CIs: percentile bootstrap over targets
(one trial per target and condition, so this is the target-clustered bootstrap), B=2000, seed 0.
  acc       item / country / joint accuracy and format_errors (trials with a field that did not parse)
  delta     paired over targets: C3-C2 scene vs separate, C4-C3 kind vs position, C2/C3/C4 - C1
  retention accuracy among targets whose field (or joint) was correct in C1
  errsrc    wrong fields by source: other_panel (a shown panel; ch "mis_binding"), unshown (decoy, or in C1
            any non-target option; ch "prior_drift"), abstain, format_error. value = count, n = wrong fields
  icr       illusory conjunctions (item and country from two different shown panels) among wrong joint
            answers, vs a slot-conditioned permutation baseline as in analyze.py (country answers shuffled
            among trials with the same target slot). Not defined for C1 (one panel shown).
C4 (collage_kind) names the target's kind, so its item field (and hence joint) is solvable by kind alone
against the 3 other-category options: for C4 read COUNTRY accuracy and the ICR, not item/joint.
"""
import argparse
import csv
from pathlib import Path
import numpy as np
import ch

B, PERMS = 2000, 1000
CONDS = ("solo", "separate", "collage_pos", "collage_kind")
NAME = dict(zip(CONDS, ("C1", "C2", "C3", "C4")))
CONTRASTS = (("collage_pos", "separate"), ("collage_kind", "collage_pos"),
             ("separate", "solo"), ("collage_pos", "solo"), ("collage_kind", "solo"))
C4_NOTE = "C4 item/joint solvable by kind: use country and ICR"
COLS = ["model", "table", "condition", "metric", "n", "value", "lo", "hi", "note"]


def boot_ci(x):
    x = np.asarray(x, float)
    if len(x) < 2:
        return np.nan, np.nan
    idx = np.random.default_rng(0).integers(0, len(x), (B, len(x)))
    lo, hi = np.percentile(x[idx].mean(1), [2.5, 97.5])
    return lo, hi


def icr(rows):
    """Observed ICR, permutation baseline and one-sided p (analyze.icr, restricted to one condition)."""
    sl = np.array([r["slot"] for r in rows])
    i = np.array([r["item_slot"] for r in rows])
    c = np.array([r["country_slot"] for r in rows])

    def rate(cc):
        wrong = ~((i == sl) & (cc == sl))
        return ((i > 0) & (cc > 0) & (i != cc) & wrong).sum() / wrong.sum()

    wrong = int((~((i == sl) & (c == sl))).sum())
    if wrong < 5:
        return wrong, np.nan, np.nan, np.nan
    rng, obs, perm = np.random.default_rng(1), rate(c), []
    groups = [np.where(sl == k)[0] for k in np.unique(sl)]
    for _ in range(PERMS):
        cp = c.copy()
        for ix in groups:
            cp[ix] = rng.permutation(c[ix])
        perm.append(rate(cp))
    perm = np.array(perm)
    return wrong, obs, perm.mean(), (1 + (perm >= obs).sum()) / (PERMS + 1)


def main(model, raw_path, trials_path):
    trials = {t["trial_id"]: t for t in ch.read_jsonl(trials_path)}
    pool = ch.load_pool()
    S = {c: {} for c in CONDS}                    # condition -> target -> scored row
    for r in ch.read_jsonl(raw_path):
        if r["mode"] != "mc":
            continue
        t = trials[r["trial_id"]]
        s = ch.score_trial(t, "mc", r["raw"], pool)
        S[t["condition"]].setdefault(t["target"], dict(s, slot=t["slot"]))   # first row wins (resume never repeats)
    if not any(S.values()):
        raise RuntimeError("no E9 MC rows to analyze")
    out = []

    def add(table, cond, metric, n, value, lo=np.nan, hi=np.nan, note=""):
        out.append(dict(model=model, table=table, condition=cond, metric=metric, n=n, value=value, lo=lo, hi=hi, note=note))

    def note(cond, metric):
        return C4_NOTE if "collage_kind" in cond and metric in ("item", "joint") else ""

    for cond in CONDS:
        g = list(S[cond].values())
        if not g:
            continue
        for m in ("item", "country", "joint"):
            x = [r[f"{m}_ok"] for r in g]
            add("acc", cond, m, len(x), np.mean(x), *boot_ci(x), note(cond, m))
        add("acc", cond, "format_errors", len(g), sum("format_error" in (r["item_err"], r["country_err"]) for r in g))
        for f in ("item", "country"):
            w = [r[f"{f}_err"] for r in g if r[f"{f}_err"]]
            for src, err in (("other_panel", "mis_binding"), ("unshown", "prior_drift"), ("abstain", "abstain"),
                             ("format_error", "format_error")):
                add("errsrc", cond, f"{f}:{src}", len(w), w.count(err))
        if cond != "solo":
            # slot == order position of the target in C2-C4 (order = panels); score_trial gives *_slot the same way
            wrong, obs, base, p = icr(g)
            add("icr", cond, "icr", wrong, obs, note="n = wrong joint answers" + ("" if wrong >= 5 else "; < 5, skipped"))
            add("icr", cond, "icr_baseline", wrong, base)
            add("icr", cond, "icr_p_perm", wrong, p)
    for a, b in CONTRASTS:
        tg = sorted(set(S[a]) & set(S[b]))
        if not tg:
            continue
        for m in ("item", "country", "joint"):
            d = [S[a][t][f"{m}_ok"] - S[b][t][f"{m}_ok"] for t in tg]
            add("delta", f"{NAME[a]}-{NAME[b]} {a}-{b}", m, len(d), np.mean(d), *boot_ci(d), note(a + b, m))
    for cond in CONDS[1:]:
        for m in ("item", "country", "joint"):
            x = [S[cond][t][f"{m}_ok"] for t in sorted(set(S[cond]) & set(S["solo"])) if S["solo"][t][f"{m}_ok"]]
            if x:
                add("retention", cond, m, len(x), np.mean(x), *boot_ci(x), note(cond, m))
    Path("results").mkdir(exist_ok=True)
    path = Path(f"results/collage_{model}.csv")
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        w.writerows(out)
    for r in out:
        if r["table"] in ("acc", "delta", "icr") and r["metric"] != "icr_p_perm":
            print(f"{r['table']:6} {r['condition']:36} {r['metric']:14} n={r['n']:<4} {r['value']:.3f}"
                  + ("" if np.isnan(r["lo"]) else f" [{r['lo']:.3f}, {r['hi']:.3f}]") + (f"  ({r['note']})" if r["note"] else ""))
    print("NOTE:", C4_NOTE)
    print("wrote", path)
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("model"); p.add_argument("--raw", required=True); p.add_argument("--trials", default="trials_collage.jsonl")
    a = p.parse_args(); main(a.model, a.raw, a.trials)
