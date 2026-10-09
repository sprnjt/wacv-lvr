#!/usr/bin/env python3
"""Independent audit of the analysis pipeline: recomputes headline results from
scored_<model>.csv / raw_<model>.jsonl / attn_<model>.jsonl WITHOUT importing
analyze.py, and diffs them against results/*.csv. Prints PASS/FAIL per check.

  python3 verify_results.py
"""
import json, sys
from pathlib import Path
import pandas as pd

MODELS = ["qwen3vl", "qwen25vl", "gemma4", "gemma3_4b", "granite4v", "lfm25v"]
TOL = 2e-3
fails = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}{('  — ' + detail) if detail else ''}")
    if not ok:
        fails.append(name)


def res(name):
    return pd.read_csv(f"results/{name}.csv")


sc = {m: pd.read_csv(f"scored_{m}.csv") for m in MODELS}
trials = {json.loads(l)["trial_id"]: json.loads(l) for l in open("trials_main.jsonl")}

# 1. completeness: raw rows, repeats, repeat identity
for m in MODELS:
    raw = [json.loads(l) for l in open(f"raw_{m}.jsonl")]
    rep = [json.loads(l) for l in open(f"raw_{m}_repeat.jsonl")]
    check(f"{m}: raw completeness", len(raw) == 6096 and len(rep) == 20,
          f"{len(raw)}/6096 main, {len(rep)}/20 repeat")
    r0 = {(r["trial_id"], r["mode"]): r["raw"] for r in raw}
    same = sum(r0.get((r["trial_id"], r["mode"])) == r["raw"] for r in rep)
    check(f"{m}: repeat identity", same == 20, f"{same}/20")

# 2. joint consistency + 3. judge overrides applied per model
ov = pd.read_csv("judge_overrides.csv")
main_ids = set(trials)
for m in MODELS:
    d = sc[m]
    check(f"{m}: joint == item&country", bool((d.joint_ok == (d.item_ok & d.country_ok)).all()))
    o = ov[(ov.model == m) & (ov.trial_id.isin(main_ids))]
    j = o.merge(d, on=["trial_id", "mode"])
    if len(j):
        ok = j.apply(lambda r: r[f"{r.field}_ok"] == int(r.label == "correct"), axis=1)
        check(f"{m}: overrides applied", bool(ok.all()), f"{len(j)} override rows checked")

# 4. acc_by_N recompute (E1+E2, kind main) vs results/acc_by_N.csv
acc = res("acc_by_N")
worst = 0.0
for m in MODELS:
    d = sc[m][(sc[m].exp.isin(["E1", "E2"])) & (sc[m].kind == "main")]
    for mode in ("open", "mc"):
        g = d[d["mode"] == mode].groupby("N")[["joint_ok", "item_ok", "country_ok"]].mean()
        for _, r in acc[(acc.model == m) & (acc["mode"] == mode)].iterrows():
            mine = g.loc[r.N, f"{r.metric}_ok"]
            worst = max(worst, abs(mine - r.acc))
check("acc_by_N recompute", worst < TOL, f"max |diff| = {worst:.4f}")

# 5. error shares at N=5 (mc) vs results/error_shares_by_N.csv
esh = res("error_shares_by_N")
worst = 0.0
for m in MODELS:
    d = sc[m][(sc[m].exp.isin(["E1", "E2"])) & (sc[m].kind == "main") & (sc[m].N == 5)]
    d = d[d["mode"] == "mc"]
    f = pd.concat([d.item_err[d.item_ok == 0], d.country_err[d.country_ok == 0]]).value_counts(normalize=True)
    r = esh[(esh.model == m) & (esh["mode"] == "mc") & (esh.N == 5)].iloc[0]
    for k in ("mis_binding", "prior_drift", "abstain", "format_error"):
        worst = max(worst, abs(f.get(k, 0.0) - r[k]))
check("error_shares N=5 mc recompute", worst < TOL, f"max |diff| = {worst:.4f}")

# 6. H3a: among wrong fields naming one of the 5 set images (src notna), share that is mis_binding
h3a = res("h3a_context_attraction")
worst = 0.0
for m in MODELS:
    d = sc[m][(sc[m].exp.isin(["E1", "E2"])) & (sc[m].kind == "main") & (sc[m]["mode"] == "mc")]
    for N in (2, 3, 4):
        g = d[d.N == N]
        w = pd.concat([g[(g.item_ok == 0) & g.item_src.notna()].item_err,
                       g[(g.country_ok == 0) & g.country_src.notna()].country_err])
        share = (w == "mis_binding").mean()
        r = h3a[(h3a.model == m) & (h3a["mode"] == "mc") & (h3a.N == N)].iloc[0]
        worst = max(worst, abs(share - r.share_shown))
check("h3a share_shown recompute", worst < TOL, f"max |diff| = {worst:.4f}")

# 7. illusory-conjunction rate among wrong joint, N=5 mc
icr = res("illusory_conjunctions")
worst = 0.0
for m in MODELS:
    d = sc[m][(sc[m].exp.isin(["E1", "E2"])) & (sc[m].kind == "main") & (sc[m]["mode"] == "mc") & (sc[m].N == 5)]
    w = d[d.joint_ok == 0]
    mine = w.illusory.mean()
    r = icr[(icr.model == m) & (icr["mode"] == "mc") & (icr.N == 5)].iloc[0]
    worst = max(worst, abs(mine - r.icr_observed))
check("ICR observed recompute", worst < TOL, f"max |diff| = {worst:.4f}")

# 8. position profile at N=5 (E2, P1, main) vs results/position_profile.csv
pp = res("position_profile")
worst = 0.0
for m in MODELS:
    d = sc[m][(sc[m].exp == "E2") & (sc[m].variant == "P1") & (sc[m].kind == "main") & (sc[m].N == 5)]
    for mode in ("open", "mc"):
        g = d[d["mode"] == mode].groupby("slot").joint_ok.mean()
        for s in range(1, 6):
            r = pp[(pp.model == m) & (pp["mode"] == mode) & (pp.slot == s)].iloc[0]
            worst = max(worst, abs(g.loc[s] - r.acc))
check("position_profile recompute", worst < TOL, f"max |diff| = {worst:.4f}")

# 9. E6: p1 = E5/P1, p4 = E6/P4 joint accuracy, all-N row
dc = res("describe_then_answer")
worst = 0.0
for m in MODELS:
    d = sc[m][sc[m].kind == "main"]
    for mode in ("open", "mc"):
        p1 = d[(d.exp == "E5") & (d.variant == "P1") & (d["mode"] == mode)].joint_ok.mean()
        p4 = d[(d.exp == "E6") & (d.variant == "P4") & (d["mode"] == mode)].joint_ok.mean()
        r = dc[(dc.model == m) & (dc["mode"] == mode) & (dc.N == "all")].iloc[0]
        worst = max(worst, abs(p1 - r.p1_acc), abs(p4 - r.p4_acc))
check("describe_then_answer recompute", worst < TOL, f"max |diff| = {worst:.4f}")

# 10. leak recompute
lk = res("text_only_leak")
worst = 0.0
for m in MODELS:
    d = sc[m][sc[m].exp == "LEAK"]
    for mode in ("open", "mc"):
        g = d[d["mode"] == mode]
        r = lk[(lk.model == m) & (lk["mode"] == mode)].iloc[0]
        worst = max(worst, abs(g.item_ok.mean() - r.item_correct), abs(g.country_ok.mean() - r.country_correct))
check("text_only_leak recompute", worst < TOL, f"max |diff| = {worst:.4f}")

# 11. attention files: shape sanity + argmax-capture recompute at N=5
atc = res("attention_capture")
for m in ("qwen3vl", "gemma4", "granite4v"):
    p = Path(f"attn_{m}.jsonl")
    if not p.exists() or p.stat().st_size == 0:
        check(f"{m}: attention capture", False, "attn file missing/empty — probe rerun pending")
        continue
    rows = [json.loads(l) for l in open(p)]
    ok_shape = all(len(r["attn"]) == trials[r["trial_id"]]["N"] and min(r["attn"]) >= 0 for r in rows)
    d = sc[m][(sc[m].exp == "E2") & (sc[m].kind == "main") & (sc[m].N == 5)]
    hits = tot = 0
    for r in rows:
        t = trials[r["trial_id"]]
        if t["N"] != 5 or t["exp"] != "E2":
            continue
        srow = d[(d.trial_id == r["trial_id"]) & (d["mode"] == r["mode"])]
        if not len(srow):
            continue
        srow = srow.iloc[0]
        am = max(range(5), key=lambda i: r["attn"][i]) + 1
        for f in ("item", "country"):
            if srow[f"{f}_err"] == "mis_binding":
                tot += 1
                hits += int(t["order"].index(srow[f"{f}_src"]) + 1 == am)
    mine = hits / tot if tot else float("nan")
    # table rows are per mode; recompute per mode for a fair diff
    ok_modes = True
    for mode in ("open", "mc"):
        hits = tot = 0
        for row in rows:
            t = trials[row["trial_id"]]
            if t["N"] != 5 or t["exp"] != "E2" or row["mode"] != mode:
                continue
            srow = d[(d.trial_id == row["trial_id"]) & (d["mode"] == mode)]
            if not len(srow):
                continue
            srow = srow.iloc[0]
            am = max(range(5), key=lambda i: row["attn"][i]) + 1
            for f in ("item", "country"):
                if srow[f"{f}_err"] == "mis_binding":
                    tot += 1
                    hits += int(t["order"].index(srow[f"{f}_src"]) + 1 == am)
        mine_m = hits / tot if tot else float("nan")
        rt = atc[(atc.model == m) & (atc["mode"] == mode)]
        if len(rt) and abs(mine_m - rt.iloc[0].share_argmax_is_named) > TOL:
            ok_modes = False
    check(f"{m}: attention capture", ok_shape and ok_modes and len(rows) == 1600,
          f"{len(rows)} rows, shape ok={ok_shape}, argmax share matches={ok_modes}")

print(f"\n{'ALL CHECKS PASS' if not fails else f'{len(fails)} FAILURES: {fails}'}")
sys.exit(1 if fails else 0)
