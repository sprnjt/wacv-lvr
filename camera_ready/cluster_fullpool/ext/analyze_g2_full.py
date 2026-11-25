# G2 full-pool copy of analyze_ext.py (post-registration, exploratory); usage: python ext/analyze_g2_full.py g2
"""Post-registration / exploratory analysis of G1, G2, G3 and F1 subsets (CPU). cwd = fullpool/. Usage: python ext/analyze_ext.py [g1] [g2] [g3] [f1]
Writes results/g1_near_far.csv, g2_split_full.csv, g3_table2.csv, f1_subsets.csv, repro_large.csv. Missing inputs are skipped."""
import os, sys, json
from pathlib import Path
import numpy as np, pandas as pd
H = Path.home(); W = H/"CultureHaystack_handoff"/"wacv-lvr"; F = H/"CultureHaystack_handoff"/"fullpool"
sys.path.insert(0, str(W)); os.chdir(F)
import ch
pool = ch.load_pool(); rng = np.random.default_rng(0); B = 2000
OUT = Path("results"); OUT.mkdir(exist_ok=True)
M5 = ["gemma3_4b", "lfm25v", "qwen25vl", "qwen3vl", "gemma4"]
rd = lambda p: [json.loads(l) for l in open(p) if l.strip()]
frozen_t = {json.loads(l)["target"] for l in open(W/"trials_main.jsonl") if l.strip()}
def score(raw_path, trials, mode="mc", ov=None):
    rows = []
    for r in rd(raw_path):
        if r["mode"] != mode or r["trial_id"] not in trials: continue
        t = trials[r["trial_id"]]
        s = ch.score_trial(t, mode, r["raw"], pool, ov)
        rows.append(dict(trial_id=t["trial_id"], target=t["target"], exp=t["exp"], list=t["list"], N=t["N"], slot=t["slot"], category=t["category"], ask=t.get("ask"), order=t["order"], **s))
    return pd.DataFrame(rows)
def ci(v):
    v = np.asarray(v, float)
    if len(v) == 0: return (np.nan, np.nan, np.nan)
    m = v[rng.integers(0, len(v), (B, len(v)))].mean(1)
    return float(v.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))
def g1():
    tr = {t["trial_id"]: t for t in rd("trials_g1.jsonl")}; out = []
    for m in M5 + ["gemma3_12b"]:
        p = f"raw_{m}_g1.jsonl"
        if not Path(p).exists(): continue
        d = score(p, tr)
        if d.empty: continue
        for met in ("joint_ok", "item_ok", "country_ok"):
            w = d.pivot_table(index="target", columns="list", values=met, aggfunc="mean").dropna()
            for sub, mask in (("frozen160", w.index.isin(frozen_t)), ("all860", np.ones(len(w), bool))):
                x = w[mask]
                if len(x) == 0: continue
                mean, lo, hi = ci((x["near"] - x["far"]).values)
                out.append(dict(model=m, subset=sub, metric=met, n_targets=len(x), near=x["near"].mean(), far=x["far"].mean(), near_minus_far=mean, lo=lo, hi=hi))
    pd.DataFrame(out).to_csv(OUT/"g1_near_far.csv", index=False); print("g1 rows", len(out))
def cells(d):
    c = {N: d[(d.exp == "E1") & (d.N == N)].groupby("target").joint_ok.mean() for N in (1, 2, 3, 4)}
    c[5] = d[d.exp == "E2"].groupby("target").joint_ok.mean()
    return pd.DataFrame(c).dropna()
def table2(d, model, mode, subset):
    Wd = cells(d)
    if len(Wd) == 0: return None
    xs = np.array([2, 3, 4, 5]); sl = Wd[[2, 3, 4, 5]].apply(lambda r: np.polyfit(xs, r.values, 1)[0], axis=1).values
    a1, a5, dr, s = ci(Wd[1]), ci(Wd[5]), ci(Wd[1] - Wd[2]), ci(sl)
    return dict(model=model, mode=mode, subset=subset, n_targets=len(Wd), acc1=a1[0], acc1_lo=a1[1], acc1_hi=a1[2], acc5=a5[0], acc5_lo=a5[1], acc5_hi=a5[2],
                drop_1_to_2=dr[0], drop_lo=dr[1], drop_hi=dr[2], slope_2_5=s[0], slope_lo=s[1], slope_hi=s[2])
def g3():
    tr = {t["trial_id"]: t for t in rd(W/"trials_main.jsonl")}; out = []
    for model, p in (("gemma3_12b", "raw_gemma3_12b_main.jsonl"), ("gemma3_4b", str(W/"raw_gemma3_4b.jsonl"))):
        if not Path(p).exists(): continue
        for mode in ("mc", "open"):
            ov = ch.read_overrides(model=model) if mode == "open" else None
            d = score(p, tr, mode, ov); r = table2(d, model, mode, "frozen160") if len(d) else None
            if r: out.append(r)
    pd.DataFrame(out).to_csv(OUT/"g3_table2.csv", index=False); print("g3 rows", len(out))
def cls(i, c, t, shown):
    if i == t and c == t: return "both_right"
    if i == t or c == t: return "half_right"
    if i is not None and i == c and i in shown: return "whole_swap_same_image"
    if i in shown and c in shown and i is not None and c is not None: return "two_image_mix"
    return "other"
def g2():
    tr = {t["trial_id"]: t for t in rd("trials_g2_full.jsonl")}; out = []
    ft = {t["trial_id"]: t for t in rd(W/"trials_main.jsonl")}
    for m in M5 + ["gemma3_12b"]:
        p = f"raw_{m}_g2_full.jsonl"
        if not Path(p).exists(): continue
        d = score(p, tr)
        if d.empty: continue
        d["sid"] = [x[3:].rsplit("_Q", 1)[0] for x in d.trial_id]
        it = d[d.ask == "item"].set_index("sid"); co = d[d.ask == "country"].set_index("sid"); ids = it.index.intersection(co.index)
        def stats(get):
            k = {"both_right": 0, "half_right": 0, "whole_swap_same_image": 0, "two_image_mix": 0, "other": 0}
            for s in ids:
                i, c, t, sh = get(s); k[cls(i, c, t, sh)] += 1
            n = max(len(ids), 1); bw = k["whole_swap_same_image"] + k["two_image_mix"] + k["other"]
            return n, k, bw
        n, k, bw = stats(lambda s: (it.loc[s, "item_src"], co.loc[s, "country_src"], tr["G2_" + s + "_Qitem"]["target"], set(tr["G2_" + s + "_Qitem"]["order"])))
        row = dict(model=m, n_source_trials=n, joint_split=k["both_right"] / n, half_right=k["half_right"] / n, whole_swap_same_image=k["whole_swap_same_image"] / n, two_image_mix=k["two_image_mix"] / n,
                   same_image_given_both_wrong=(k["whole_swap_same_image"] / bw) if bw else np.nan)
        fp = W/f"raw_{m}.jsonl"
        if fp.exists():
            fd = score(str(fp), ft)
            fd = fd[fd.trial_id.isin(set(ids))].set_index("trial_id")
            kk = {"both_right": 0, "half_right": 0, "whole_swap_same_image": 0, "two_image_mix": 0, "other": 0}
            for s, r in fd.iterrows(): kk[cls(r["item_src"], r["country_src"], ft[s]["target"], set(ft[s]["order"]))] += 1
            nn = max(len(fd), 1); b2 = kk["whole_swap_same_image"] + kk["two_image_mix"] + kk["other"]
            row.update(joint_frozen_prompt=kk["both_right"] / nn, same_image_given_both_wrong_frozen=(kk["whole_swap_same_image"] / b2) if b2 else np.nan)
        out.append(row)
    pd.DataFrame(out).to_csv(OUT/"g2_split_full.csv", index=False); print("g2 rows", len(out))
def f1():
    tr = {t["trial_id"]: t for t in rd("trials_full.jsonl")}; out = []; rep = []
    for m in M5:
        p = f"raw_{m}.jsonl"
        if not Path(p).exists(): continue
        for mode in ("mc", "open"):
            ov = ch.read_overrides(model=m) if mode == "open" else None
            d = score(p, tr, mode, ov)
            if d.empty: continue
            subs = [("all884", d), ("frozen160", d[d.target.isin(frozen_t)]), ("new724", d[~d.target.isin(frozen_t)])] + [("cat_" + c, d[d.category == c]) for c in sorted(d.category.unique())]
            for name, dd in subs:
                r = table2(dd, m, mode, name)
                if r: out.append(r)
            e2 = d[d.exp == "E2"]
            for k_, g in e2.groupby("slot"): out.append(dict(model=m, mode=mode, subset="E2_slot_profile_N5", slot=int(k_), joint_acc=g.joint_ok.mean(), n=len(g)))
            for lst, g in d[d.exp == "E3"].groupby(["list", "N"]): out.append(dict(model=m, mode=mode, subset=f"E3_{lst[0]}_N{lst[1]}", joint_acc=g.joint_ok.mean(), n=len(g)))
        fz = W/f"raw_{m}.jsonl"
        if fz.exists():
            fr = {(r["trial_id"], r["mode"]): r["raw"] for r in rd(fz)}
            for mode in ("mc", "open"):
                cur = {(r["trial_id"], r["mode"]): r["raw"] for r in rd(p) if r["mode"] == mode}
                both = [k for k in fr if k[1] == mode and k in cur]
                if both: rep.append(dict(model=m, mode=mode, frozen_trials=len(both), identical=sum(fr[k] == cur[k] for k in both) / len(both)))
    pd.DataFrame(out).to_csv(OUT/"f1_subsets.csv", index=False); pd.DataFrame(rep).to_csv(OUT/"repro_large.csv", index=False); print("f1 rows", len(out), "repro rows", len(rep))
if __name__ == "__main__":
    for a in (sys.argv[1:] or ["g1", "g2", "g3", "f1"]):
        globals()[a]()
