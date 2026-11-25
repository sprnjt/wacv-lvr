"""Task 2 (POST-REGISTRATION / EXPLORATORY). G2 split query, 160 targets, 800 E2 N=5 source trials per model.
Each source trial = one item-only call + one country-only call (trials_g2_160.jsonl), scored with frozen ch.score_trial.
Frozen definition (work/analysis/error_breakdown.py): among answers whose item AND country both name a shown
distractor (both != target slot, both resolved to a shown image), share where they name the SAME image
= whole / (whole + two). Also given: package definition (denominator adds 'other' = unresolved field).
Frozen-prompt comparison: same 800 trial ids from the frozen raw_<model>.jsonl (MC).
95% CIs: target bootstrap (B=2000, frozen RNG seed), ratio of pooled counts. Run from cr_analyses/_fullpool."""
import sys, json; sys.path.insert(0, ".")
import numpy as np, pandas as pd
import ch, analyze as A
FROZ = "../../wacv-lvr/"
pool = ch.load_pool()
rd = lambda p: [json.loads(l) for l in open(p) if l.strip()]
g2t = {t["trial_id"]: t for t in rd("trials_g2_160.jsonl")}
ft = {t["trial_id"]: t for t in rd(FROZ + "trials_main.jsonl")}
MODELS = ["gemma4", "qwen3vl", "qwen25vl", "gemma3_4b", "lfm25v"]

def kind(i, c, sl):  # i, c = slot of item / country answer (0 = not a shown image / unresolved)
    if i == sl and c == sl: return "right"
    if (i == sl) != (c == sl): return "half"  # one field right (other field wrong or unresolved)
    if i > 0 and c > 0: return "whole" if i == c else "two"
    return "other"

def cis(d):
    per = d.groupby("target").agg(w=("k", lambda s: (s == "whole").sum()), t=("k", lambda s: (s == "two").sum()),
                                  o=("k", lambda s: (s == "other").sum()), n=("k", "size"), r=("k", lambda s: (s == "right").sum()))
    W, T, O, N, R = (per[c].values.astype(float) for c in "wtonr")
    idx = A.RNG.integers(0, len(per), (A.B, len(per)))
    f = lambda num, den: (num.sum() / den.sum(), *A.ci(num[idx].sum(1) / np.maximum(den[idx].sum(1), 1)))
    return dict(n=int(N.sum()), joint=f(R, N), both_wrong=int(W.sum() + T.sum()), whole=int(W.sum()), two=int(T.sum()),
                other_bothwrong=int(O.sum()), same_frozen_def=f(W, W + T), same_pkg_def=f(W, W + T + O))

rows = []
for m in MODELS:
    sc = {}
    for r in rd(f"raw_{m}_g2_160.jsonl"):
        t = g2t[r["trial_id"]]; s = ch.score_trial(t, "mc", r["raw"], pool)
        sc[r["trial_id"]] = s[f"{t['ask']}_slot"]
    sids = sorted({k[3:].rsplit("_Q", 1)[0] for k in sc})
    split = pd.DataFrame([dict(target=ft[s]["target"], k=kind(sc[f"G2_{s}_Qitem"], sc[f"G2_{s}_Qcountry"], ft[s]["slot"]))
                          for s in sids])
    fr = pd.read_csv(f"{FROZ}scored_{m}.csv"); fr = fr[(fr["mode"] == "mc") & fr.trial_id.isin(sids)]
    froz = pd.DataFrame(dict(target=fr.target, k=[kind(a, b, c) for a, b, c in zip(fr.item_slot, fr.country_slot, fr.slot)]))
    for prompt, d in (("split", split), ("frozen", froz)):
        r = cis(d); rows.append(dict(model=m, prompt=prompt, **{k: v for k, v in r.items() if not isinstance(v, tuple)},
            **{f"{k}{s}": v[j] for k, v in r.items() if isinstance(v, tuple) for j, s in enumerate(("", "_lo", "_hi"))}))
out = pd.DataFrame(rows); out.to_csv("../02_g2_same_image/g2_same_image.csv", index=False)
print(out.round(3).to_string())
