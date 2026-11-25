"""Task 1 (POST-REGISTRATION / EXPLORATORY). Full-pool MC joint accuracy at N=1,3,5 and illusory
("answer from another photo") rate at N=3,5, with 95% target-bootstrap CIs (B=2000), frozen analyze.py helpers.
Two trial selections:
  core      = frozen analyze.core_sets (E1+E2, P1, mixed list); per-target mean, then mean over targets
              (what results/acc_by_N.csv uses).
  firstlook = all kind=='main' trials pooled (the run report's first-look table); bootstrap resamples targets,
              statistic = pooled trial mean.
Illusory = ch.score_trial 'illusory' flag (wrong joint answer with item and country both from shown, different
images), denominator = ALL trials at that N (same as the first-look table; 0 at N=1 by construction).
Run from cr_analyses/_fullpool."""
import sys; sys.path.insert(0, ".")
import numpy as np, pandas as pd
import analyze as A
MODELS = ["gemma4", "qwen3vl", "qwen25vl", "gemma3_4b", "lfm25v", "gemma3_12b"]
df = A.load(MODELS); df = df[df["mode"] == "mc"]
core, _ = A.core_sets(df)
fl = df[df.kind == "main"]
rows = []
for m in MODELS:
    for sel, d in (("core", core), ("firstlook", fl)):
        g = d[d.model == m]
        for metric in ("joint_ok", "illusory"):
            for n in (1, 3, 5):
                h = g[g.N == n]
                per = h.groupby("target")[metric].agg(["sum", "size"])
                S, K = per["sum"].values.astype(float), per["size"].values.astype(float)
                idx = A.RNG.integers(0, len(per), (A.B, len(per)))
                if sel == "core":
                    v = S / K; est = v.mean(); bs = v[idx].mean(1)
                else:
                    est = S.sum() / K.sum(); bs = S[idx].sum(1) / K[idx].sum(1)
                lo, hi = A.ci(bs)
                rows.append(dict(model=m, selection=sel, metric=metric.replace("_ok", ""), N=n, est=est, lo=lo, hi=hi,
                                 targets=len(per), trials=int(K.sum())))
out = pd.DataFrame(rows); out.to_csv("../01_fullpool_ci/fullpool_ci.csv", index=False)
# check vs package acc_by_N (core joint)
pk = pd.read_csv("../../g3_table1/dl/tar/fullpool/results/acc_by_N.csv")
pk = pk[(pk["mode"] == "mc") & (pk.metric == "joint")].set_index(["model", "N"]).acc
c = out[(out.selection == "core") & (out.metric == "joint")].set_index(["model", "N"]).est
print("max |core - package acc_by_N| =", float((c - pk).dropna().abs().max()))
w = out.assign(cell=lambda x: x.apply(lambda r: f"{r.est:.3f} [{r.lo:.3f},{r.hi:.3f}]", axis=1))
print(w.pivot_table(index=["selection", "model"], columns=["metric", "N"], values="cell", aggfunc="first").to_string())
print(out.groupby(["selection", "model", "N"]).trials.first().unstack().to_string())
