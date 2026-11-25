"""Task 5 (POST-REGISTRATION / EXPLORATORY). (a) target slot x popularity tier (data/targets.csv) in the frozen
trials at N=2..5 (E1/E2 core, MC). (b) frozen H2 middle-slot test (analyze.position: E2, P1, main, N=5; logistic GLM
joint_ok ~ 1 + (slot-3) + (slot-3)^2, SEs clustered by target, U-shape = quad_coef>0, Holm) redone
  (i) with tier dummies (mid, low; ref high) added, and (ii) within each tier,
for the three <=4B models (gemma3_4b, granite4v, lfm25v), MC and open. Also middle-minus-ends gap
(mean of slots 2-4 minus mean of slots 1,5; per-target, target bootstrap B=2000). Run from cr_analyses/_frozen."""
import sys; sys.path.insert(0, ".")
import numpy as np, pandas as pd
import analyze as A
MODELS = ["gemma3_4b", "granite4v", "lfm25v"]
df = A.load(MODELS)
tier = pd.read_csv("data/targets.csv")[["image_id", "tier"]].rename(columns={"image_id": "target", "tier": "tier_t"})
df = df.merge(tier, on="target", how="left")
assert (df.tier == df.tier_t).all()
core, _ = A.core_sets(df)
c = core[(core.model == "gemma3_4b") & (core["mode"] == "mc")]
tab = pd.crosstab([c.N, c.tier], c.slot); tab.to_csv("../05_slot_tier/slot_by_tier.csv"); print(tab.to_string())
d0 = df[(df.exp == "E2") & (df.variant == "P1") & (df.kind == "main")]
rows = []
for (m, mode), g in d0.groupby(A.KEYS):
    for sub, h in [("all+tier_cov", g)] + [(f"tier={t}", g[g.tier == t]) for t in ("high", "mid", "low")]:
        x = h.slot.values - 3.0
        X = [np.ones_like(x), x, x ** 2]
        if sub == "all+tier_cov":
            X += [(h.tier == "mid").astype(float).values, (h.tier == "low").astype(float).values]
        r = A.fit(h.joint_ok.values.astype(float), np.c_[tuple(X)], pd.factorize(h.target)[0])
        M = A.wide(h, "joint_ok", by="slot").values
        gap = M[:, 1:4].mean(1) - M[:, [0, 4]].mean(1); bs = A.boot(gap[:, None])[:, 0]; lo, hi = A.ci(bs)
        rows.append(dict(model=m, mode=mode, subset=sub, targets=len(M), mid_minus_ends=gap.mean(), gap_lo=lo, gap_hi=hi,
                         quad_coef=r.params[2] if r is not None else np.nan, quad_p=r.pvalues[2] if r is not None else np.nan))
out = pd.DataFrame(rows)
for fam, msk in (("cov", out.subset == "all+tier_cov"), ("within", out.subset != "all+tier_cov")):
    for mode in ("mc", "open"):
        k = msk & (out["mode"] == mode); out.loc[k, "quad_p_holm"] = A.holm(out.loc[k, "quad_p"])
out["U_shape_sig"] = (out.quad_coef > 0) & (out.quad_p_holm < 0.05)
fz = pd.read_csv("../../wacv-lvr/results/h2_position.csv"); fz = fz[fz.model.isin(MODELS)]
out.to_csv("../05_slot_tier/h2_by_tier.csv", index=False)
print("\nfrozen h2:\n", fz.round(4).to_string(index=False))
print("\n", out.round(4).to_string(index=False))
