#!/usr/bin/env python3
"""CultureHaystack analysis. Reads scored_<MODEL>.csv (from `ch.py score`), writes results/*.csv and results/summary.md.

  python analyze.py MODEL [MODEL ...]

Optional: clip_sim.csv (columns: target, list, mean_sim) adds the CLIP-similarity control for H3.
Mixed-effects models are approximated here by logistic GLMs with standard errors clustered by target;
refit the final models with glmer (R) or a mixed model if you want crossed random effects.
"""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.multitest import multipletests

B, PERMS = 2000, 1000
RNG = np.random.default_rng(0)
OUT = Path("results")
OUT.mkdir(exist_ok=True)
KEYS = ["model", "mode"]


def load(models):
    df = pd.concat([pd.read_csv(f"scored_{m}.csv") for m in models], ignore_index=True)
    t = pd.read_csv("data/targets.csv")[["image_id", "idx"]].rename(columns={"image_id": "target"})
    return df.merge(t, on="target", how="left")


def wide(d, col, by="N"):
    return d.groupby(["target", by])[col].mean().unstack(by).dropna()


def boot(M):
    idx = RNG.integers(0, M.shape[0], (B, M.shape[0]))
    return M[idx].mean(axis=1)


def ci(x):
    return np.percentile(x, [2.5, 97.5], axis=0)


def slope(y, x):
    x = np.asarray(x, float)
    xc = x - x.mean()
    return (y - y.mean(axis=-1, keepdims=True)) @ xc / (xc @ xc)


def holm(p):
    p = np.asarray(p, float)
    out = np.full(p.shape, np.nan)
    ok = ~np.isnan(p)
    if ok.any():
        out[ok] = multipletests(p[ok], method="holm")[1]
    return out


def fit(y, X, groups=None):
    try:
        m = sm.GLM(y, X, family=sm.families.Binomial())
        return m.fit(cov_type="cluster", cov_kwds={"groups": groups}) if groups is not None else m.fit()
    except Exception:
        return None


def md(df, nd=3):
    if df.empty:
        return "(no rows)\n"
    d = df.copy()
    for c in d.columns:
        if d[c].dtype.kind == "f":
            d[c] = d[c].map(lambda v: "" if pd.isna(v) else f"{v:.{nd}f}")
    rows = ["| " + " | ".join(map(str, d.columns)) + " |", "| " + " | ".join("---" for _ in d.columns) + " |"]
    rows += ["| " + " | ".join(map(str, r)) + " |" for r in d.astype(str).values]
    return "\n".join(rows) + "\n"


def core_sets(df):
    ok = (df.kind == "main") & (df.variant == "P1") & (df.list == "mixed")
    core = df[ok & df.exp.isin(["E1", "E2"])]
    bal = core[(core.exp == "E1") | ((core.exp == "E2") & (core.slot == core.idx % 5 + 1))]
    return core, bal


def scaling(core):
    acc, summ = [], []
    for (m, mode), g in core.groupby(KEYS):
        for metric in ("joint_ok", "item_ok", "country_ok"):
            M = wide(g, metric)
            if M.shape[1] != 5:
                continue
            A, bm = M.values, boot(M.values)
            lo, hi = ci(bm)[0], ci(bm)[1]
            for j, n in enumerate(M.columns):
                acc.append(dict(model=m, mode=mode, metric=metric.replace("_ok", ""), N=n, acc=A[:, j].mean(),
                                lo=lo[j], hi=hi[j], targets=len(A)))
            if metric == "joint_ok":
                d = A.mean(0)[0] - A.mean(0)[1]
                dd = ci(bm[:, 0] - bm[:, 1])
                s = slope(A.mean(0)[1:], [2, 3, 4, 5])
                ss = ci(slope(bm[:, 1:], [2, 3, 4, 5]))
                summ.append(dict(model=m, mode=mode, acc_N1=A.mean(0)[0], initial_drop=d, drop_lo=dd[0], drop_hi=dd[1],
                                 slope_per_image=s, slope_lo=ss[0], slope_hi=ss[1]))
    return pd.DataFrame(acc), pd.DataFrame(summ)


def h1(bal):
    rows = []
    for (m, mode), g in bal.groupby(KEYS):
        y = g.joint_ok.values.astype(float)
        N = g.N.values.astype(float)
        step = (N >= 2).astype(float)
        one = np.ones_like(N)
        grp = pd.factorize(g.target)[0]
        lin, st, both = (fit(y, np.c_[one, N]), fit(y, np.c_[one, step]), fit(y, np.c_[one, step, N]))
        if None in (lin, st, both):
            rows.append(dict(model=m, mode=mode, note="fit failed (perfect separation?)"))
            continue
        lr = 2 * (both.llf - lin.llf)
        bc = fit(y, np.c_[one, step, N], grp)
        lc = fit(y, np.c_[one, N], grp)
        rows.append(dict(model=m, mode=mode, aic_linear=lin.aic, aic_step=st.aic, aic_step_plus_linear=both.aic,
                         lr_p_step_given_linear=stats.chi2.sf(lr, 1), step_coef=both.params[1],
                         linear_coef_alone=lin.params[1], linear_p_clustered=lc.pvalues[1] if lc is not None else np.nan))
    d = pd.DataFrame(rows)
    if "lr_p_step_given_linear" in d:
        d["lr_p_holm"] = holm(d.lr_p_step_given_linear)
        d["verdict"] = np.where((d.lr_p_holm < 0.05) & (d.aic_step < d.aic_linear), "H1 supported (step)",
                        np.where((d.aic_linear <= d.aic_step) & (d.linear_p_clustered < 0.05), "H1 rejected (gradual)",
                                 "inconclusive or no decline"))
    return d


def position(df):
    d0 = df[(df.exp == "E2") & (df.variant == "P1") & (df.kind == "main")]
    prof, quad = [], []
    for (m, mode), g in d0.groupby(KEYS):
        M = wide(g, "joint_ok", by="slot")
        if M.shape[1] != 5:
            continue
        A, bm = M.values, boot(M.values)
        lo, hi = ci(bm)
        for j, k in enumerate(M.columns):
            prof.append(dict(model=m, mode=mode, slot=k, acc=A[:, j].mean(), lo=lo[j], hi=hi[j]))
        gap_ci = ci(bm.max(1) - bm.min(1))
        x = (g.slot.values - 3.0)
        r = fit(g.joint_ok.values.astype(float), np.c_[np.ones_like(x), x, x ** 2], pd.factorize(g.target)[0])
        quad.append(dict(model=m, mode=mode, position_gap=A.mean(0).max() - A.mean(0).min(), gap_lo=gap_ci[0],
                         gap_hi=gap_ci[1], quad_coef=r.params[2] if r is not None else np.nan,
                         quad_p=r.pvalues[2] if r is not None else np.nan))
    q = pd.DataFrame(quad)
    if not q.empty:
        q["quad_p_holm"] = holm(q.quad_p)
        q["verdict"] = np.where((q.quad_coef > 0) & (q.quad_p_holm < 0.05), "H2 supported (U-shape)", "H2 not supported")
    return pd.DataFrame(prof), q


def errors(core):
    """Descriptive error composition by N (shares of wrong item/country answers)."""
    rows = []
    for (m, mode), g in core[core.N >= 2].groupby(KEYS):
        long = pd.concat([g[["target", "N", "item_err"]].rename(columns={"item_err": "err"}),
                          g[["target", "N", "country_err"]].rename(columns={"country_err": "err"})]).dropna()
        for n, h in long.groupby("N"):
            rows.append(dict(model=m, mode=mode, N=n, wrong_fields=len(h), **{k: (h.err == k).mean() for k in
                             ("mis_binding", "prior_drift", "abstain", "format_error")}))
    return pd.DataFrame(rows)


def attraction(core):
    """H3a. Among wrong answers that name one of the 5 images of the set, what share name a SHOWN distractor?
    If a wrong choice were random among the 4 non-target options, that share would be (N-1)/4.
    Only meaningful in multiple-choice mode, where all 5 images' names are always on the page."""
    rows = []
    for (m, mode), g in core[(core.N >= 2) & (core.N <= 4)].groupby(KEYS):
        for n, h in g.groupby("N"):
            recs = []
            for f in ("item", "country"):
                w = h[(h[f"{f}_ok"] == 0) & h[f"{f}_src"].notna()]
                recs.append(w.assign(shown=w[f"{f}_err"].eq("mis_binding").astype(int)))
            w = pd.concat(recs)
            if w.empty:
                continue
            per = w.groupby("target").agg(shown=("shown", "sum"), named=("shown", "size"))
            S, K = per.shown.values.astype(float), per.named.values.astype(float)
            idx = RNG.integers(0, len(per), (B, len(per)))
            bs = S[idx].sum(1) / np.maximum(K[idx].sum(1), 1)
            exp = (n - 1) / 4
            lo, hi = ci(bs)
            rows.append(dict(model=m, mode=mode, N=n, wrong_naming_an_image=int(K.sum()), share_shown=S.sum() / K.sum(),
                             expected_if_random=exp, excess=S.sum() / K.sum() - exp, excess_lo=lo - exp, excess_hi=hi - exp))
    d = pd.DataFrame(rows)
    if not d.empty:
        ok = d.groupby(KEYS).excess_lo.transform("min") > 0
        d["verdict"] = np.where(d["mode"] == "mc", np.where(ok, "H3a supported (all N)", "H3a not supported at every N"),
                                "descriptive only (open mode)")
    return d


def near_far(df):
    d0 = df[(df.exp == "E3") & (df.kind == "main")]
    rows, reg = [], []
    clip = pd.read_csv("clip_sim.csv") if Path("clip_sim.csv").exists() else None
    for (m, mode), g in d0.groupby(KEYS):
        g = g.assign(mis=(g.item_err.eq("mis_binding").astype(int) + g.country_err.eq("mis_binding").astype(int)) / 2)
        for n, h in g.groupby("N"):
            for col, nm in (("joint_ok", "joint_acc"), ("mis", "mis_binding_rate")):
                W = h.groupby(["target", "list"])[col].mean().unstack("list").dropna()
                if W.empty:
                    continue
                A = W[["near", "far"]].values
                bm = boot(A[:, [1]] - A[:, [0]])[:, 0]
                lo, hi = ci(bm)
                rows.append(dict(model=m, mode=mode, N=n, metric=nm, near=A[:, 0].mean(), far=A[:, 1].mean(),
                                 far_minus_near=(A[:, 1] - A[:, 0]).mean(), lo=lo, hi=hi, targets=len(A)))
        near = (g.list == "near").astype(float).values
        X = pd.DataFrame({"const": 1.0, "near": near, "N5": (g.N == 5).astype(float).values})
        if clip is not None:
            X = X.join(g[["target", "list"]].reset_index(drop=True).merge(clip, on=["target", "list"], how="left")["mean_sim"])
            X = X.fillna(X.mean())
        r = fit(g.joint_ok.values.astype(float), X.values, pd.factorize(g.target)[0])
        if r is not None:
            reg.append(dict(model=m, mode=mode, near_coef=r.params[1], near_p=r.pvalues[1],
                            odds_ratio=np.exp(r.params[1]), clip_adjusted=clip is not None))
    reg = pd.DataFrame(reg)
    if not reg.empty:
        reg["near_p_holm"] = holm(reg.near_p)
        reg["verdict"] = np.where((reg.near_coef < 0) & (reg.near_p_holm < 0.05), "H3b supported (near hurts)", "H3b not supported")
    return pd.DataFrame(rows), reg


def icr(core):
    """Illusory Conjunction Rate vs an independence baseline.
    Baseline: shuffle the country answers among trials with the same target slot (so each field keeps its own
    error pattern but the two fields are made independent), then recompute the rate among wrong joint answers."""
    rows = []
    rng = np.random.default_rng(1)

    def rate(sl, i, c):
        wrong = ~((i == sl) & (c == sl))
        flag = (i > 0) & (c > 0) & (i != c) & wrong
        return flag.sum() / wrong.sum() if wrong.sum() else np.nan

    for (m, mode), g in core[core.N >= 2].groupby(KEYS):
        for n, h in g.groupby("N"):
            sl, i, c = h.slot.values, h.item_slot.values, h.country_slot.values
            wrong = ~((i == sl) & (c == sl))
            if wrong.sum() < 5:
                continue
            obs = rate(sl, i, c)
            groups = [np.where(sl == k)[0] for k in np.unique(sl)]
            perm = []
            for _ in range(PERMS):
                cp = c.copy()
                for ix in groups:
                    cp[ix] = rng.permutation(c[ix])
                perm.append(rate(sl, i, cp))
            perm = np.array(perm)
            rows.append(dict(model=m, mode=mode, N=n, wrong_joint=int(wrong.sum()), icr_observed=obs,
                             icr_independence_baseline=perm.mean(), excess=obs - perm.mean(),
                             p_perm=(1 + (perm >= obs).sum()) / (PERMS + 1)))
    d = pd.DataFrame(rows)
    if not d.empty:
        d["p_holm"] = holm(d.p_perm)
    return d


def tiers(core):
    rows = []
    for (m, mode, t), g in core.groupby(KEYS + ["tier"]):
        n1, n5 = g[g.N == 1], g[g.N == 5]
        long = pd.concat([g[g.N >= 2].item_err, g[g.N >= 2].country_err]).dropna()
        rows.append(dict(model=m, mode=mode, tier=t, acc_N1=n1.joint_ok.mean(), acc_N5=n5.joint_ok.mean(),
                         prior_drift_share=(long == "prior_drift").mean() if len(long) else np.nan))
    return pd.DataFrame(rows)


def paraphrase(df, bal):
    e5 = df[(df.exp == "E5") & (df.kind == "main")]
    if e5.empty:
        return pd.DataFrame()
    tg = set(e5.target)
    base = bal[bal.target.isin(tg)].assign(variant="P1")
    d = pd.concat([base, e5])
    t = d.groupby(KEYS + ["variant", "N"]).joint_ok.mean().unstack("N").reset_index()
    return t


def leak(df):
    d = df[df.kind == "leak"]
    return d.groupby(KEYS).agg(trials=("joint_ok", "size"), item_correct=("item_ok", "mean"),
                               country_correct=("country_ok", "mean")).reset_index()


def describe(df, bal):
    """E6 describe-then-answer (P4) vs the P1 baseline on the same targets: does verbalizing first rescue accuracy?"""
    e6 = df[(df.exp == "E6") & (df.kind == "main")]
    if e6.empty:
        return pd.DataFrame()
    tg = set(e6.target)
    p1 = bal[bal.target.isin(tg)]
    rows = []
    for (m, mode), g6 in e6.groupby(KEYS):
        g1 = p1[(p1["model"] == m) & (p1["mode"] == mode)]   # brackets: df.mode is a method, not the column
        for n in list(g6.N.unique()) + ["all"]:
            h6, h1 = (g6, g1) if n == "all" else (g6[g6.N == n], g1[g1.N == n])
            both = pd.concat([h6.groupby("target").joint_ok.mean(),
                              h1.groupby("target").joint_ok.mean()], axis=1, keys=["p4", "p1"]).dropna()
            if len(both) < 5:
                continue
            bm = boot(both.values)
            lo, hi = ci(bm[:, 0] - bm[:, 1])
            rows.append(dict(model=m, mode=mode, N=n, targets=len(both), p1_acc=both.p1.mean(),
                             p4_acc=both.p4.mean(), rescue=both.p4.mean() - both.p1.mean(), lo=lo, hi=hi))
    d = pd.DataFrame(rows)
    if not d.empty:
        d["verdict"] = np.where(d.lo > 0, "rescues accuracy", np.where(d.hi < 0, "hurts", "no clear effect"))
    return d


def attention(df, models):
    """LVR probe. Among mis-binding errors, is the named (wrong) image also the one the answer tokens
    attend to most? Reads attn_<model>.jsonl (run_models.py --attn); needs scored_<model>.csv for labels."""
    rows = []
    for m in models:
        p = Path(f"attn_{m}.jsonl")
        if not p.exists():
            continue
        att = {}
        for line in open(p, encoding="utf-8"):
            r = json.loads(line)
            att[(r["trial_id"], r["mode"])] = r["attn"]
        sc = df[(df["model"] == m) & (df.kind == "main") & (df.N >= 2)]
        for (mode, n), g in sc.groupby(["mode", "N"]):
            recs = []
            for r in g.itertuples():
                a = att.get((r.trial_id, mode))
                if not a or len(a) != r.N:
                    continue
                for f in ("item", "country"):
                    slot, err = getattr(r, f"{f}_slot"), getattr(r, f"{f}_err")
                    if err == "mis_binding" and slot >= 1:
                        recs.append((r.target, int(np.argmax(a) == slot - 1)))
            if len(recs) < 5:
                continue
            per = pd.DataFrame(recs, columns=["target", "hit"]).groupby("target").hit.mean().values
            lo, hi = ci(boot(per.reshape(-1, 1))[:, 0])
            rows.append(dict(model=m, mode=mode, N=n, mis_binding_fields=len(recs),
                             share_argmax_is_named=float(np.mean([x[1] for x in recs])),
                             chance=1.0 / n, lo=lo, hi=hi))
    d = pd.DataFrame(rows)
    if not d.empty:
        d["verdict"] = np.where(d.lo > d.chance, "attention follows the named image", "no attentional capture")
    return d


LOCUS_ROWS = [
    ("Step in accuracy at N=1 -> 2 (H1)", "flat (each image encoded independently)",
     "sharp step (fixed-capacity state overloaded by a second item)", "gradual decline (longer transcripts)"),
    ("Middle slots worse than end slots at N=5 (H2)", "no slot effect",
     "U-shape (primacy/recency of a serial state)", "possible (lost-in-the-middle in text) - weak test"),
    ("Near > far confusion, CLIP controlled (H3b)", "no (visual similarity explains it)",
     "yes (cultural features co-stored in the state)", "yes (language co-occurrence priors) - weak test"),
    ("Illusory conjunctions above independence baseline", "no (binding fixed per object)",
     "yes (attributes detach in a crowded state)", "no (whole answer copied from one verbalized image)"),
    ("Describe-then-answer rescues accuracy (E6)", "no",
     "partial (offloading to text may help)", "yes (converts the task to its native medium)"),
    ("Answer attention peaks on the named wrong image", "no (attention stays on the target)",
     "yes (selection happened; the binding was wrong)", "no commitment"),
]


def locus(tables):
    """Locus-discrimination table: which representation the observed signatures point to.
    Predictions are fixed in LOCUS_ROWS (pre-registered in README.md); only the Observed column is data."""
    def obs(name, col="verdict"):
        d = tables.get(name)
        if d is None or d.empty or col not in d:
            return "pending"
        return "; ".join(f"{r.model}/{r.mode}: {getattr(r, col)}" for r in d.itertuples())

    icr = tables.get("illusory_conjunctions")
    if icr is None or icr.empty:
        icr_obs = "pending"
    else:
        g = icr.groupby(KEYS).agg(excess=("excess", "mean"), p=("p_holm", "min")).reset_index()
        icr_obs = "; ".join(f"{r.model}/{r.mode}: " + ("above" if (r.excess > 0 and r.p < 0.05) else "at/below")
                            + " baseline" for r in g.itertuples())
    dc = tables.get("describe_then_answer")
    if dc is None or dc.empty:
        dc_obs = "pending"
    else:
        pooled = dc[dc.N == "all"]
        dc_obs = "; ".join(f"{r.model}/{r.mode}: {r.verdict} ({r.rescue:+.2f})" for r in pooled.itertuples()) or "pending"
    observed = [obs("h1_scaling_fit"), obs("h2_position"), obs("h3_near_regression"), icr_obs, dc_obs,
                obs("attention_capture")]
    lines = ["# Locus discrimination: where does image-attribute binding live?\n",
             "Predictions are pre-registered (README.md); only the Observed column comes from the data.\n",
             "| Signature | Encoder-bound | Latent scene state | Text-chain readout | Observed |",
             "| --- | --- | --- | --- | --- |"]
    for (sig, e, l, t), o in zip(LOCUS_ROWS, observed):
        lines.append(f"| {sig} | {e} | {l} | {t} | {o} |")
    return "\n".join(lines) + "\n"


def main(models):
    df = load(models)
    core, bal = core_sets(df)
    acc, summ = scaling(core)
    h1d = h1(bal)
    prof, quad = position(df)
    err, att = errors(core), attraction(core)
    nf, nfreg = near_far(df)
    ic = icr(core)
    tr, pp, lk = tiers(core), paraphrase(df, bal), leak(df)
    dc, at = describe(df, bal), attention(df, models)
    tables = {"acc_by_N": acc, "scaling_summary": summ, "h1_scaling_fit": h1d, "position_profile": prof,
              "h2_position": quad, "error_shares_by_N": err, "h3a_context_attraction": att, "near_vs_far": nf,
              "h3_near_regression": nfreg, "illusory_conjunctions": ic, "by_popularity_tier": tr,
              "paraphrase_check": pp, "text_only_leak": lk, "describe_then_answer": dc, "attention_capture": at}
    for k, v in tables.items():
        v.to_csv(OUT / f"{k}.csv", index=False)
    loc = locus(tables)
    (OUT / "locus_discrimination.md").write_text(loc)
    parts = ["# CultureHaystack results\n", f"Models: {', '.join(models)}\n"]
    for k, v in tables.items():
        parts += [f"\n## {k}\n", md(v)]
    (OUT / "summary.md").write_text("\n".join(parts))
    print("\n".join(parts[:2]))
    for k in ("scaling_summary", "h1_scaling_fit", "h2_position", "h3a_context_attraction", "h3_near_regression", "illusory_conjunctions"):
        print(f"\n## {k}\n" + md(tables[k]))
    print("\n" + loc)
    print(f"all tables written to {OUT}/")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1:])