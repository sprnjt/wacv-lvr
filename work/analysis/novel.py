#!/usr/bin/env python3
"""Exploratory / post-hoc analyses of the frozen CultureHaystack main study (CPU only, read-only on inputs).

  cd <repo> && .venv/bin/python work/analysis/novel.py

Reads scored_<model>.csv, trials_main.jsonl, attn_<model>.jsonl, data/targets.csv, data/pool.csv.
Writes work/analysis/out/*.csv and out/novel_log.txt. Every result here is POST-HOC / EXPLORATORY.
Uncertainty: percentile bootstrap over targets (B=2000), like analyze.py. Holm within each family.
Scope matches analyze.py: kind=main, variant=P1, list=mixed, E1 (N=1..4) + E2 (N=5, all slots), MC mode
unless stated. "bal" = E1 + the E2 slot that continues the E1 rotation (one trial per target per N).
"""
import json
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "work/analysis/out"
OUT.mkdir(parents=True, exist_ok=True)
MODELS = ["gemma4", "qwen3vl", "qwen25vl", "gemma3_4b", "granite4v", "lfm25v"]
SMALL = {"gemma3_4b", "granite4v", "lfm25v"}
ATTN_MODELS = ["qwen3vl", "gemma4", "granite4v"]
L = "ABCDE"
B = 2000
RNG = np.random.default_rng(2026)
LOG = []


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    LOG.append(s)


def holm(p):
    p = np.asarray(p, float)
    out = np.full(p.shape, np.nan)
    ok = ~np.isnan(p)
    if ok.any():
        out[ok] = multipletests(p[ok], method="holm")[1]
    return out


def boot_rows(n, b=B):
    return RNG.integers(0, n, (b, n))


def ci(x, axis=0):
    return np.nanpercentile(x, [2.5, 97.5], axis=axis)


def boot_p(bs, null=0.0):
    """Two-sided percentile-bootstrap p for H0: stat == null."""
    bs = np.asarray(bs)
    return min(1.0, 2 * min((bs <= null).mean(), (bs >= null).mean()) + 1 / len(bs))


def slope_mat(Y, x):
    x = np.asarray(x, float)
    xc = x - x.mean()
    return (Y - Y.mean(axis=-1, keepdims=True)) @ xc / (xc @ xc)


# ---------------------------------------------------------------- load
def load():
    df = pd.concat([pd.read_csv(ROOT / f"scored_{m}.csv") for m in MODELS], ignore_index=True)
    tr = {}
    for line in open(ROOT / "trials_main.jsonl"):
        r = json.loads(line)
        tr[r["trial_id"]] = r
    pool = pd.read_csv(ROOT / "data/pool.csv").set_index("image_id")
    tg = pd.read_csv(ROOT / "data/targets.csv")[["image_id", "idx", "region"]].rename(
        columns={"image_id": "target", "region": "t_region"})
    df = df.merge(tg, on="target", how="left")
    df["size"] = np.where(df.model.isin(SMALL), "<=4B", ">=7B")
    # gold / chosen letters (MC) and the chosen image's slot
    gi, gc, li, lc = [], [], [], []
    for r in df.itertuples():
        t = tr.get(r.trial_id)
        if t is None:
            gi.append(None); gc.append(None); li.append(None); lc.append(None)
            continue
        gi.append(L[t["item_options"].index(pool.at[r.target, "item"])])
        gc.append(L[t["country_options"].index(pool.at[r.target, "country"])])
        li.append(L[t["item_options"].index(r.item_ans)] if r.mode == "mc" and r.item_ans in t["item_options"] else None)
        lc.append(L[t["country_options"].index(r.country_ans)] if r.mode == "mc" and r.country_ans in t["country_options"] else None)
    df["gold_item_let"], df["gold_country_let"], df["item_let"], df["country_let"] = gi, gc, li, lc
    return df, tr, pool


def core_sets(df):
    ok = (df.kind == "main") & (df.variant == "P1") & (df.list == "mixed") & df.exp.isin(["E1", "E2"])
    core = df[ok]
    bal = core[(core.exp == "E1") | (core.slot == core.idx % 5 + 1)]
    return core, bal


def wide(g, col, by="N"):
    return g.groupby(["target", by])[col].mean().unstack(by).dropna()


# ---------------------------------------------------------------- 1. per-image cost (headline table)
def per_image_cost(core):
    rows = []
    mc = core[core["mode"] == "mc"]
    for m, g in mc.groupby("model"):
        mats = {f: wide(g, f"{f}_ok").values for f in ("joint", "item", "country")}
        idx = boot_rows(len(mats["joint"]))
        rec = dict(model=m, size="<=4B" if m in SMALL else ">=7B")
        for f, A in mats.items():
            s15, s15b = slope_mat(A.mean(0), [1, 2, 3, 4, 5]), slope_mat(A[idx].mean(1), [1, 2, 3, 4, 5])
            d, db = A[:, 0].mean() - A[:, 4].mean(), A[idx][:, :, 0].mean(1) - A[idx][:, :, 4].mean(1)
            lo, hi = ci(s15b)
            dlo, dhi = ci(db)
            rec.update({f"{f}_acc1": A[:, 0].mean(), f"{f}_acc5": A[:, 4].mean(), f"{f}_slope": s15, f"{f}_slope_lo": lo,
                        f"{f}_slope_hi": hi, f"{f}_slope_p": boot_p(s15b), f"{f}_drop15": d, f"{f}_drop_lo": dlo, f"{f}_drop_hi": dhi})
            if f == "item":
                item_b = s15b
            if f == "country":
                diff = s15b - item_b
                rec.update(country_minus_item_slope=s15 - rec["item_slope"], cmi_lo=ci(diff)[0], cmi_hi=ci(diff)[1],
                           cmi_p=boot_p(diff))
        rows.append(rec)
    d = pd.DataFrame(rows)
    d["joint_slope_p_holm"] = holm(d.joint_slope_p)
    d["cmi_p_holm"] = holm(d.cmi_p)
    # pooled by size class: average the per-target curves of the 3 models in the class
    pooled = []
    for sz, g in mc.groupby("size"):
        A = g.groupby(["target", "N"]).joint_ok.mean().unstack("N").dropna().values
        idx = boot_rows(len(A))
        sb = slope_mat(A[idx].mean(1), [1, 2, 3, 4, 5])
        pooled.append(dict(size=sz, slope=slope_mat(A.mean(0), [1, 2, 3, 4, 5]), lo=ci(sb)[0], hi=ci(sb)[1]))
    pooled = pd.DataFrame(pooled)
    # class contrast
    Ab = {sz: g.groupby(["target", "N"]).joint_ok.mean().unstack("N").sort_index() for sz, g in mc.groupby("size")}
    A1, A2 = Ab["<=4B"].values, Ab[">=7B"].values
    idx = boot_rows(len(A1))
    db = slope_mat(A1[idx].mean(1), [1, 2, 3, 4, 5]) - slope_mat(A2[idx].mean(1), [1, 2, 3, 4, 5])
    pooled["class_diff_small_minus_large"] = slope_mat(A1.mean(0), [1, 2, 3, 4, 5]) - slope_mat(A2.mean(0), [1, 2, 3, 4, 5])
    pooled["class_diff_lo"], pooled["class_diff_hi"] = ci(db)
    d.to_csv(OUT / "f1_per_image_cost.csv", index=False)
    pooled.to_csv(OUT / "f1_per_image_cost_pooled.csv", index=False)
    log("\n## 1. Per-image cost (MC, slope N=1..5, per added image)")
    log(d[["model", "joint_slope", "joint_slope_lo", "joint_slope_hi", "item_slope", "country_slope",
           "country_minus_item_slope", "cmi_lo", "cmi_hi", "cmi_p_holm"]].round(4).to_string(index=False))
    log(pooled.round(4).to_string(index=False))
    return d


# ---------------------------------------------------------------- 2. binding-conditional accuracy
def conditional_on_known(core, bal):
    """Acc@N restricted to targets the model got jointly right at N=1 (so recognition is known to succeed).
    Also: the share of N=5 errors that fall on targets known at N=1 ('binding-induced' errors)."""
    rows = []
    mc = core[core["mode"] == "mc"]
    for m, g in mc.groupby("model"):
        W = wide(g, "joint_ok")
        known = W[W[1] == 1]
        A = known.values
        idx = boot_rows(len(A))
        bm = A[idx].mean(1)
        lo, hi = ci(bm)
        sb = slope_mat(bm, [1, 2, 3, 4, 5])
        unk = W[W[1] == 0].values
        err5_known = ((1 - known[5]).sum()) / ((1 - W[5]).sum())
        for j, n in enumerate(W.columns):
            rows.append(dict(model=m, N=n, targets_known=len(A), acc_given_known=A[:, j].mean(), lo=lo[j], hi=hi[j],
                             acc_given_unknown=unk[:, j].mean() if len(unk) else np.nan,
                             slope_known=slope_mat(A.mean(0), [1, 2, 3, 4, 5]), slope_known_lo=ci(sb)[0],
                             slope_known_hi=ci(sb)[1], share_N5_errors_on_known=err5_known))
    d = pd.DataFrame(rows)
    d.to_csv(OUT / "f2_acc_given_known_at_N1.csv", index=False)
    log("\n## 2. Accuracy at N given joint-correct at N=1 (MC)")
    log(d.pivot(index="model", columns="N", values="acc_given_known").round(3).to_string())
    log(d.groupby("model")[["targets_known", "slope_known", "slope_known_lo", "slope_known_hi",
                            "share_N5_errors_on_known"]].first().round(3).to_string())
    log("acc on targets WRONG at N=1:")
    log(d.pivot(index="model", columns="N", values="acc_given_unknown").round(3).to_string())
    return d


# ---------------------------------------------------------------- 3. nested-haystack survival
def survival(bal):
    """bal: one trial per target per N, nested haystacks. P(correct at N+1 | correct at N) and the converse;
    'ever-survive' curve S(n) = share of targets correct at every N <= n, against the independence expectation."""
    rows = []
    mc = bal[bal["mode"] == "mc"]
    for m, g in mc.groupby("model"):
        A = wide(g, "joint_ok").values  # targets x 5
        idx = boot_rows(len(A))
        for n in range(1, 5):
            def stay(X):
                c = X[..., n - 1] == 1
                return (X[..., n] * c).sum(-1) / np.maximum(c.sum(-1), 1)

            def recover(X):
                c = X[..., n - 1] == 0
                return (X[..., n] * c).sum(-1) / np.maximum(c.sum(-1), 1)
            sb, rb = stay(A[idx]), recover(A[idx])
            rows.append(dict(model=m, step=f"{n}->{n + 1}", p_stay=stay(A), stay_lo=ci(sb)[0], stay_hi=ci(sb)[1],
                             p_recover=recover(A), rec_lo=ci(rb)[0], rec_hi=ci(rb)[1], marg_next=A[:, n].mean()))
        # among targets KNOWN at N=1: P(recover at N+1 | binding failure at N), pooled over N=2..4, vs marginal acc
        K = A[A[:, 0] == 1]
        kix = boot_rows(len(K))

        def rec_known(X):
            w = X[..., 1:4] == 0
            return (X[..., 2:5] * w).sum((-1, -2)) / np.maximum(w.sum((-1, -2)), 1)
        rb = rec_known(K[kix])
        mb = K[kix][..., 2:].mean((-1, -2))
        rows.append(dict(model=m, step="known: recover|fail", p_recover=rec_known(K), rec_lo=ci(rb)[0], rec_hi=ci(rb)[1],
                         marg_next=K[:, 2:].mean(), rec_minus_marg_lo=ci(rb - mb)[0], rec_minus_marg_hi=ci(rb - mb)[1],
                         n_fail=int((K[:, 1:4] == 0).sum())))
        S = np.cumprod(A, axis=1).mean(0)
        Sind = np.cumprod(A.mean(0))
        for n in range(5):
            rows.append(dict(model=m, step=f"S({n + 1})", p_stay=S[n], marg_next=Sind[n]))
    d = pd.DataFrame(rows)
    d.to_csv(OUT / "f3_nested_survival.csv", index=False)
    log("\n## 3. Nested survival (MC joint, balanced-slot chain). p_stay = P(ok@N+1 | ok@N); p_recover = P(ok@N+1 | wrong@N)")
    log(d[d.step.str.contains("->")].round(3).to_string(index=False))
    log("S(n) = share correct at every N<=n (p_stay col) vs independence product (marg_next col)")
    log(d[d.step.str.startswith("S")].pivot(index="model", columns="step", values=["p_stay", "marg_next"]).round(3).to_string())
    return d


# ---------------------------------------------------------------- 4. category
def category(core):
    rows, tests = [], []
    mc = core[core["mode"] == "mc"]
    for (m, c), g in mc.groupby(["model", "category"]):
        A = wide(g, "joint_ok").values
        idx = boot_rows(len(A))
        sb = slope_mat(A[idx].mean(1), [1, 2, 3, 4, 5])
        rows.append(dict(model=m, category=c, targets=len(A), acc1=A[:, 0].mean(), acc5=A[:, 4].mean(),
                         slope=slope_mat(A.mean(0), [1, 2, 3, 4, 5]), lo=ci(sb)[0], hi=ci(sb)[1]))
    # heterogeneity test: logistic GLM joint ~ C(category) * N, target-clustered; Wald test on the 3 interaction terms
    for m, g in mc.groupby("model"):
        X = pd.get_dummies(g.category, drop_first=True).astype(float)
        N = g.N.astype(float).values - 3
        Xm = np.c_[np.ones(len(g)), N, X.values, X.values * N[:, None]]
        r = sm.GLM(g.joint_ok.values.astype(float), Xm, family=sm.families.Binomial()).fit(
            cov_type="cluster", cov_kwds={"groups": pd.factorize(g.target)[0]})
        k = X.shape[1]
        R = np.zeros((k, Xm.shape[1]))
        R[:, 2 + k:] = np.eye(k)
        w = r.wald_test(R, scalar=True)
        tests.append(dict(model=m, wald_chi2=float(w.statistic), df=k, p=float(w.pvalue)))
    d, t = pd.DataFrame(rows), pd.DataFrame(tests)
    t["p_holm"] = holm(t.p)
    # pooled within size class
    pooled = []
    for (sz, c), g in mc.groupby(["size", "category"]):
        A = g.groupby(["target", "N"]).joint_ok.mean().unstack("N").dropna().values
        idx = boot_rows(len(A))
        sb = slope_mat(A[idx].mean(1), [1, 2, 3, 4, 5])
        pooled.append(dict(size=sz, category=c, acc1=A[:, 0].mean(), acc5=A[:, 4].mean(),
                           slope=slope_mat(A.mean(0), [1, 2, 3, 4, 5]), lo=ci(sb)[0], hi=ci(sb)[1]))
    pooled = pd.DataFrame(pooled)
    d.to_csv(OUT / "f4_category_slopes.csv", index=False)
    t.to_csv(OUT / "f4_category_heterogeneity.csv", index=False)
    pooled.to_csv(OUT / "f4_category_slopes_pooled.csv", index=False)
    log("\n## 4. Category slopes (MC joint, N=1..5)")
    log(d.pivot(index="model", columns="category", values="slope").round(3).to_string())
    log(t.round(4).to_string(index=False))
    log(pooled.round(3).to_string(index=False))
    return d


# ---------------------------------------------------------------- 5. partial binding
def partial(core):
    rows = []
    mc = core[(core["mode"] == "mc") & (core.N >= 1)]
    for m, g in mc.groupby("model"):
        g = g.assign(io_cw=((g.item_ok == 1) & (g.country_ok == 0)).astype(int),
                     iw_co=((g.item_ok == 0) & (g.country_ok == 1)).astype(int),
                     both_w=((g.item_ok == 0) & (g.country_ok == 0)).astype(int))
        for col in ("io_cw", "iw_co", "both_w"):
            A = wide(g, col).values
            idx = boot_rows(len(A))
            bm = A[idx].mean(1)
            lo, hi = ci(bm)
            sb = slope_mat(bm, [1, 2, 3, 4, 5])
            for j in range(5):
                rows.append(dict(model=m, cell=col, N=j + 1, rate=A[:, j].mean(), lo=lo[j], hi=hi[j],
                                 slope=slope_mat(A.mean(0), [1, 2, 3, 4, 5]), slope_lo=ci(sb)[0], slope_hi=ci(sb)[1]))
    d = pd.DataFrame(rows)
    d.to_csv(OUT / "f5_partial_binding.csv", index=False)
    log("\n## 5. Partial binding cells (MC). io_cw = item right/country wrong; iw_co = item wrong/country right")
    log(d.pivot_table(index=["model", "cell"], columns="N", values="rate").round(3).to_string())
    log(d.groupby(["model", "cell"])[["slope", "slope_lo", "slope_hi"]].first().round(4).to_string())
    return d


# ---------------------------------------------------------------- 6. slot profile at every N
def slots(core):
    """Primacy (slot 1) and recency (slot N) vs the interior, at each N. N<5: between targets (slot = idx%N+1);
    N=5: within target (E2). Relative position r=(k-1)/(N-1)."""
    rows = []
    mc = core[core["mode"] == "mc"]
    for (m, n), g in mc.groupby(["model", "N"]):
        if n == 1:
            continue
        for k, h in g.groupby("slot"):
            a = h.groupby("target").joint_ok.mean().values
            idx = boot_rows(len(a))
            lo, hi = ci(a[idx].mean(1))
            rows.append(dict(model=m, size="<=4B" if m in SMALL else ">=7B", N=n, slot=k, rel=(k - 1) / (n - 1),
                             acc=a.mean(), lo=lo, hi=hi, trials=len(a)))
    d = pd.DataFrame(rows)
    # last-minus-first, and ends-minus-middle (N>=3) per model x N, bootstrap over targets
    tests = []
    for (m, n), g in mc[mc.N >= 2].groupby(["model", "N"]):
        g = g.assign(pos=np.where(g.slot == 1, "first", np.where(g.slot == n, "last", "mid")))
        per = g.groupby(["target", "pos"]).joint_ok.mean().unstack("pos")
        cols = {c: per[c].dropna().values for c in per.columns}

        def bmean(a):
            return a[boot_rows(len(a))].mean(1)
        bf, bl = bmean(cols["first"]), bmean(cols["last"])
        rec = dict(model=m, N=n, first=cols["first"].mean(), last=cols["last"].mean(), last_minus_first=cols["last"].mean() - cols["first"].mean(),
                   lmf_lo=ci(bl - bf)[0], lmf_hi=ci(bl - bf)[1], lmf_p=boot_p(bl - bf))
        if "mid" in cols:
            bmid = bmean(cols["mid"])
            e = (bf + bl) / 2 - bmid
            rec.update(mid=cols["mid"].mean(), ends_minus_mid=(cols["first"].mean() + cols["last"].mean()) / 2 - cols["mid"].mean(),
                       emm_lo=ci(e)[0], emm_hi=ci(e)[1], emm_p=boot_p(e))
        tests.append(rec)
    t = pd.DataFrame(tests)
    t["lmf_p_holm"] = holm(t.lmf_p)
    t["emm_p_holm"] = holm(t.emm_p)
    d.to_csv(OUT / "f6_slot_profile_all_N.csv", index=False)
    t.to_csv(OUT / "f6_primacy_recency_tests.csv", index=False)
    log("\n## 6. Slot effects at each N (MC joint). N<5 between-target; N=5 within-target")
    log(t.round(3).to_string(index=False))
    return d, t


# ---------------------------------------------------------------- 7. which wrong image is picked (N=5 MC)
def wrong_pick(core, pool):
    """For every MC mis-binding field at N=5 (E2), the chosen wrong image is at slot j != k (target slot).
    Null: uniform over the 4 shown distractors (MC lists all 5 set members at N=5, so availability is equal).
    Features: |j-k|=1 (adjacent), j>k (later), j==1, j==5, same region as target, letter adjacency |L_j-L_gold|=1,
    chosen letter == 'A'. Report observed - expected with target-cluster bootstrap; conditional logit for joint effects."""
    e2 = core[(core.exp == "E2") & (core["mode"] == "mc")]
    recs = []
    for r in e2.itertuples():
        order = None
        for f in ("item", "country"):
            if getattr(r, f"{f}_err") != "mis_binding":
                continue
            j, k = int(getattr(r, f"{f}_slot")), int(r.slot)
            recs.append(dict(model=r.model, target=r.target, field=f, k=k, j=j, src=getattr(r, f"{f}_src"),
                             gold_let=getattr(r, f"gold_{f}_let"), let=getattr(r, f"{f}_let"), trial_id=r.trial_id))
    ev = pd.DataFrame(recs)
    # per-event expected values under uniform choice among the 4 other slots
    trials = {json.loads(l)["trial_id"]: json.loads(l) for l in open(ROOT / "trials_main.jsonl")}
    rows = []
    feats = []
    for e in ev.itertuples():
        t = trials[e.trial_id]
        opts = t[f"{e.field}_options"]
        name_col = e.field
        alts = []
        for jj, img in enumerate(t["order"], 1):
            if jj == e.k:
                continue
            letter = L[opts.index(pool.at[img, name_col])]
            alts.append(dict(ev=e.Index, chosen=int(jj == e.j), adj=int(abs(jj - e.k) == 1), dist=abs(jj - e.k),
                             later=int(jj > e.k), first=int(jj == 1), last=int(jj == 5),
                             same_region=int(pool.at[img, "region"] == pool.at[e.target, "region"]),
                             let_adj=int(abs(L.index(letter) - L.index(e.gold_let)) == 1), let_A=int(letter == "A"),
                             let_E=int(letter == "E"), model=e.model, target=e.target, k=e.k, img=img))
        feats += alts
    F = pd.DataFrame(feats)
    obs = F[F.chosen == 1].set_index("ev")
    exp = F.groupby("ev")[["adj", "later", "first", "last", "same_region", "let_adj", "let_A", "let_E", "dist"]].mean()
    for m in MODELS:
        o, x = obs[obs.model == m], exp.loc[obs[obs.model == m].index]
        tgt = o.target.values
        ut = np.unique(tgt)
        groups = [np.where(tgt == u)[0] for u in ut]
        for feat in ["adj", "later", "first", "last", "same_region", "let_adj", "let_A", "let_E", "dist"]:
            diff = o[feat].values - x[feat].values
            per = np.array([diff[gix].sum() for gix in groups])
            cnt = np.array([len(gix) for gix in groups])
            idx = boot_rows(len(ut))
            bs = per[idx].sum(1) / cnt[idx].sum(1)
            rows.append(dict(model=m, feature=feat, events=len(o), observed=o[feat].mean(), expected=x[feat].mean(),
                             excess=diff.mean(), lo=ci(bs)[0], hi=ci(bs)[1], p=boot_p(bs)))
    d = pd.DataFrame(rows)
    d["p_holm"] = holm(d.p)  # one family: all model x feature tests
    # distance distribution observed vs expected
    dist = []
    for m in MODELS:
        o = obs[obs.model == m]
        fe = F[F.ev.isin(o.index)]
        for dd in range(1, 5):
            dist.append(dict(model=m, dist=dd, observed=(o.dist == dd).mean(),
                             expected=fe.assign(x=(fe.dist == dd)).groupby("ev").x.mean().mean()))
    dist = pd.DataFrame(dist)
    # conditional logit: choice among 4 alternatives, all features jointly; bootstrap over targets for CIs
    from statsmodels.discrete.conditional_models import ConditionalLogit
    cl = []
    X_cols = ["adj", "later", "same_region", "let_adj", "let_A"]
    for m in MODELS:
        Fm = F[F.model == m]
        try:
            res = ConditionalLogit(Fm.chosen.values, Fm[X_cols].values.astype(float), groups=Fm.ev.values).fit(disp=0)
        except Exception as ex:
            log("clogit failed", m, ex)
            continue
        ut = Fm.target.unique()
        bys = {u: Fm[Fm.target == u] for u in ut}
        bc = []
        for b in range(300):
            pick = RNG.choice(ut, len(ut))
            parts = []
            for i, u in enumerate(pick):
                p = bys[u].copy()
                p["ev"] = p["ev"].astype(str) + f"_{i}"
                parts.append(p)
            Fb = pd.concat(parts)
            try:
                rb = ConditionalLogit(Fb.chosen.values, Fb[X_cols].values.astype(float), groups=pd.factorize(Fb.ev)[0]).fit(disp=0)
                bc.append(rb.params)
            except Exception:
                pass
        bc = np.array(bc)
        for i, c in enumerate(X_cols):
            cl.append(dict(model=m, feature=c, coef=res.params[i], OR=np.exp(res.params[i]),
                           OR_lo=np.exp(np.percentile(bc[:, i], 2.5)), OR_hi=np.exp(np.percentile(bc[:, i], 97.5)),
                           p_model=res.pvalues[i]))
    cl = pd.DataFrame(cl)
    d.to_csv(OUT / "f7_wrong_pick_features.csv", index=False)
    dist.to_csv(OUT / "f7_wrong_pick_distance.csv", index=False)
    cl.to_csv(OUT / "f7_wrong_pick_clogit.csv", index=False)
    F.to_csv(OUT / "f7_wrong_pick_alternatives.csv", index=False)
    log("\n## 7. Which wrong image is picked (MC N=5 mis-binding fields; null = uniform over 4 shown distractors)")
    log(d.round(3).to_string(index=False))
    log(dist.pivot(index="model", columns="dist", values=["observed", "expected"]).round(3).to_string())
    log(cl.round(3).to_string(index=False))
    return d, dist, cl


# ---------------------------------------------------------------- 8. option-letter bias
def letters(core):
    """Is there a letter preference, does it grow with N, and does it confound the N-slope?
    (a) chosen-letter distribution among wrong MC fields vs the null uniform over the 4 non-gold letters;
    (b) accuracy slope by gold letter; (c) same-letter responding (item letter == country letter)."""
    mc = core[core["mode"] == "mc"]
    rows = []
    for (m, n), g in mc[mc.N >= 2].groupby(["model", "N"]):
        for f in ("item", "country"):
            w = g[(g[f"{f}_ok"] == 0) & g[f"{f}_let"].notna()]
            for x in L:
                elig = w[w[f"gold_{f}_let"] != x]
                rows.append(dict(model=m, N=n, field=f, letter=x, wrong=len(w), share=(w[f"{f}_let"] == x).mean(),
                                 null=(w[f"gold_{f}_let"] != x).mean() / 4))
    d = pd.DataFrame(rows)
    agg = d.groupby(["model", "N", "letter"])[["share", "null"]].mean().reset_index()
    agg["excess"] = agg.share - agg.null
    # chi-square of chosen-letter vs null per model (fields pooled, N>=2)
    chi = []
    for m, g in mc[mc.N >= 2].groupby("model"):
        obs, ex = np.zeros(5), np.zeros(5)
        for f in ("item", "country"):
            w = g[(g[f"{f}_ok"] == 0) & g[f"{f}_let"].notna()]
            for i, x in enumerate(L):
                obs[i] += (w[f"{f}_let"] == x).sum()
                ex[i] += (w[f"gold_{f}_let"] != x).sum() / 4
        st, p = stats.chisquare(obs, ex)
        chi.append(dict(model=m, **{f"obs_{x}": int(o) for x, o in zip(L, obs)}, **{f"exp_{x}": e for x, e in zip(L, ex)},
                        chi2=st, p=p, max_dev_letter=L[int(np.argmax(np.abs(obs - ex) / ex))]))
    chi = pd.DataFrame(chi)
    chi["p_holm"] = holm(chi.p)
    # letter-A share among wrong answers vs N (does fallback to a letter grow with N?)
    trend = []
    for m, g in mc[mc.N >= 2].groupby("model"):
        recs = []
        for f in ("item", "country"):
            w = g[(g[f"{f}_ok"] == 0) & g[f"{f}_let"].notna() & (g[f"gold_{f}_let"] != "A")]
            recs.append(pd.DataFrame(dict(N=w.N.values, A=(w[f"{f}_let"] == "A").astype(int).values, target=w.target.values)))
        R = pd.concat(recs)
        X = np.c_[np.ones(len(R)), R.N.values - 3.5]
        r = sm.GLM(R.A.values.astype(float), X, family=sm.families.Binomial()).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(R.target)[0]})
        trend.append(dict(model=m, wrong_fields=len(R), share_A=R.A.mean(), null=0.25, logit_N_coef=r.params[1], p=r.pvalues[1]))
    trend = pd.DataFrame(trend)
    trend["p_holm"] = holm(trend.p)
    # (b) N-slope by gold letter (item field) -> is the decline concentrated where the gold letter is disfavoured?
    gl = []
    for m, g in mc.groupby("model"):
        for x in L:
            h = g[g.gold_item_let == x]
            A = wide(h, "item_ok").values
            gl.append(dict(model=m, gold_item_letter=x, targets=len(A), acc1=A[:, 0].mean(), acc5=A[:, 4].mean(),
                           slope=slope_mat(A.mean(0), [1, 2, 3, 4, 5])))
    gl = pd.DataFrame(gl)
    het = []
    for m, g in mc.groupby("model"):
        X = pd.get_dummies(g.gold_item_let, drop_first=True).astype(float)
        N = g.N.astype(float).values - 3
        Xm = np.c_[np.ones(len(g)), N, X.values, X.values * N[:, None]]
        r = sm.GLM(g.item_ok.values.astype(float), Xm, family=sm.families.Binomial()).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(g.target)[0]})
        k = X.shape[1]
        R = np.zeros((k, Xm.shape[1]))
        R[:, 2 + k:] = np.eye(k)
        w = r.wald_test(R, scalar=True)
        het.append(dict(model=m, wald_chi2=float(w.statistic), df=k, p=float(w.pvalue)))
    het = pd.DataFrame(het)
    het["p_holm"] = holm(het.p)
    # (c) same-letter responding
    same = []
    for (m, n), g in mc.groupby(["model", "N"]):
        v = g[g.item_let.notna() & g.country_let.notna()]
        gs = v.gold_item_let == v.gold_country_let
        rs = v.item_let == v.country_let
        same.append(dict(model=m, N=n, resp_same=rs.mean(), gold_same=gs.mean(),
                         joint_acc_gold_same=v[gs].joint_ok.mean(), joint_acc_gold_diff=v[~gs].joint_ok.mean(),
                         resp_same_when_wrong_both=(rs[(v.item_ok == 0) & (v.country_ok == 0)]).mean()))
    same = pd.DataFrame(same)
    for name, x in [("f8_letter_shares", agg), ("f8_letter_chisq", chi), ("f8_letterA_trend", trend),
                    ("f8_slope_by_gold_letter", gl), ("f8_gold_letter_heterogeneity", het), ("f8_same_letter", same)]:
        x.to_csv(OUT / f"{name}.csv", index=False)
    log("\n## 8. Option-letter bias (MC, wrong fields, N>=2)")
    log(chi.round(3).to_string(index=False))
    log(trend.round(4).to_string(index=False))
    log(gl.pivot(index="model", columns="gold_item_letter", values="slope").round(3).to_string())
    log(het.round(4).to_string(index=False))
    log(same.round(3).to_string(index=False))
    return chi, trend, same


# ---------------------------------------------------------------- 9. attention: does target mass predict correctness?
def auc(score, y):
    y = np.asarray(y).astype(bool)
    if y.all() or (~y).all():
        return np.nan
    r = stats.rankdata(score)
    n1 = y.sum()
    return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * (~y).sum())


def attention(core):
    rows, slot_rows, dist_rows, recs = [], [], [], []
    e2 = core[core.exp == "E2"]
    for m in ATTN_MODELS:
        att = {}
        for line in open(ROOT / f"attn_{m}.jsonl"):
            r = json.loads(line)
            att[(r["trial_id"], r["mode"])] = np.array(r["attn"], float)
        g = e2[e2.model == m]
        for r in g.itertuples():
            a = att.get((r.trial_id, r.mode))
            if a is None or len(a) != 5:
                continue
            p = a / a.sum()
            k = r.slot - 1
            ent = -(p * np.log(p)).sum() / np.log(5)
            recs.append(dict(model=m, mode=r.mode, target=r.target, slot=r.slot, joint_ok=r.joint_ok, item_ok=r.item_ok,
                             country_ok=r.country_ok, tgt_share=p[k], tgt_is_argmax=int(np.argmax(a) == k),
                             tgt_rank=int((a > a[k]).sum()) + 1, entropy=ent, img_total=a.sum(),
                             **{f"p{j + 1}": p[j] for j in range(5)}))
    A = pd.DataFrame(recs)
    for (m, mode), g in A.groupby(["model", "mode"]):
        ut = g.target.unique()
        gi = {u: np.where(g.target.values == u)[0] for u in ut}
        y, s, e, am = g.joint_ok.values, g.tgt_share.values, g.entropy.values, g.tgt_is_argmax.values
        bs_a, bs_e = [], []
        for b in range(B):
            ix = np.concatenate([gi[u] for u in RNG.choice(ut, len(ut))])
            bs_a.append(auc(s[ix], y[ix]))
            bs_e.append(auc(-e[ix], y[ix]))
        # within-target AUC: target-mean-centred share (removes item-level difficulty / image-size confounds)
        sc = s - g.groupby("target").tgt_share.transform("mean").values
        bs_w = []
        for b in range(B):
            ix = np.concatenate([gi[u] for u in RNG.choice(ut, len(ut))])
            bs_w.append(auc(sc[ix], y[ix]))
        rows.append(dict(model=m, mode=mode, trials=len(g), acc=y.mean(), auc_tgt_share=auc(s, y), auc_lo=np.nanpercentile(bs_a, 2.5),
                         auc_hi=np.nanpercentile(bs_a, 97.5), auc_within_target=auc(sc, y), auc_w_lo=np.nanpercentile(bs_w, 2.5),
                         auc_w_hi=np.nanpercentile(bs_w, 97.5), auc_neg_entropy=auc(-e, y), auc_e_lo=np.nanpercentile(bs_e, 2.5),
                         auc_e_hi=np.nanpercentile(bs_e, 97.5),
                         tgt_share_correct=s[y == 1].mean(), tgt_share_wrong=s[y == 0].mean(),
                         argmax_is_target_correct=am[y == 1].mean(), argmax_is_target_wrong=am[y == 0].mean(),
                         argmax_is_target_all=am.mean()))
        # attention share by slot of the target, and by image position regardless of role
        for k, h in g.groupby("slot"):
            slot_rows.append(dict(model=m, mode=mode, slot=k, tgt_share=h.tgt_share.mean(), acc=h.joint_ok.mean(),
                                  **{f"mean_p{j}": h[f"p{j}"].mean() for j in range(1, 6)}))
        # attention on distractors by distance from target
        for r in g.itertuples():
            for j in range(1, 6):
                if j != r.slot:
                    dist_rows.append(dict(model=m, mode=mode, dist=abs(j - r.slot), share=getattr(r, f"p{j}")))
    d = pd.DataFrame(rows)
    sl = pd.DataFrame(slot_rows)
    dd = pd.DataFrame(dist_rows).groupby(["model", "mode", "dist"]).share.mean().reset_index()
    A.to_csv(OUT / "f9_attention_trials.csv", index=False)
    d.to_csv(OUT / "f9_attention_auc.csv", index=False)
    sl.to_csv(OUT / "f9_attention_by_slot.csv", index=False)
    dd.to_csv(OUT / "f9_attention_by_distance.csv", index=False)
    log("\n## 9. Attention (E2, N=5): normalised target share predicts correctness?")
    log(d.round(3).to_string(index=False))
    log(sl.round(3).to_string(index=False))
    log(dd.pivot_table(index=["model", "mode"], columns="dist", values="share").round(3).to_string())
    return d


# ---------------------------------------------------------------- 10. cross-model item difficulty
def agreement(core):
    mc = core[core["mode"] == "mc"]
    acc1 = mc[mc.N == 1].pivot_table(index="target", columns="model", values="joint_ok")
    acc5 = mc[mc.N == 5].pivot_table(index="target", columns="model", values="joint_ok")
    cost = acc1 - acc5   # per-target binding cost (N=1 single trial minus mean over 5 slots)
    out = []
    for name, M in (("acc_N1", acc1), ("acc_N5", acc5), ("cost_N1_minus_N5", cost)):
        C = M[MODELS].corr(method="spearman")
        iu = np.triu_indices(len(MODELS), 1)
        V = M[MODELS].values
        bs = []
        for b in range(1000):
            ix = RNG.integers(0, len(V), len(V))
            bs.append(np.nanmean(pd.DataFrame(V[ix]).corr(method="spearman").values[iu]))
        out.append(dict(quantity=name, mean_pairwise_spearman=C.values[iu].mean(), lo=np.percentile(bs, 2.5),
                        hi=np.percentile(bs, 97.5)))
        C.to_csv(OUT / f"f10_spearman_{name}.csv")
    # split-half reliability of per-target N=5 accuracy (slots {1,3,5} vs {2,4}) as a noise ceiling
    rel = []
    for m in MODELS:
        g = mc[(mc.model == m) & (mc.exp == "E2")]
        a = g[g.slot.isin([1, 3, 5])].groupby("target").joint_ok.mean()
        b = g[g.slot.isin([2, 4])].groupby("target").joint_ok.mean()
        r = stats.spearmanr(a, b.reindex(a.index)).statistic
        rel.append(dict(model=m, split_half_rho_N5=r))
    # small-vs-large: do targets that cost the small models also cost the large ones?
    sc, lc = cost[[m for m in MODELS if m in SMALL]].mean(1), cost[[m for m in MODELS if m not in SMALL]].mean(1)
    rho = stats.spearmanr(sc, lc)
    # target ease (mean N1 acc of large models) predicts small-model cost?
    ease = acc1[[m for m in MODELS if m not in SMALL]].mean(1)
    rho2 = stats.spearmanr(ease, sc)
    o = pd.DataFrame(out)
    o.to_csv(OUT / "f10_agreement.csv", index=False)
    pd.DataFrame(rel).to_csv(OUT / "f10_split_half.csv", index=False)
    log("\n## 10. Cross-model agreement on per-target difficulty (MC joint)")
    log(o.round(3).to_string(index=False))
    log(pd.DataFrame(rel).round(3).to_string(index=False))
    log(f"cost small-class vs large-class Spearman rho={rho.statistic:.3f} p={rho.pvalue:.3g}")
    log(f"large-model N1 ease vs small-class cost rho={rho2.statistic:.3f} p={rho2.pvalue:.3g}")
    return o


# ---------------------------------------------------------------- 11. tier x N slope
def tier_slope(core):
    rows, tests = [], []
    mc = core[core["mode"] == "mc"]
    for (m, t), g in mc.groupby(["model", "tier"]):
        A = wide(g, "joint_ok").values
        idx = boot_rows(len(A))
        sb = slope_mat(A[idx].mean(1), [1, 2, 3, 4, 5])
        rows.append(dict(model=m, tier=t, targets=len(A), slope=slope_mat(A.mean(0), [1, 2, 3, 4, 5]), lo=ci(sb)[0], hi=ci(sb)[1]))
    for m, g in mc.groupby("model"):
        tn = g.tier.map({"low": 0, "mid": 1, "high": 2}).astype(float).values - 1
        N = g.N.astype(float).values - 3
        r = sm.GLM(g.joint_ok.values.astype(float), np.c_[np.ones(len(g)), N, tn, N * tn], family=sm.families.Binomial()).fit(
            cov_type="cluster", cov_kwds={"groups": pd.factorize(g.target)[0]})
        tests.append(dict(model=m, NxTier_coef=r.params[3], p=r.pvalues[3]))
    d, t = pd.DataFrame(rows), pd.DataFrame(tests)
    t["p_holm"] = holm(t.p)
    d.to_csv(OUT / "f11_tier_slopes.csv", index=False)
    t.to_csv(OUT / "f11_tier_interaction.csv", index=False)
    log("\n## 11. Slope by popularity tier (MC joint)")
    log(d.pivot(index="model", columns="tier", values="slope").round(3).to_string())
    log(t.round(4).to_string(index=False))


# ---------------------------------------------------------------- 12. regression-to-the-mean control for (2)
def rtm_control(df, core):
    """Conditioning on 'correct at N=1' selects lucky trials, so some later 'loss' is regression to the mean.
    Control: on the 56 E5 targets, P(correct at N=1 under paraphrase P2/P3 | correct at N=1 under P1) is the
    retention expected from a pure prompt perturbation with NO added images. Compare with P(correct at N | ok@N=1, P1)."""
    mc = df[(df["mode"] == "mc") & (df.kind == "main")]
    p1 = core[core["mode"] == "mc"]
    e5 = mc[mc.exp == "E5"]
    rows = []
    for m in MODELS:
        a1 = p1[(p1.model == m) & (p1.N == 1)].set_index("target").joint_ok
        tg = e5[e5.model == m].target.unique()
        known = [t for t in tg if a1.get(t, 0) == 1]
        g5 = e5[(e5.model == m) & e5.target.isin(known)]
        # perturbation-only retention: P2/P3 at N=1 on targets known under P1 at N=1
        ret_pert = g5[g5.N == 1].groupby("target").joint_ok.mean()
        # added-image retention on the same targets: P1 at N=5 (mean over slots) and P2/P3 at N=5
        ret5_p1 = p1[(p1.model == m) & (p1.N == 5) & p1.target.isin(known)].groupby("target").joint_ok.mean()
        ret5_p23 = g5[g5.N == 5].groupby("target").joint_ok.mean()
        both = pd.concat([ret_pert, ret5_p1, ret5_p23], axis=1, keys=["pert", "n5p1", "n5p23"]).dropna()
        V = both.values
        bs = V[boot_rows(len(V))].mean(1)
        lo, hi = ci(bs)
        d = bs[:, 0] - bs[:, 2]
        rows.append(dict(model=m, targets=len(V), retention_N1_paraphrase=V[:, 0].mean(), r_lo=lo[0], r_hi=hi[0],
                         retention_N5_P1=V[:, 1].mean(), retention_N5_paraphrase=V[:, 2].mean(), r5_lo=lo[2], r5_hi=hi[2],
                         excess_loss_N5=V[:, 0].mean() - V[:, 2].mean(), ex_lo=ci(d)[0], ex_hi=ci(d)[1], p=boot_p(d)))
    d = pd.DataFrame(rows)
    d["p_holm"] = holm(d.p)
    d.to_csv(OUT / "f12_rtm_control.csv", index=False)
    log("\n## 12. Regression-to-mean control: retention of N=1-known targets under paraphrase (no images added) vs at N=5")
    log(d.round(3).to_string(index=False))
    return d


# ---------------------------------------------------------------- 13. letter copying (item letter reused for country)
def letter_copy(core):
    """Among MC country-wrong fields where the chosen item letter is NOT the gold country letter (so copying the
    item letter is one of the 4 available wrong answers), the share whose country letter equals the item letter.
    Null 1/4. Trend in N via target-clustered logistic GLM."""
    mc = core[(core["mode"] == "mc") & core.item_let.notna() & core.country_let.notna()]
    rows, trend = [], []
    for m, g in mc.groupby("model"):
        w = g[(g.country_ok == 0) & (g.item_let != g.gold_country_let)]
        y = (w.country_let == w.item_let).astype(float)
        for n, h in w.assign(y=y).groupby("N"):
            per = h.groupby("target").y.agg(["sum", "size"]).values
            ix = boot_rows(len(per))
            bs = per[ix, 0].sum(1) / per[ix, 1].sum(1)
            rows.append(dict(model=m, N=n, fields=len(h), copy_rate=h.y.mean(), lo=ci(bs)[0], hi=ci(bs)[1], null=0.25))
        per = w.assign(y=y).groupby("target").y.agg(["sum", "size"]).values
        ix = boot_rows(len(per))
        bs = per[ix, 0].sum(1) / per[ix, 1].sum(1)
        r = sm.GLM(y.values, np.c_[np.ones(len(w)), w.N.values - 3.0], family=sm.families.Binomial()).fit(
            cov_type="cluster", cov_kwds={"groups": pd.factorize(w.target)[0]})
        # joint-accuracy gap: gold letters coincide vs differ (targets fixed across N)
        gs = g.gold_item_let == g.gold_country_let
        trend.append(dict(model=m, fields=len(w), copy_rate_all=y.mean(), lo=ci(bs)[0], hi=ci(bs)[1], excess_vs_null=y.mean() - 0.25,
                          p_vs_null=boot_p(bs, 0.25), logit_N=r.params[1], p_N=r.pvalues[1],
                          acc_gold_same=g[gs].joint_ok.mean(), acc_gold_diff=g[~gs].joint_ok.mean(),
                          targets_gold_same=g[gs].target.nunique()))
    d, t = pd.DataFrame(rows), pd.DataFrame(trend)
    t["p_vs_null_holm"] = holm(t.p_vs_null)
    t["p_N_holm"] = holm(t.p_N)
    # does the N-slope survive restricting to targets whose gold letters differ (copying cannot help)?
    sl = []
    for m, g in mc.groupby("model"):
        h = g[g.gold_item_let != g.gold_country_let]
        A = wide(h, "joint_ok").values
        sb = slope_mat(A[boot_rows(len(A))].mean(1), [1, 2, 3, 4, 5])
        sl.append(dict(model=m, targets=len(A), slope_gold_diff=slope_mat(A.mean(0), [1, 2, 3, 4, 5]), lo=ci(sb)[0], hi=ci(sb)[1]))
    sl = pd.DataFrame(sl)
    d.to_csv(OUT / "f13_letter_copy_by_N.csv", index=False)
    t.to_csv(OUT / "f13_letter_copy_summary.csv", index=False)
    sl.to_csv(OUT / "f13_slope_gold_letters_differ.csv", index=False)
    log("\n## 13. Letter copying: country-wrong answers reusing the chosen item letter (null 0.25)")
    log(d.pivot(index="model", columns="N", values="copy_rate").round(3).to_string())
    log(t.round(4).to_string(index=False))
    log(sl.round(4).to_string(index=False))
    return t


# ---------------------------------------------------------------- 14. position-adjusted attention
def attention_adjusted():
    """Raw answer-token attention mass is dominated by image POSITION (see f9_attention_by_slot.csv).
    Lift_j = p_j - mean p_j over trials in which slot j holds a distractor (same model x mode).
    (a) target lift > 0? (b) AUC of target lift for joint correctness; (c) capture re-test: among mis-binding
    fields, is the named wrong image the max-lift distractor? chance = 1/4 (target excluded)."""
    A = pd.read_csv(OUT / "f9_attention_trials.csv")
    sc = pd.concat([pd.read_csv(ROOT / f"scored_{m}.csv") for m in ATTN_MODELS])
    sc = sc[sc.exp == "E2"][["model", "mode", "target", "slot", "item_err", "item_slot", "country_err", "country_slot"]]
    A = A.merge(sc, on=["model", "mode", "target", "slot"], how="left")
    rows, cap = [], []
    for (m, mode), g in A.groupby(["model", "mode"]):
        P = g[[f"p{j}" for j in range(1, 6)]].values
        k = g.slot.values - 1
        base = np.array([P[k != j, j].mean() for j in range(5)])
        sd = np.array([P[k != j, j].std() for j in range(5)])
        Z = (P - base) / sd
        lift = Z[np.arange(len(g)), k]
        y = g.joint_ok.values
        ut = g.target.unique()
        gi = {u: np.where(g.target.values == u)[0] for u in ut}
        bl, ba = [], []
        for b in range(B):
            ix = np.concatenate([gi[u] for u in RNG.choice(ut, len(ut))])
            bl.append(lift[ix].mean())
            ba.append(auc(lift[ix], y[ix]))
        rows.append(dict(model=m, mode=mode, target_lift_z=lift.mean(), lift_lo=np.percentile(bl, 2.5), lift_hi=np.percentile(bl, 97.5),
                         target_is_max_z=(np.argmax(Z, 1) == k).mean(), auc_lift=auc(lift, y), auc_lo=np.nanpercentile(ba, 2.5),
                         auc_hi=np.nanpercentile(ba, 97.5)))
        recs = []
        for i, r in enumerate(g.itertuples()):
            for f in ("item", "country"):
                if getattr(r, f"{f}_err") == "mis_binding":
                    j = int(getattr(r, f"{f}_slot")) - 1
                    zz = Z[i].copy()
                    zz[k[i]] = -np.inf
                    others = [x for x in range(5) if x not in (k[i], j)]
                    recs.append((r.target, int(np.argmax(zz) == j), Z[i, j] - Z[i, others].mean()))
        if len(recs) >= 5:
            R = pd.DataFrame(recs, columns=["target", "hit", "dz"])
            per = R.groupby("target").agg(h=("hit", "sum"), n=("hit", "size"), dz=("dz", "sum")).values
            ix = boot_rows(len(per))
            bh = per[ix, 0].sum(1) / per[ix, 1].sum(1)
            bd = per[ix, 2].sum(1) / per[ix, 1].sum(1)
            cap.append(dict(model=m, mode=mode, mis_binding_fields=len(R), named_is_max_lift=R.hit.mean(), lo=ci(bh)[0], hi=ci(bh)[1],
                            chance=0.25, p=boot_p(bh, 0.25), named_minus_other_z=R.dz.mean(), dz_lo=ci(bd)[0], dz_hi=ci(bd)[1],
                            p_dz=boot_p(bd)))
    d, c = pd.DataFrame(rows), pd.DataFrame(cap)
    c["p_holm"] = holm(c.p)
    c["p_dz_holm"] = holm(c.p_dz)
    d.to_csv(OUT / "f14_attention_position_adjusted.csv", index=False)
    c.to_csv(OUT / "f14_capture_position_adjusted.csv", index=False)
    log("\n## 14. Position-adjusted attention (z-scored against the same slot when it holds a distractor)")
    log(d.round(3).to_string(index=False))
    log(c.round(3).to_string(index=False))
    return d, c


# ---------------------------------------------------------------- 15. pooled region attraction + cost reliability
def region_pooled():
    F = pd.read_csv(OUT / "f7_wrong_pick_alternatives.csv")
    obs = F[F.chosen == 1].set_index("ev")
    exp = F.groupby("ev").same_region.mean()
    rows = []
    for name, ms in (("all6", MODELS), ("<=4B", [m for m in MODELS if m in SMALL]), (">=7B", [m for m in MODELS if m not in SMALL])):
        o = obs[obs.model.isin(ms)]
        diff = o.same_region.values - exp.loc[o.index].values
        tg = o.target.values
        ut = np.unique(tg)
        per = np.array([diff[tg == u].sum() for u in ut])
        cnt = np.array([(tg == u).sum() for u in ut])
        ix = boot_rows(len(ut))
        bs = per[ix].sum(1) / cnt[ix].sum(1)
        rows.append(dict(models=name, events=len(o), observed=o.same_region.mean(), expected=exp.loc[o.index].mean(),
                         excess=diff.mean(), lo=ci(bs)[0], hi=ci(bs)[1], p=boot_p(bs),
                         rel_risk=o.same_region.mean() / exp.loc[o.index].mean()))
    d = pd.DataFrame(rows)
    d.to_csv(OUT / "f15_region_attraction_pooled.csv", index=False)
    log("\n## 15. Same-region attraction in the MIXED list, N=5 MC, pooled (null: uniform over 4 shown distractors)")
    log(d.round(4).to_string(index=False))
    return d


def cost_reliability(df, core):
    """Is per-target binding cost (acc N=1 - acc N=5) a stable target property within a model?
    Test-retest on the 56 E5 targets: cost under P1 vs cost under P2/P3 (same slots, different wording)."""
    mc = df[(df["mode"] == "mc") & (df.kind == "main")]
    p1 = core[(core["mode"] == "mc")]
    rows = []
    for m in MODELS:
        e5 = mc[(mc.model == m) & (mc.exp == "E5")]
        tg = e5.target.unique()
        b = p1[(p1.model == m) & p1.target.isin(tg)]
        c1 = b[b.N == 1].groupby("target").joint_ok.mean() - b[(b.N == 5) & (b.slot == b.idx % 5 + 1)].groupby("target").joint_ok.mean()
        c2 = e5[e5.N == 1].groupby("target").joint_ok.mean() - e5[e5.N == 5].groupby("target").joint_ok.mean()
        a1 = b[b.N == 1].groupby("target").joint_ok.mean()
        a2 = e5[e5.N == 1].groupby("target").joint_ok.mean()
        rows.append(dict(model=m, targets=len(tg), cost_retest_spearman=stats.spearmanr(c1, c2.reindex(c1.index)).statistic,
                         acc1_retest_spearman=stats.spearmanr(a1, a2.reindex(a1.index)).statistic))
    d = pd.DataFrame(rows)
    d.to_csv(OUT / "f15_cost_reliability.csv", index=False)
    log("\n## 15b. Within-model test-retest of per-target cost (P1 vs P2/P3, 56 targets)")
    log(d.round(3).to_string(index=False))
    return d


if __name__ == "__main__":
    df, tr, pool = load()
    core, bal = core_sets(df)
    which = set(sys.argv[1:]) or {str(i) for i in range(1, 16)}
    steps = {"1": lambda: per_image_cost(core), "2": lambda: conditional_on_known(core, bal), "3": lambda: survival(bal),
             "4": lambda: category(core), "5": lambda: partial(core), "6": lambda: slots(core),
             "7": lambda: wrong_pick(core, pool), "8": lambda: letters(core), "9": lambda: attention(core),
             "10": lambda: agreement(core), "11": lambda: tier_slope(core), "12": lambda: rtm_control(df, core),
             "13": lambda: letter_copy(core), "14": attention_adjusted, "15": lambda: (region_pooled(), cost_reliability(df, core))}
    for k in sorted(which, key=int):
        steps[k]()
    (OUT / f"novel_log_{'_'.join(sorted(which, key=int))}.txt").write_text("\n".join(LOG) + "\n")
