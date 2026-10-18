#!/usr/bin/env python3
"""Post-freeze GPU results (EXPLORATORY) -> paper-ready macros, tables and figures.

  .venv/bin/python work/analysis/gpu_results.py                    # -> work/analysis/out/paper/{sec,figures}
  PAPER_OUT=~/Desktop/paper .venv/bin/python work/analysis/gpu_results.py   # write into the LaTeX paper

Reads only work/gpu_out/ (raw_<m>_interventions.jsonl, raw_<m>_collage.jsonl, causal_<m>.jsonl,
causal_cases_<m>.jsonl, results/causal_<m>.csv, attn_lfm25v.jsonl, logs/) plus the frozen scored_<m>.csv,
trials_*.jsonl and data/pool.csv. Every model/experiment that is missing or partial is skipped or flagged,
so the script can be re-run after each sync. Writes:
  $PAPER_OUT/sec/gen_macros.tex        \\gpudef{key}{value}; the paper reads values with \\gpu{key}
  $PAPER_OUT/sec/gen_tab_expl.tex      main-body table rows (E7/E8/E9 per model)
  $PAPER_OUT/sec/gen_tab_e9.tex        supp: E9 accuracy per condition
  $PAPER_OUT/sec/gen_tab_causal.tex    supp: causal patching per layer
  $PAPER_OUT/sec/gen_supp_runs.tex     supp S4: hardware, versions, n, timings
  $PAPER_OUT/sec/gen_supp_prompts.tex  supp S1: prompt text (ch.make_parts, text only)
  $PAPER_OUT/figures/figE9.pdf, figCausal.pdf
CIs: percentile bootstrap over targets (cluster = target), B=2000, numpy default_rng(0) per estimate.
"""
import json
import os
import re
import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)                                   # ch.load_pool() etc. use repo-relative paths
sys.path.insert(0, str(ROOT))
import ch                                        # noqa: E402
from analyze_collage import icr as collage_icr   # noqa: E402  (pure function, no side effects)

import matplotlib                                # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
import style as S                                # noqa: E402  shared design system (work/analysis/style.py)

GPU = Path(os.environ.get("GPU_OUT", ROOT / "work/gpu_out"))   # override only for layout stress tests
# Output folder: PAPER_OUT=<paper dir> writes straight into a LaTeX paper (sec/ + figures/); default stays in the repo.
OUT = Path(os.environ.get("PAPER_OUT", ROOT / "work/analysis/out/paper")).expanduser()
SEC, FIG = OUT / "sec", OUT / "figures"
SEC.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)
B = 2000
COL, NAME = S.COL, S.NAME                       # work/analysis/style.py
TEX = {"qwen3vl": r"\qthree", "qwen25vl": r"\qtwo", "gemma4": r"\gfour", "gemma3_4b": r"\gthree", "granite4v": r"\gran", "lfm25v": r"\lfm"}
IV_MODELS = ["lfm25v", "granite4v", "gemma3_4b"]           # E7/E8
E9_MODELS = ["gemma4", "qwen3vl", "qwen25vl", "gemma3_4b", "granite4v", "lfm25v"]   # >=7B then <=4B
BIG = {"qwen3vl", "qwen25vl", "gemma4"}             # the >=7B size class
CAUSAL_MODELS = ["lfm25v", "qwen3vl"]
CONDS = ["solo", "separate", "collage_pos", "collage_kind"]
CN = dict(zip(CONDS, ["C1", "C2", "C3", "C4"]))
EXPECTED = {"E7": 280, "E8": 800, "E9": 640}
M = {}                                                     # macro key -> LaTeX value
EST = {}                                                   # macro key -> (mean, lo, hi), numeric


def rows(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def andjoin(xs):
    return "none" if not xs else xs[0] if len(xs) == 1 else ", ".join(xs[:-1]) + " and " + xs[-1]


def cboot(x, cl):
    """Mean of x and its target-clustered percentile-bootstrap 95% CI."""
    d = pd.DataFrame({"x": np.asarray(x, float), "c": list(cl)}).groupby("c").x.agg(["sum", "size"])
    S, C = d["sum"].values, d["size"].values
    if len(S) < 2:
        return S.sum() / C.sum(), np.nan, np.nan
    ix = np.random.default_rng(0).integers(0, len(S), (B, len(S)))
    lo, hi = np.percentile(S[ix].sum(1) / C[ix].sum(1), [2.5, 97.5])
    return S.sum() / C.sum(), lo, hi


def f(x, d=3, sign=False):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    s = f"{x:+.{d}f}" if sign else f"{x:.{d}f}"
    return s.replace("-", "$-$").replace("+", "$+$")


def put(key, v):
    M[key] = v


def put_est(key, est, sign=True, d=3):
    """est = (mean, lo, hi): key, key.lo, key.hi, key.ci ('[lo, hi]'), key.sig (CI excludes 0)."""
    m, lo, hi = est
    EST[key] = (m, lo, hi)
    put(key, f(m, d, sign))
    put(key + ".ci", f"[{f(lo, d)}, {f(hi, d)}]")
    put(key + ".full", f"{f(m, d, sign)} [{f(lo, d)}, {f(hi, d)}]")
    put(key + ".abs", f(abs(m), d))
    sig = (not np.isnan(lo)) and (lo > 0 or hi < 0)
    put(key + ".sig", "1" if sig else "0")
    return sig


def score(trials, raw, pool):
    out = {}
    for r in raw:
        if r.get("mode") != "mc" or r["trial_id"] in out:   # first row wins (resume never repeats)
            continue
        t = trials[r["trial_id"]]
        out[r["trial_id"]] = dict(ch.score_trial(t, "mc", r["raw"], pool), trial=t)
    return out


# ------------------------------------------------------------------ E7 / E8
def interventions(pool):
    trials = {t["trial_id"]: t for t in ch.read_jsonl("trials_interventions.jsonl")}
    main = {t["trial_id"]: t for t in ch.read_jsonl("trials_main.jsonl")}
    res = {}
    for m in IV_MODELS:
        p = GPU / f"raw_{m}_interventions.jsonl"
        if not p.exists():
            continue
        S = score(trials, rows(p), pool)
        fz = pd.read_csv(f"scored_{m}.csv")
        fz = fz[(fz["mode"] == "mc") & (fz.kind == "main") & (fz.list == "mixed")]
        key = {(r.exp, r.variant, r.target, r.N, r.slot): r.joint_ok for r in fz.itertuples()}
        kfmt = {(r.exp, r.variant, r.target, r.N, r.slot): int("format_error" in (r.item_err, r.country_err))
                for r in fz.itertuples()}
        direct = lambda t: key.get(("E1" if t["N"] < 5 else "E2", "P1", t["target"], t["N"], t["slot"]))
        res[m] = {}
        for exp, (bexp, bvar) in (("E7", ("E6", "P4")), ("E8", ("E2", "P1"))):
            d = [s for s in S.values() if s["trial"]["exp"] == exp]
            if not d:
                continue
            pr = []
            for s in d:
                t = s["trial"]
                b = key.get((bexp, bvar, t["target"], t["N"], t["slot"]))
                # pairing check: same target, N, slot, shown order and option lists as the frozen trial
                old = main[t["trial_id"].replace(exp + "_", bexp + "_").replace("_P5", "_P4")]
                assert all(old[k] == t[k] for k in ("target", "N", "slot", "order", "item_options", "country_options"))
                pr.append(dict(target=t["target"], N=t["N"], new=s["joint_ok"], base=b, direct=direct(t),
                               fmt=int("format_error" in (s["item_err"], s["country_err"])),
                               basefmt=kfmt[(bexp, bvar, t["target"], t["N"], t["slot"])]))
            P = pd.DataFrame(pr)
            assert P.base.notna().all() and P.direct.notna().all(), f"{m} {exp}: unpaired frozen rows"
            k = f"iv.{m}.{exp}"
            n = len(P)
            put(k + ".n", str(n))
            put(k + ".complete", "1" if n == EXPECTED[exp] else "0")
            put(k + ".acc", f(P.new.mean()))
            put(k + ".base", f(P.base.mean()))
            put(k + ".fmt", str(int(P.fmt.sum())))
            put(k + ".basefmt", str(int(P.basefmt.sum())))
            W = P[(P.fmt == 0) & (P.basefmt == 0)]       # pairs where both replies parsed
            put(k + ".nwf", str(len(W)))
            dwf = cboot(W.new - W.base, W.target)
            put_est(k + ".dwf", dwf)
            r = dict(n=n, complete=n == EXPECTED[exp], acc=P.new.mean(), base=P.base.mean(), fmt=int(P.fmt.sum()), dwf=dwf,
                     d=cboot(P.new - P.base, P.target))
            put_est(k + ".d", r["d"])
            if exp == "E7":                       # also vs the direct P1 prompt and the frozen E6-direct gap
                r["ddir"] = cboot(P.new - P.direct, P.target)
                r["e6dir"] = cboot(P.base - P.direct, P.target)
                put_est(k + ".ddir", r["ddir"])
                put_est(k + ".e6dir", r["e6dir"])
                put(k + ".direct", f(P.direct.mean()))
            res[m][exp] = r
    return res


# ------------------------------------------------------------------ E9
def collage(pool):
    trials = {t["trial_id"]: t for t in ch.read_jsonl("trials_collage.jsonl")}
    res = {}
    for m in E9_MODELS:
        p = GPU / f"raw_{m}_collage.jsonl"
        if not p.exists():
            continue
        S = score(trials, rows(p), pool)
        by = {c: {} for c in CONDS}
        for s in S.values():
            by[s["trial"]["condition"]][s["trial"]["target"]] = dict(s, slot=s["trial"]["slot"])
        n = sum(len(v) for v in by.values())
        if n < 40 or any(len(by[c]) == 0 for c in CONDS):     # smoke-only file
            continue
        k, r = f"e9.{m}", dict(n=n, complete=n == EXPECTED["E9"], acc={}, d={}, icr={})
        put(k + ".n", str(n))
        put(k + ".complete", "1" if r["complete"] else "0")
        for c in CONDS:
            g = list(by[c].values())
            for fld in ("item", "country", "joint"):
                x = [s[f"{fld}_ok"] for s in g]
                r["acc"][c, fld] = cboot(x, [s["trial"]["target"] for s in g])
                put_est(f"{k}.{CN[c]}.{fld}", r["acc"][c, fld], sign=False)
            if c != "solo":
                wrong, obs, base, pp = collage_icr(g)
                r["icr"][c] = (wrong, obs, base, pp)
                put(f"{k}.{CN[c]}.icr", f(obs, 2))
                put(f"{k}.{CN[c]}.icrbase", f(base, 2))
                put(f"{k}.{CN[c]}.icrp", f(pp, 2))
                put(f"{k}.{CN[c]}.icrn", str(wrong))
        for a, b in (("collage_pos", "separate"), ("collage_kind", "collage_pos"), ("separate", "solo"),
                     ("collage_pos", "solo"), ("collage_kind", "solo")):
            tg = sorted(set(by[a]) & set(by[b]))
            for fld in ("item", "country", "joint"):
                r["d"][a, b, fld] = cboot([by[a][t][f"{fld}_ok"] - by[b][t][f"{fld}_ok"] for t in tg], tg)
                put_est(f"{k}.{CN[a]}m{CN[b]}.{fld}", r["d"][a, b, fld])
            # country-vs-item loss: (country drop) - (item drop) relative to b, i.e. extra country loss
            x = [(by[a][t]["country_ok"] - by[b][t]["country_ok"]) - (by[a][t]["item_ok"] - by[b][t]["item_ok"]) for t in tg]
            r["d"][a, b, "cmi"] = cboot(x, tg)
            put_est(f"{k}.{CN[a]}m{CN[b]}.cmi", r["d"][a, b, "cmi"])
        res[m] = r
    return res


# ------------------------------------------------------------------ causal patching
def causal():
    res = {}
    for m in CAUSAL_MODELS:
        p, done = GPU / f"causal_{m}.jsonl", (GPU / "results" / f"causal_{m}.csv").exists()
        if not p.exists():
            continue
        R = pd.DataFrame(rows(p))
        ncases = len(rows(GPU / f"causal_cases_{m}.jsonl")) if (GPU / f"causal_cases_{m}.jsonl").exists() else np.nan
        gate = R[R.intervention == "wrong_donor_gate_fail"].case_id.nunique()
        D = R[R.intervention != "wrong_donor_gate_fail"].copy()
        layers = sorted(int(x) for x in D.layer.dropna().unique())
        # keep only cases with all layers x interventions (a case interrupted mid-way is dropped)
        full = D.groupby("case_id").size()
        D = D[D.case_id.isin(full[full == 3 * len(layers)].index)]
        if D.empty:
            continue
        cases = D.drop_duplicates("case_id")
        k = f"causal.{m}"
        r = dict(done=done, layers=layers, n=len(cases), ntg=cases.target.nunique(), gate=gate, ncases=ncases, L={})
        put(k + ".done", "1" if done else "0")
        put(k + ".n", str(len(cases)))
        put(k + ".ntargets", str(cases.target.nunique()))
        put(k + ".gate", str(gate))
        put(k + ".selected", str(ncases))
        put(k + ".nitem", str(int((cases.field == "item").sum())))
        put(k + ".ncountry", str(int((cases.field == "country").sum())))
        put(k + ".layers", ", ".join(map(str, layers)))
        put(k + ".base", f(cases.baseline_logodds.mean(), 2, True))
        put(k + ".basepos", str(int((cases.baseline_logodds > 0).sum())))
        selfmax = D[D.intervention == "self_to_target_control"].delta_logodds.abs().max()
        r["selfmax"] = selfmax
        put(k + ".selfmax", f"{selfmax:.3f}")
        for L in layers:
            for iv, name in (("target_clean_to_target", "rescue"), ("wrong_clean_to_target", "induct"),
                             ("self_to_target_control", "self")):
                g = D[(D.layer == L) & (D.intervention == iv)]
                est = cboot(g.delta_logodds, g.target)
                r["L"][L, name] = est
                put_est(f"{k}.L{L}.{name}", est, d=2)
                put(f"{k}.L{L}.{name}.flip", f((g.patched_logodds > 0).mean(), 2))
        put(k + ".flipbase", f((cases.baseline_logodds > 0).mean(), 2))
        # summaries over the grid: largest |rescue| and strongest induction
        resc = max(layers, key=lambda L: abs(r["L"][L, "rescue"][0]))
        ind = min(layers, key=lambda L: r["L"][L, "induct"][0])
        put(k + ".rescuemaxabs", f(abs(r["L"][resc, "rescue"][0]), 2))
        put(k + ".rescuemaxL", str(resc))
        put(k + ".inductpeakL", str(ind))
        put(k + ".inductpeak", f(r["L"][ind, "induct"][0], 2, True))
        put(k + ".inductpeak.ci", M[f"{k}.L{ind}.induct.ci"])
        sig_ind = [L for L in layers if M[f"{k}.L{L}.induct.sig"] == "1" and r["L"][L, "induct"][0] < 0]
        put(k + ".inductsigL", ", ".join(map(str, sig_ind)) or "none")
        put(k + ".inductlast", str(max(sig_ind)) if sig_ind else "none")
        later = [L for L in layers if L > (max(sig_ind) if sig_ind else -1)]
        put(k + ".nullL", ", ".join(map(str, later)) or "none")
        put(k + ".rescuesig", ", ".join(str(L) for L in layers if M[f"{k}.L{L}.rescue.sig"] == "1") or "none")
        res[m] = r
    return res


# ------------------------------------------------------------------ LFM attention extension
def attention_lfm():
    p = GPU / "attn_lfm25v.jsonl"
    if not p.exists():
        return None
    att = {(r["trial_id"], r["mode"]): np.array(r["attn"], float) for r in rows(p)}
    sc = pd.read_csv("scored_lfm25v.csv")
    sc = sc[(sc.exp == "E2") & (sc.kind == "main")]
    recs = []
    for r in sc.itertuples():
        a = att.get((r.trial_id, r.mode))
        if a is not None and len(a) == 5:
            recs.append(dict(r._asdict(), **{f"p{j}": a[j] / a.sum() for j in range(5)}, amax=int(np.argmax(a))))
    if not recs:
        return None
    A = pd.DataFrame(recs)
    put("attn.lfm.rows", str(len(A)))
    put("attn.lfm.complete", "1" if len(A) == 1600 else "0")
    out = {}
    for mode, g in A.groupby("mode"):
        P = g[[f"p{j}" for j in range(5)]].values
        k = g.slot.values - 1
        Z = (P - np.array([P[k != j, j].mean() for j in range(5)])) / np.array([P[k != j, j].std() for j in range(5)])
        raw, adj, tg = [], [], []
        for i, r in enumerate(g.itertuples()):
            for fld in ("item", "country"):
                if getattr(r, f"{fld}_err") == "mis_binding" and getattr(r, f"{fld}_slot") >= 1:
                    j = int(getattr(r, f"{fld}_slot")) - 1
                    zz = Z[i].copy()
                    zz[k[i]] = -np.inf
                    raw.append(int(r.amax == j))
                    adj.append(int(np.argmax(zz) == j))
                    tg.append(r.target)
        if len(raw) < 5:
            continue
        kk = f"attn.lfm.{mode}"
        put(kk + ".n", str(len(raw)))
        put_est(kk + ".raw", cboot(raw, tg), sign=False, d=2)
        put_est(kk + ".adj", cboot(adj, tg), sign=False, d=2)
        put(kk + ".img1", f(P[:, 0].mean(), 2))
        out[mode] = (len(raw), cboot(raw, tg), cboot(adj, tg))
    return out


# ------------------------------------------------------------------ figures
# Design system: paper/figures/style.py. Sizes are final print size: figCausal + 0.25in + figE9 = \textwidth.
H = 1.42                              # in, both panels of fig:causal-e9


def _depth_split(r, big=1.0):
    """Layer boundary after the last layer where the wrong-image patch moves the answer by > `big` log-odds."""
    L = r["layers"]
    hit = [l for l in L if r["L"][l, "induct"][0] < -big]
    if not hit or max(hit) == L[-1]:
        return None
    return (max(hit) + L[L.index(max(hit)) + 1]) / 2


def fig_causal(C):
    S.apply()
    fig = plt.figure(figsize=(2.75, H))
    ms = [m for m in CAUSAL_MODELS if m in C]
    lo = min(min(C[m]["L"][l, k][1] for l in C[m]["layers"] for k in ("rescue", "induct")) for m in ms)
    ylim = (np.floor(lo) - 0.2, 1.3)
    axs = [fig.add_axes([0.165 + i * 0.425, 0.25, 0.375, 0.62]) for i in range(len(ms))]
    sm = S.FS_SMALL - 0.5
    for i, (ax, m) in enumerate(zip(axs, ms)):
        r, L = C[m], C[m]["layers"]
        ax.axhline(0, color=S.GREY, lw=0.5, alpha=0.7, zorder=1)
        cut = _depth_split(r)
        if cut is not None:
            ax.axvspan(cut, L[-1] + 3, color=S.GRID, lw=0, zorder=0)
            if i == len(ms) - 1:                        # label the shaded band once, where it is empty
                ax.text((cut + L[-1] + 2) / 2, -4.2, "no effect\nafter mid-depth", ha="center", va="center",
                        fontsize=sm, color=S.MUTED, linespacing=0.95)
        # clean target = ink, wrong image = ACCENT (same encoding in both panels; model = panel title colour)
        for name, c, mk in (("rescue", S.INK, "o"), ("induct", S.ACCENT, "v")):
            e = np.array([r["L"][l, name] for l in L])
            S.ci(ax, L, e[:, 1], e[:, 2], c, alpha=0.13)
            ax.plot(L, e[:, 0], color=c, marker=mk, ms=3, lw=1.1, zorder=3, mec="white", mew=0.4)
        ax.set_xticks(L, [str(l) for l in L])
        ax.set_xlim(L[0] - 2, L[-1] + 2)
        ax.set_ylim(*ylim)
        ax.set_yticks(np.arange(np.ceil(ylim[0] / 2) * 2, 1, 2))
        ax.text(0.0, 1.03, NAME[m], transform=ax.transAxes, color=S.COL[m], fontsize=S.FS, weight="bold",
                va="bottom")
        if i:
            ax.tick_params(labelleft=False)
        else:                                              # direct labels, once
            ax.text((cut or L[2]) + 0.6, -3.5, "wrong-image patch\npulls the answer\nto the wrong image\n(early layers)",
                    fontsize=sm, color=S.ACCENT, ha="left", va="center", linespacing=0.95)
            ax.text(L[0] - 1.5, 0.3, "clean target never rescues", fontsize=sm, color=S.INK, ha="left",
                    va="bottom")
    axs[0].set_ylabel("Δ log-odds (correct − wrong)", fontsize=S.FS_SMALL)
    fig.text(0.565, 0.03, "patched decoder layer (early → late)", ha="center", fontsize=S.FS_SMALL)
    S.panel(fig, "a", 0.0, 0.995)
    S.save(fig, FIG / "figCausal")
    plt.close(fig)


SHORT = {"gemma4": "Gemma-4", "qwen3vl": "Qwen3-VL", "qwen25vl": "Qwen2.5-VL", "gemma3_4b": "Gemma-3",
         "granite4v": "Granite", "lfm25v": "LFM2.5"}


def fig_e9(E9):
    """Joint (item + country) MC accuracy per condition, one panel per size class (shared y). End labels give the
    collage cost (collage minus the target alone in the same canvas, which controls panel resolution; paired by target;
    bold = 95% CI excludes 0)."""
    S.apply()
    fs = S.FS_SMALL - 0.5                                    # 6 pt at print size
    fig = plt.figure(figsize=(3.875, H))
    x = np.arange(4)
    groups = [("≥7B models", S.LARGE_C, [m for m in E9_MODELS if m in E9 and m not in S.SMALL]),
              ("≤4B models", S.SMALL_C, [m for m in E9_MODELS if m in E9 and m in S.SMALL])]
    for i, (title, tc, ms) in enumerate(groups):
        ax = fig.add_axes([0.085 + i * 0.465, 0.25, 0.27, 0.6])
        ax.axvspan(1.5, 3.5, color=S.GRID, lw=0, zorder=0)
        ax.axhline(0.04, color=S.GREY, lw=0.6, ls=(0, (3, 2)), zorder=1)
        ax.text(2.5, 0.98, "one 2×2\nimage", fontsize=fs, color=S.MUTED, ha="center", va="top", linespacing=0.95)
        ax.text(0.0, 1.03, title, transform=ax.transAxes, fontsize=S.FS_SMALL, color=tc, weight="bold", va="bottom")
        end = {}
        for j, m in enumerate(ms):
            r, dx = E9[m], (j - (len(ms) - 1) / 2) * 0.1
            est = np.array([r["acc"][c, "joint"] for c in CONDS])
            S.errbar(ax, x + dx, est[:, 0], est[:, 1], est[:, 2], S.COL[m], ls=S.LS[m], zorder=3, mec="white",
                     mew=0.4, marker=S.MARK[m])
            end[m] = (3 + dx, est[3, 0])
        ax.set_xticks(x, ["alone", "4\nimages", "2×2\ncollage", "ask by\nkind"], fontsize=fs)
        ax.tick_params(axis="x", length=0, pad=2)
        ax.tick_params(axis="y", labelsize=fs)
        ax.set_xlim(-0.5, 3.5)
        ax.set_ylim(0, 1)
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], ["0", ".2", ".4", ".6", ".8", "1"])
        if i == 0:
            ax.set_ylabel("accuracy (item and country)", fontsize=S.FS_SMALL)
            ax.text(-0.45, 0.055, "chance", fontsize=fs, color=S.MUTED, va="bottom")
        else:
            ax.tick_params(labelleft=False)
        # end labels: short name + collage cost, spread vertically in line-end order, thin leaders
        tx = 3.6
        ax.text(tx, 0.98, "collage vs.\nalone", fontsize=fs, color=S.INK, va="top", ha="left",
                linespacing=0.95, clip_on=False)
        order = sorted(ms, key=lambda m: end[m][1], reverse=True)
        ys = []
        for m in order:
            ys.append(min(end[m][1], ys[-1] - 0.12) if ys else min(end[m][1], 0.72))
        for m, y in zip(order, ys):
            d = E9[m]["d"]["collage_pos", "solo", "joint"]
            sig = d[2] < 0 or d[1] > 0
            ax.annotate(f"{SHORT[m]} " + f"{d[0]:+.2f}".replace("-", "−"), end[m], xytext=(tx, y),
                        textcoords="data", fontsize=fs, color=S.COL[m], va="center",
                        weight="bold" if sig else "normal", annotation_clip=False,
                        arrowprops=dict(arrowstyle="-", color=S.COL[m], lw=0.4, shrinkA=0, shrinkB=2))
    S.panel(fig, "b", 0.0, 0.995)
    S.save(fig, FIG / "figE9")
    plt.close(fig)

# ------------------------------------------------------------------ tables
def ci3(est, sign=True):
    m, lo, hi = est
    return f"{f(m, 3, sign)} {{\\scriptsize[{f(lo, 2)}, {f(hi, 2)}]}}"


def tab_expl(IV):
    """Supp rows: model | exp | n | acc | frozen baseline acc | delta | delta (both replies parsed) | E7 vs P1 | format errors."""
    out = ["% auto-generated by work/analysis/gpu_results.py -- EXPLORATORY, post-freeze. Do not edit."]
    for m in IV_MODELS:
        for exp, base in (("E7", "E6"), ("E8", "E2")):
            r = IV.get(m, {}).get(exp)
            if r is None:
                continue
            k = f"iv.{m}.{exp}"
            out.append(" & ".join([TEX[m] if exp == "E7" or "E7" not in IV[m] else "", f"{exp} vs.\\ {base}",
                                   str(r["n"]) + ("" if r["complete"] else "$^\\ast$"), f(r["acc"]), f(r["base"]),
                                   ci3(r["d"]), ci3(r["dwf"]),
                                   ci3(r["ddir"]) if exp == "E7" else "--",
                                   f"{r['fmt']} / {M[k + '.basefmt']}"]) + r" \\")
    if len(out) == 1:
        out.append(r"\multicolumn{9}{c}{\todo{no E7/E8 results synced yet}} \\")
    (SEC / "gen_tab_expl.tex").write_text("\n".join(out) + "\n\\bottomrule\n")   # \bottomrule here: LaTeX \input cannot precede it


def tab_e9(E9):
    out = ["% auto-generated by work/analysis/gpu_results.py -- EXPLORATORY, post-freeze. Do not edit."]
    for m in E9_MODELS:
        if m not in E9:
            continue
        r = E9[m]
        for fld in ("item", "country", "joint"):
            cells = [TEX[m] + ("" if r["complete"] else "$^\\ast$") if fld == "item" else "", fld]
            for c in CONDS:
                v = f(r["acc"][c, fld][0])
                cells.append(f"({v})" if c == "collage_kind" and fld != "country" else v)
            cells.append(ci3(r["d"]["separate", "solo", fld]))
            cells.append(ci3(r["d"]["collage_pos", "separate", fld]))
            cells.append(ci3(r["d"]["collage_kind", "collage_pos", fld]) if fld == "country" else
                         "(" + f(r["d"]["collage_kind", "collage_pos", fld][0], 3, True) + ")")
            out.append(" & ".join(cells) + r" \\")
        icr = "; ".join(f"{CN[c]} {f(r['icr'][c][1], 2)} vs.\\ {f(r['icr'][c][2], 2)}" for c in CONDS[1:])
        out.append(rf"\multicolumn{{8}}{{@{{}}l}}{{\scriptsize\quad ICR (obs.\ vs.\ permutation baseline): {icr}}} \\")
        out.append(r"\addlinespace[2pt]")
    if len(out) == 1:
        out.append(r"\multicolumn{8}{c}{\todo{E9 results pending}} \\")
    (SEC / "gen_tab_e9.tex").write_text("\n".join(out) + "\n\\bottomrule\n")   # \bottomrule here: LaTeX \input cannot precede it


def tab_causal(C):
    out = ["% auto-generated by work/analysis/gpu_results.py -- EXPLORATORY, post-freeze. Do not edit."]
    for m in CAUSAL_MODELS:
        if m not in C:
            continue
        r = C[m]
        for i, L in enumerate(r["layers"]):
            name = (TEX[m] + ("" if r["done"] else "$^\\ast$")) if i == 0 else ""
            out.append(" & ".join([name, str(L), ci3(r["L"][L, "rescue"]), ci3(r["L"][L, "induct"]),
                                   f(r["L"][L, "self"][0], 3, True)]) + r" \\")
        out.append(rf"\multicolumn{{5}}{{@{{}}l}}{{\scriptsize\quad $n{{=}}{r['n']}$ cases on {r['ntg']} targets; "
                   rf"{r['gate']} of {r['ncases']} selected cases failed the wrong-donor gate; "
                   rf"max $|\Delta|$ self-patch {r['selfmax']:.3f}}} \\")
        out.append(r"\addlinespace[2pt]")
    if len(out) == 1:
        out.append(r"\multicolumn{5}{c}{\todo{causal results pending}} \\")
    (SEC / "gen_tab_causal.tex").write_text("\n".join(out) + "\n\\bottomrule\n")   # \bottomrule here: LaTeX \input cannot precede it


# ------------------------------------------------------------------ S4 run details from logs
def runs():
    L = GPU / "logs"
    env = (L / "setup.log").read_text(errors="ignore") if (L / "setup.log").exists() else ""
    mt = re.search(r"torch (\S+?)(?:\+\S+)? cuda (\S+) \| (.+?) \| transformers (\S+)", env)
    if mt:
        put("run.torch", mt.group(1)); put("run.cuda", mt.group(2)); put("run.gpu", mt.group(3)); put("run.tf", mt.group(4))
    for m in ("lfm25v", "granite4v", "gemma3_4b", "qwen3vl"):
        p = L / f"repro_{m}.log"
        if p.exists():
            r = re.search(r"REPRO \S+: (\d+)/(\d+)", p.read_text(errors="ignore"))
            if r:
                put(f"repro.{m}", f"{r.group(1)}/{r.group(2)}")
    jobs = {}
    for p in sorted(L.glob("lane*.log")):
        for j, mins in re.findall(r"done\s+(\S+) \((\d+) min\)", p.read_text(errors="ignore")):
            jobs[j] = int(mins)
    tab = []
    for m in ("lfm25v", "granite4v", "gemma3_4b", "qwen3vl"):
        for job, label, n in ((f"E7_{m}", "E7", "280"), (f"E8_{m}", "E8", "800"), (f"E9_{m}", "E9", "640"),
                              (f"attn_{m}", "attn.\ (E2)", "1{,}600"), (f"causal_{m}", "causal", None)):
            p = L / f"causal_{m}.log"
            if label == "causal" and p.exists():
                s = [float(x) for x in re.findall(r"cases \((\d+\.\d)s\)", p.read_text(errors="ignore"))]
                if s:
                    per = f"{np.median(s):.0f}\\,s/case"
                    tab.append(f"{TEX[m]} & causal & {len(s)} & " + (f"{jobs[job]} min ({per})" if job in jobs else per)
                               + r" \\")
            elif job in jobs:
                tab.append(f"{TEX[m]} & {label} & {n} & {jobs[job]} min \\\\")
    (SEC / "gen_supp_runs.tex").write_text("% auto-generated by work/analysis/gpu_results.py\n"
                                           + ("\n".join(tab) if tab else r"\multicolumn{4}{c}{--} \\") + "\n\\bottomrule\n")


# ------------------------------------------------------------------ S1 prompts
def prompts():
    main = ch.read_jsonl("trials_main.jsonl")
    iv = ch.read_jsonl("trials_interventions.jsonl")
    e9 = ch.read_jsonl("trials_collage.jsonl")
    t1 = next(t for t in main if t["exp"] == "E1" and t["N"] == 3 and t["slot"] == 2)
    t7 = next(t for t in iv if t["exp"] == "E7" and t["N"] == 3)
    t6 = next(t for t in main if t["exp"] == "E6" and t["N"] == 3)
    t5 = {v: next(t for t in main if t["exp"] == "E5" and t["variant"] == v and t["N"] == 3) for v in ("P2", "P3")}
    t8 = next(t for t in iv if t["exp"] == "E8" and t["slot"] == 3)
    e9t = next(t["target"] for t in e9 if t["slot"] == 3)
    t9 = {t["condition"]: t for t in e9 if t["target"] == e9t}

    def text(t, mode="mc"):
        s = "".join("<image>" if k == "image" else v for k, v in ch.make_parts(t, mode))
        lines = []
        for ln in s.split("\n"):
            lines += textwrap.wrap(ln, 62, subsequent_indent="  ") or [""]
        return "\n".join(lines).strip("\n")

    def diff(t, ref):                      # lines of t's prompt that differ from ref's, wrapped
        raw = lambda x: "".join("<image>" if k == "image" else v for k, v in ch.make_parts(x, "mc")).split("\n")
        r = set(raw(ref))
        return "\n".join(textwrap.fill(l, 62, subsequent_indent="  ") for l in raw(t) if l not in r)

    blocks = [("Main study, P1, MC, $N{=}3$", "(target in Image~2). The open-ended prompt asks the same two "
               "questions without options and ends with \\texttt{Item: <name>} / \\texttt{Country: <country>}.", text(t1)),
              ("Paraphrases (E5)", "replace the first instruction line of P1 (MC shown):",
               "\n".join(textwrap.fill(ch.make_parts(t5[v], "mc")[-1][1].split("\n\n")[1], 62, subsequent_indent="  ")
                         for v in ("P2", "P3"))),
              ("E6 (P4)", "(describe-then-answer), $N{=}3$; open mode uses the same instruction.", text(t6)),
              ("E7", "(describe-first, options before the images), $N{=}3$.", text(t7)),
              ("E8", "(tagged addressing), $N{=}5$, target \\texttt{[IMG\\_3]}.", text(t8)),
              ("E9 C2", "(four separate panels).", text(t9["separate"])),
              ("E9 C3", "(one collage, positional reference). C1 uses the same reference, its own header and a "
               "collage with only the target panel filled.", text(t9["collage_pos"])),
              ("E9 C4", "(one collage, reference by kind): identical to C3 except the reference.",
               diff(t9["collage_kind"], t9["collage_pos"]))]
    out = ["% auto-generated by work/analysis/gpu_results.py from ch.make_parts (text only; <image> marks an image)"]
    for label, cap, body in blocks:
        out += [r"\noindent\textbf{" + label + "} " + cap, r"{\scriptsize", r"\begin{verbatim}", body,
                r"\end{verbatim}", "}"]
    (SEC / "gen_supp_prompts.tex").write_text("\n".join(out) + "\n")


def write_macros(IV, E9, C, A):
    lst = lambda ms: ", ".join(TEX[m] for m in ms) if ms else "none"
    put("iv.models", lst([m for m in IV_MODELS if m in IV]))
    put("e9.models", lst([m for m in E9_MODELS if m in E9]))
    put("causal.models", lst([m for m in CAUSAL_MODELS if m in C]))
    put("iv.nmodels", str(len(IV)))
    put("e9.nmodels", str(len(E9)))
    put("attn.lfm.has", "1" if A else "0")
    # cross-model ranges (only over models present)
    for exp in ("E7", "E8"):
        ds = [IV[m][exp]["d"][0] for m in IV if exp in IV[m]]
        if ds:
            put(f"iv.{exp}.dmin", f(min(ds), 3, True)); put(f"iv.{exp}.dmax", f(max(ds), 3, True))
            put(f"iv.{exp}.nsig", str(sum(M[f"iv.{m}.{exp}.d.sig"] == "1" for m in IV if exp in IV[m])))
            put(f"iv.{exp}.nm", str(len(ds)))
    if E9:
        for key in ("C3mC2.joint", "C4mC3.country", "C2mC1.country", "C2mC1.item", "C3mC1.country", "C3mC1.item"):
            v = [float(M[f"e9.{m}.{key}"].replace("$-$", "-").replace("$+$", "+")) for m in E9]
            put(f"e9.{key}.min", f(min(v), 3, True)); put(f"e9.{key}.max", f(max(v), 3, True))
            put(f"e9.{key}.nsig", str(sum(M[f"e9.{m}.{key}.sig"] == "1" for m in E9)))
        put("e9.icrbelow", str(sum(E9[m]["icr"][c][1] <= E9[m]["icr"][c][2] for m in E9 for c in CONDS[1:])))
        put("e9.icrcells", str(3 * len(E9)))
        ps = [E9[m]["icr"][c][3] for m in E9 for c in CONDS[1:]]
        put("e9.icrsig", str(sum(p < 0.05 for p in ps)))
        o = np.argsort(ps)                       # Holm over all C2-C4 cells present
        holm = np.maximum.accumulate([min(1, (len(ps) - r) * ps[i]) for r, i in enumerate(o)])
        put("e9.icrsigholm", str(int((holm < 0.05).sum())))
        put("e9.icrminpholm", f(float(holm[0]), 2))
        put("e9.icrminp", f(min(ps), 2))
    # per-model lists "\\lfm -0.114 [..], \\gran ..." for the text ($^\\ast$ = partial run)
    star = lambda ok: "" if ok else "$^\\ast$"
    for exp, keys in (("E7", ("d", "ddir", "dwf")), ("E8", ("d", "dwf"))):
        ms = [m for m in IV_MODELS if exp in IV.get(m, {})]
        for kk in keys:
            put(f"iv.{exp}.{kk}.list", "; ".join(f"{TEX[m]} {M[f'iv.{m}.{exp}.{kk}.full']}{star(IV[m][exp]['complete'])}"
                                                 for m in ms) or "pending")
        put(f"iv.{exp}.acc.list", "; ".join(f"{TEX[m]} {M[f'iv.{m}.{exp}.acc']} vs.\\ {M[f'iv.{m}.{exp}.base']}"
                                            f"{star(IV[m][exp]['complete'])}" for m in ms) or "pending")
    for key in ("C3mC2.joint", "C3mC2.country", "C4mC3.country", "C2mC1.country", "C2mC1.item", "C3mC1.country",
                "C3mC1.item", "C2mC1.cmi", "C3mC1.cmi", "C2mC1.joint"):
        put(f"e9.{key}.list", "; ".join(f"{TEX[m]} {M[f'e9.{m}.{key}.full']}{star(E9[m]['complete'])}"
                                        for m in E9_MODELS if m in E9) or "pending")
    # compact lists: value with a dagger where the 95% CI excludes 0; ranges over models
    dag = lambda k: M[k] + ("$^\\dagger$" if M[k + ".sig"] == "1" else "")
    for exp, keys in (("E7", ("d", "ddir")), ("E8", ("d", "dwf"))):
        ms = [m for m in IV_MODELS if exp in IV.get(m, {})]
        for kk in keys:
            put(f"iv.{exp}.{kk}.clist", ", ".join(f"{TEX[m]} {dag(f'iv.{m}.{exp}.{kk}')}{star(IV[m][exp]['complete'])}"
                                                  for m in ms) or "pending")
    for key in ("C3mC2.joint", "C3mC2.country", "C4mC3.country", "C2mC1.country", "C2mC1.item", "C3mC1.country",
                "C3mC1.item", "C2mC1.cmi", "C3mC1.cmi", "C2mC1.joint"):
        ms = [m for m in E9_MODELS if m in E9]
        put(f"e9.{key}.clist", ", ".join(f"{TEX[m]} {dag(f'e9.{m}.{key}')}{star(E9[m]['complete'])}" for m in ms) or "pending")
        v = sorted(float(M[f"e9.{m}.{key}"].replace("$-$", "-").replace("$+$", "+")) for m in ms)
        if v:
            put(f"e9.{key}.range", f(v[0], 3, True) if len(v) == 1 else f"{f(v[0], 3, True)} to {f(v[-1], 3, True)}")
    # direction counts over models present: CI entirely above (npos) / below (nneg) zero
    def counts(prefix, keyfmt, ms):
        lo = {m: EST[keyfmt.format(m)][1] for m in ms}
        hi = {m: EST[keyfmt.format(m)][2] for m in ms}
        put(prefix + ".npos", str(sum(lo[m] > 0 for m in ms)))
        put(prefix + ".nneg", str(sum(hi[m] < 0 for m in ms)))
        put(prefix + ".negmodels", andjoin([TEX[m] for m in ms if hi[m] < 0]))
        put(prefix + ".nullmodels", andjoin([TEX[m] for m in ms if not (hi[m] < 0 or lo[m] > 0)]))
        put(prefix + ".nm", str(len(ms)))
        big = [m for m in ms if m in BIG]          # >=7B models among those present
        put(prefix + ".nbig", str(len(big)))
        put(prefix + ".nnegbig", str(sum(hi[m] < 0 for m in big)))
        put(prefix + ".negbigmodels", andjoin([TEX[m] for m in big if hi[m] < 0]))
        nb, nnb = len(big), sum(hi[m] < 0 for m in big)   # prose for "including ... of the >=7B models"
        put(prefix + ".bigphrase", (f"the only ${{\\geq}}7$B model tested ({TEX[big[0]]})" if nb == 1 else
                                    f"all {['', 'one', 'two', 'three', 'four'][nb] if nb < 5 else nb} ${{\\geq}}7$B models tested" if nnb == nb else
                                    f"{nnb} of the {nb} ${{\\geq}}7$B models tested"
                                    + (f" ({andjoin([TEX[m] for m in big if hi[m] < 0])})" if 0 < nnb < nb else "")))
    for exp, keys in (("E7", ("d", "ddir")), ("E8", ("d", "dwf"))):
        ms = [m for m in IV_MODELS if exp in IV.get(m, {})]
        for kk in keys:
            counts(f"iv.{exp}.{kk}", "iv.{}." + exp + "." + kk, ms)
    for key in ("C3mC2.joint", "C3mC2.country", "C4mC3.country", "C2mC1.joint", "C2mC1.country", "C3mC1.joint"):
        counts(f"e9.{key}", "e9.{}." + key, [m for m in E9_MODELS if m in E9])
    put("e9.icr.list", "; ".join(f"{TEX[m]} " + ", ".join(f"{M[f'e9.{m}.{CN[c]}.icr']}/{M[f'e9.{m}.{CN[c]}.icrbase']}"
                                                          for c in CONDS[1:]) for m in E9_MODELS if m in E9) or "pending")
    partial = [n for n, ok in [(f"{m} {e}", IV[m][e]["complete"]) for m in IV for e in IV[m]]
               + [(f"{m} E9", E9[m]["complete"]) for m in E9] + [(f"{m} causal", C[m]["done"]) for m in C] if not ok]
    put("anypartial", "" if not partial else "~($^\\ast$partial run)")   # leading space lives here
    print("partial:", partial or "none")
    for m, ok in [(f"iv.{m}.{e}", 1) for m in IV for e in IV[m]] + [(f"e9.{m}", 1) for m in E9] + \
                 [(f"causal.{m}", 1) for m in C]:
        put(m + ".has", "1")
    lines = ["% auto-generated by work/analysis/gpu_results.py -- EXPLORATORY, post-freeze. Do not edit; re-run the script.",
             "% Keys are read with \\gpu{key}; an undefined key prints a red [TODO: key]."]
    lines += [f"\\gpudef{{{k}}}{{{v}}}" for k, v in sorted(M.items())]
    (SEC / "gen_macros.tex").write_text("\n".join(lines) + "\n")


def main():
    FIG.mkdir(exist_ok=True)
    pool = ch.load_pool()
    IV, E9, C, A = interventions(pool), collage(pool), causal(), attention_lfm()
    runs()
    prompts()
    fig_e9(E9)
    fig_causal(C)
    tab_expl(IV)
    tab_e9(E9)
    tab_causal(C)
    write_macros(IV, E9, C, A)
    print("E7/E8:", {m: {e: (r["n"], round(r["acc"], 3), [round(x, 3) for x in r["d"]]) for e, r in v.items()} for m, v in IV.items()})
    print("E9:", {m: r["n"] for m, r in E9.items()}, "| causal:", {m: (r["n"], r["done"]) for m, r in C.items()},
          "| attn_lfm:", bool(A))
    print(f"wrote {len(M)} macros to {SEC / 'gen_macros.tex'}")


def _selftest():
    # cluster bootstrap: identical clusters -> zero-width CI; mean is the pooled mean, not the mean of means
    m, lo, hi = cboot([1, 1, 0, 0], ["a", "a", "b", "b"])
    assert m == 0.5 and lo <= m <= hi
    m, lo, hi = cboot([1, 0, 0], ["a", "b", "b"])
    assert abs(m - 1 / 3) < 1e-12
    assert f(-0.1234, 3, True) == "$-$0.123" and f(0.5) == "0.500"


if __name__ == "__main__":
    _selftest()
    main()
