"""Results-at-a-glance figure for Sec. 5.6 (POST-REGISTRATION / EXPLORATORY), frozen paper style (style.py).

(a) Full pool (884 targets, all main-kind MC trials pooled; same selection as Supp. Tab. S12 and the Sec. 5.6 text):
    MC joint accuracy at N = 1, 3, 5 with 95% target-bootstrap bands, five models + Gemma-3-12B.
    Source: cr_analyses/01_fullpool_ci/fullpool_ci.csv (selection == firstlook).
(b) Near vs far answer options, 152 near-feasible paper targets: accuracy lost with near options
    - with five photos (frozen near-distractor test, N=5; results/near_vs_far.csv; Gemma-3-12B from its
      post-registration main run, computed with frozen analyze.near_far: 0.112 [0.040, 0.184])
    - with the target alone (one-photo option control G1; package results/g1_near_far.csv, frozen160).
(c) Same image if both wrong (whole / (whole + two-image mix)), frozen joint prompt vs split query,
    800 N=5 trials per model; cr_analyses/02_g2_same_image/g2_same_image.csv. Gemma-3-12B was not run in G2.

Usage: python make_glance.py [A|B]  -> figGlance_A.pdf/png or figGlance_B.pdf/png
"""
import sys
sys.path.insert(0, "wacv/wacv-lvr/work/analysis")
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, ConnectionPatch, Polygon
import style as S

S.apply()
R = "wacv/"
COL = {**S.COL, "gemma3_12b": "#3e8e5e"}           # validated vs co-plotted models (dataviz validator)
NAME = {**S.NAME, "gemma3_12b": "Gemma-3-12B"}
LS = {**S.LS, "gemma3_12b": "-"}
MARK = {**S.MARK, "gemma3_12b": "^"}
ROWS = ["gemma4", "gemma3_12b", "qwen3vl", "qwen25vl", "gemma3_4b", "lfm25v"]   # large -> small; Granite excluded (failed repro)
LW, ELW = 0.8, 0.6

# ---------------- data (all read from result files) ----------------
fp = pd.read_csv(R + "cr_analyses/01_fullpool_ci/fullpool_ci.csv")
fp = fp[(fp.selection == "firstlook") & (fp.metric == "joint")]
nf = pd.read_csv(R + "wacv-lvr/results/near_vs_far.csv")
nf = nf[(nf["mode"] == "mc") & (nf.N == 5) & (nf.metric == "joint_acc")].set_index("model")
five = {m: (r.far_minus_near, r.lo, r.hi) for m, r in nf.iterrows()}
five["gemma3_12b"] = (0.11184210526315785, 0.039473684210526314, 0.18421052631578946)  # frozen near_far on g3 main run (see docstring)
g1 = pd.read_csv(R + "g3_table1/dl/tar/fullpool/results/g1_near_far.csv")
g1 = g1[(g1.subset == "frozen160") & (g1.metric == "joint_ok")].set_index("model")
one = {m: (-r.near_minus_far, -r.hi, -r.lo) for m, r in g1.iterrows()}           # as a loss (far - near)
g2 = pd.read_csv(R + "cr_analyses/02_g2_same_image/g2_same_image.csv").set_index(["model", "prompt"])
same = {(m, p): (100 * r.same_frozen_def, 100 * r.same_frozen_def_lo, 100 * r.same_frozen_def_hi) for (m, p), r in g2.iterrows()}


def repel(ys, gap):
    """Spread label y positions to a minimum gap, keeping order and the mean position."""
    order = np.argsort(ys); y = np.array(ys, float)[order]
    for _ in range(200):
        moved = False
        for i in range(1, len(y)):
            if y[i] - y[i - 1] < gap:
                d = (gap - (y[i] - y[i - 1])) / 2; y[i] += d; y[i - 1] -= d; moved = True
        if not moved:
            break
    out = np.empty_like(y); out[order] = y
    return out


FS = S.FS_SMALL   # 6.5 pt floor for every text element


def panel_curves(axz, ax):
    """(c): full-range panel (ax) with a thin box on the >=7B cluster, magnified in the strip above (axz)."""
    x0, x1, y0, y1 = 0.8, 5.2, 0.775, 0.885
    for a in (axz, ax):
        for m in ROWS:
            d = fp[fp.model == m].sort_values("N")
            a.fill_between(d.N, d.lo, d.hi, color=COL[m], alpha=0.07 if a is axz else 0.13, lw=0)
            a.plot(d.N, d.est, ls=LS[m], color=COL[m], lw=1.0, marker=MARK[m], ms=2.8, mec="white", mew=0.4, zorder=3)
        a.set_xlim(x0, x1)
    ax.set_ylim(0.25, 0.92); ax.set_yticks([0.3, 0.5, 0.7, 0.9]); ax.set_xticks([1, 3, 5])
    ax.set_xlabel("Photos shown ($N$), 884 targets")
    ax.text(-0.2, 0.75, "MC joint accuracy", rotation=90, transform=ax.transAxes, ha="center", va="center", fontsize=S.FS)
    axz.set_ylim(y0, y1); axz.set_yticks([0.80, 0.85]); axz.set_xticks([1, 3, 5]); axz.set_xticklabels([])
    axz.tick_params(axis="x", length=0)
    for sp in axz.spines.values():
        sp.set_visible(True); sp.set_linewidth(0.6); sp.set_color(S.MUTED)
    ax.add_patch(Rectangle((x0 + 0.03, y0), x1 - x0 - 0.06, y1 - y0, fill=False, lw=0.6, ec=S.MUTED, zorder=6))
    # light wedge from the box to the magnified strip
    fig = ax.figure
    wx = [ax.transData.transform((x0 + 0.03, y1)), ax.transData.transform((x1 - 0.03, y1)),
          axz.transAxes.transform((1, 0)), axz.transAxes.transform((0, 0))]
    inv = fig.transFigure.inverted()
    fig.add_artist(Polygon([inv.transform(p) for p in wx], closed=True, fc=S.GRID, ec=S.GREY, lw=0.4, alpha=0.55, zorder=0))
    def endlabels(a, models, gap):
        ys = [fp[(fp.model == m) & (fp.N == 5)].est.iloc[0] for m in models]
        for m, y, yl in zip(models, ys, repel(ys, gap)):
            a.annotate(NAME[m], (5, y), xytext=(5.42, yl), textcoords="data", va="center", fontsize=FS, color=S.INK,
                       arrowprops=dict(arrowstyle="-", lw=0.4, color=S.GREY, shrinkA=0, shrinkB=2.5), annotation_clip=False)
    endlabels(axz, ("gemma4", "gemma3_12b", "qwen3vl"), 0.032)
    endlabels(ax, ("qwen25vl", "gemma3_4b", "lfm25v"), 0.07)


def panel_a_slope(ax):
    for m in ROWS:
        d = fp[(fp.model == m) & fp.N.isin([1, 5])].sort_values("N")
        ax.plot([0, 1], d.est, ls=LS[m], color=COL[m], lw=1.0, marker=MARK[m], ms=3.2, mec="white", mew=0.4)
        for x, (_, r) in zip([0, 1], d.iterrows()):
            ax.plot([x, x], [r.lo, r.hi], color=COL[m], lw=ELW, alpha=0.7)
    y5 = [fp[(fp.model == m) & (fp.N == 5)].est.iloc[0] for m in ROWS]
    for m, y, yl in zip(ROWS, y5, repel(y5, 0.034)):
        ax.annotate(NAME[m], (1, y), xytext=(1.08, yl), textcoords="data", va="center", fontsize=S.FS_SMALL - 0.5, color=S.INK,
                    arrowprops=dict(arrowstyle="-", lw=0.4, color=S.GREY, shrinkA=0, shrinkB=2), annotation_clip=False)
    ax.set_xlim(-0.15, 1.1); ax.set_ylim(0.25, 0.92); ax.set_xticks([0, 1]); ax.set_xticklabels(["$N{=}1$", "$N{=}5$"])
    ax.set_ylabel("MC joint accuracy")


def dumbbell(ax, vals, rows, xlim, xlabel, ref=None, reflabel=None, keys=("", ""), arrow=False):
    for i, m in enumerate(rows):
        y = len(rows) - 1 - i
        a, b = vals.get((m, 0)), vals.get((m, 1))
        if a is None and b is None:
            ax.text(np.mean(xlim), y, "not run", ha="center", va="center", fontsize=FS, color=S.MUTED, style="italic")
            continue
        if a is not None and b is not None:
            if arrow and abs(b[0] - a[0]) > 0.04 * (xlim[1] - xlim[0]):
                ax.annotate("", xy=(b[0], y), xytext=(a[0], y), arrowprops=dict(arrowstyle="-|>,head_length=0.25,head_width=0.12", lw=0.6, color=S.GREY, shrinkA=2.2, shrinkB=2.6), zorder=1)
            else:
                ax.plot([a[0], b[0]], [y, y], color=S.GREY, lw=0.6, zorder=1)
        for k, v in ((0, a), (1, b)):
            if v is None:
                continue
            off = 0.13 if k == 0 else -0.13
            ax.plot([v[1], v[2]], [y + off] * 2, color=COL[m], lw=ELW, alpha=0.55, solid_capstyle="butt")
            ax.plot(v[0], y + off * 0 , marker=MARK[m], ms=3.6, color=COL[m] if k == 0 else "white", mec=COL[m], mew=0.8, zorder=3)
    if ref is not None:
        ax.axvline(ref, color=S.GREY, lw=0.6, ls=(0, (2, 2)), zorder=0)
        ax.text(ref - 1.5, -0.55, reflabel, ha="right", va="bottom", fontsize=FS, color=S.MUTED)
    ax.set_ylim(-0.6, len(rows) - 0.4); ax.set_xlim(*xlim)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([])
    ax.grid(axis="y", visible=False); ax.grid(axis="x", visible=True, color=S.GRID, lw=0.4)
    ax.set_xlabel(xlabel)
    ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)


def legend_key(ax, labels, x=0.0, y=1.02):
    """Tiny filled/hollow key above a panel (marker shape is generic here)."""
    ax.plot([], [])
    t = ax.transAxes
    ax.plot([x + 0.02], [y + 0.045], marker="o", ms=3.4, color=S.MUTED, mec=S.MUTED, transform=t, clip_on=False)
    ax.text(x + 0.05, y + 0.045, labels[0], transform=t, va="center", fontsize=FS, color=S.INK)
    ax.plot([x + 0.52], [y + 0.045], marker="o", ms=3.4, color="white", mec=S.MUTED, mew=0.8, transform=t, clip_on=False)
    ax.text(x + 0.55, y + 0.045, labels[1], transform=t, va="center", fontsize=FS, color=S.INK)


def build(variant):
    fig = plt.figure(figsize=(6.9, 1.75))
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1.05], height_ratios=[0.62, 1], wspace=0.17, hspace=0.32,
                          left=0.118, right=0.885, top=0.85, bottom=0.2)
    axb = fig.add_subplot(gs[:, 0]); axc = fig.add_subplot(gs[:, 1])
    axz = fig.add_subplot(gs[0, 2]); axa = fig.add_subplot(gs[1, 2])
    vb = {}
    for m in ROWS:
        vb[(m, 0)] = five.get(m); vb[(m, 1)] = one.get(m)
    dumbbell(axb, vb, ROWS, (-0.012, 0.24), "Accuracy lost with near options")
    legend_key(axb, ("five photos", "target alone"))
    for i, m in enumerate(ROWS):
        axb.text(-0.04, len(ROWS) - 1 - i, NAME[m], ha="right", va="center", fontsize=FS, color=S.INK, transform=axb.get_yaxis_transform())
    vc = {}
    for m in ROWS:
        if (m, "frozen") in same:
            vc[(m, 0)] = same[(m, "frozen")]; vc[(m, 1)] = same[(m, "split")]
    dumbbell(axc, vc, ROWS, (0, 100), "Same image if both wrong (%)", ref=25, reflabel="chance", arrow=True)
    legend_key(axc, ("joint prompt", "split query"))
    panel_curves(axz, axa)
    S.panel(axb, "a", -0.45, 1.2); S.panel(axc, "b", -0.04, 1.2); S.panel(axz, "c", -0.2, 1.38)
    return fig


if __name__ == "__main__":
    v = sys.argv[1] if len(sys.argv) > 1 else "A"
    f = build(v)
    S.save(f, f"figGlance_{v}", bbox_inches="tight", pad_inches=0.01)
    print("saved", v)
