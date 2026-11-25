"""Redraw the CultureHaystack figures on all 884 targets in the co-authors' Nov-24 style (STIX Two Text,
legend on top, solid blue >=7B, dashed warm <=4B, Gemma-3-12B green). Sizes match the figures they replace."""
import json, sys
import numpy as np
import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path
D = json.load(open("work/figs884/figdata.json"))
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "work/figs884/out"); OUT.mkdir(parents=True, exist_ok=True)
PT = 1 / 72
plt.rcParams.update({"font.family": "STIX Two Text", "mathtext.fontset": "stix", "font.size": 7, "axes.titlesize": 7.5,
                     "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.2,
                     "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5, "lines.linewidth": 1.1,
                     "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42, "axes.grid": True,
                     "grid.color": "#ECECEC", "grid.linewidth": 0.5, "legend.frameon": False})
M6 = ["gemma4", "qwen3vl", "qwen25vl", "gemma3_4b", "granite4v", "lfm25v"]
NAME = {"gemma4": "Gemma-4-12B", "qwen3vl": "Qwen3-VL-8B", "qwen25vl": "Qwen2.5-VL-7B", "gemma3_4b": "Gemma-3-4B",
        "granite4v": "Granite-Vision-4B", "lfm25v": "LFM2.5-VL-3B", "gemma3_12b": "Gemma-3-12B"}
COL = {"gemma4": "#1B3A6B", "qwen3vl": "#2F7BC4", "qwen25vl": "#5BB8D0", "gemma3_4b": "#B03A1E", "granite4v": "#E06B26",
       "lfm25v": "#E8A93A", "gemma3_12b": "#2E8B3E"}
MK = {"gemma4": "s", "qwen3vl": "o", "qwen25vl": "D", "gemma3_4b": "o", "granite4v": "s", "lfm25v": "D", "gemma3_12b": "^"}
SMALL = {"gemma3_4b", "granite4v", "lfm25v"}
LS = lambda m: "--" if m in SMALL else "-"
ACCENT = "#C2338F"; GREY = "#9A9A9A"; INK = "#222222"
def fig(wpt, hpt, *a, **k): return plt.subplots(*a, figsize=(wpt * PT, hpt * PT), **k)
def save(f, name): f.savefig(OUT / f"{name}.pdf"); plt.close(f); print("wrote", name)
def curve(ax, m, series, band=True, ms=2.6, lw=None, label=True):
    N = sorted(int(n) for n in series); y = [series[str(n)][0] for n in N]
    ax.plot(N, y, LS(m), color=COL[m], marker=MK[m], ms=ms, lw=lw or 1.1, label=NAME[m] if label else None)
    if band: ax.fill_between(N, [series[str(n)][1] for n in N], [series[str(n)][2] for n in N], color=COL[m], alpha=0.13, lw=0)
def spread(ys, gap):
    """Nudge label positions apart so none are closer than `gap` (keeps order)."""
    idx = sorted(range(len(ys)), key=lambda i: ys[i]); out = list(ys)
    for a, b in zip(idx, idx[1:]):
        if out[b] - out[a] < gap: out[b] = out[a] + gap
    return out
def legend_top(f, models, ncol=6, y=1.0):
    h = [Line2D([], [], color=COL[m], ls=LS(m), marker=MK[m], ms=2.6, label=NAME[m]) for m in models]
    f.legend(handles=h, loc="upper center", ncol=ncol, bbox_to_anchor=(0.5, y), handlelength=2.2, columnspacing=1.2)

# ---- Fig. S (accuracy vs N): MC and open
f, ax = fig(428.562, 203.351, 1, 2, sharey=True)
for m in M6 + ["gemma3_12b"]:
    curve(ax[0], m, D["acc"][m]["mc"])
    if "open" in D["acc"][m]: curve(ax[1], m, D["acc"][m]["open"])
for a, t in zip(ax, ["Multiple choice", "Open-ended (lower bounds)"]):
    a.set_title(t); a.set_xticks(range(1, 6)); a.set_xlabel("photos shown ($N$)"); a.set_ylim(0, 1)
ax[0].set_ylabel("joint accuracy"); legend_top(f, M6 + ["gemma3_12b"], ncol=7); f.tight_layout(rect=(0, 0, 1, 0.9)); save(f, "fig1_accuracy_vs_N")

# ---- Fig. 1d: all targets vs known alone (small)
f, ax = fig(113.386, 103.465, 1, 2, sharey=True)
for m in M6:
    curve(ax[0], m, D["acc"][m]["mc"], ms=1.4, lw=0.8, label=False); curve(ax[1], m, D["known"][m], ms=1.4, lw=0.8, label=False)
for a, t, s in zip(ax, ["all targets", "known alone"], ["pre-registered", "exploratory"]):
    a.set_title(f"{t}\n{{\\it {s}}}".replace("{\\it ", "").replace("}", ""), fontsize=5.6, pad=2)
    a.set_xticks(range(1, 6)); a.tick_params(labelsize=4.8, length=1.5, pad=1); a.set_ylim(0.2, 1.02); a.set_xlabel("photos ($N$)", fontsize=5, labelpad=1)
ax[0].set_ylabel("MC joint accuracy", fontsize=5, labelpad=1); f.subplots_adjust(left=0.2, right=0.98, bottom=0.2, top=0.8, wspace=0.12); save(f, "fig0_d")

# ---- Error anatomy vs N
f, axs = fig(428.562, 321.23, 2, 3, sharey=True)
for a, m in zip(axs.flat, M6):
    s = D["errshares"][m]; N = [2, 3, 4, 5]
    wi = [s[str(n)][0] for n in N]; pdv = [s[str(n)][1] for n in N]; fe = [s[str(n)][2] for n in N]
    a.bar(N, wi, color=ACCENT, width=0.65); a.bar(N, pdv, bottom=wi, color=GREY, width=0.65)
    a.bar(N, fe, bottom=np.add(wi, pdv), color="#333333", width=0.65)
    for n, v in zip(N, wi): a.text(n, v / 2, f"{v:.2f}", ha="center", va="center", color="white", fontsize=5.6)
    a.set_title(NAME[m]); a.set_xticks(N); a.set_ylim(0, 1); a.grid(False)
for a in axs[1]: a.set_xlabel("photos shown ($N$)")
for a in axs[:, 0]: a.set_ylabel("share of wrong fields")
h = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (ACCENT, GREY, "#333333")]
f.legend(h, ["wrong image (another shown image; share printed)", "prior drift (not shown)", "format error"], loc="lower center", ncol=3)
f.tight_layout(rect=(0, 0.06, 1, 1)); save(f, "fig2_error_anatomy")

# ---- Slot profile at N=5
f, ax = fig(238.843, 225.129)
for m in M6: curve(ax, m, D["slots"][m])
ax.set_xticks(range(1, 6)); ax.set_xlabel("slot of the asked image ($N{=}5$)"); ax.set_ylabel("joint accuracy"); ax.set_ylim(0, 1)
legend_top(f, M6, ncol=2); f.tight_layout(rect=(0, 0, 1, 0.8)); save(f, "fig3_position_N5")

# ---- Describe-first minus direct, by N (two panels by size)
f, ax = fig(243.061, 191.101, 2, 1, sharex=True)
for a, grp, t in ((ax[0], ["gemma4", "qwen3vl", "qwen25vl"], "${\\geq}$7B"), (ax[1], ["gemma3_4b", "lfm25v"], "${\\leq}$4B")):
    for m in grp:
        if m in D["describe"]: curve(a, m, D["describe"][m])
    a.axhline(0, color=INK, lw=0.6, ls=":"); a.text(0.01, 0.92, t, transform=a.transAxes, fontsize=6.5, va="top"); a.set_ylim(-0.35, 0.15)
ax[1].set_xticks(range(1, 6)); ax[1].set_xlabel("photos shown ($N$)"); f.supylabel("describe-first $-$ direct (accuracy)", fontsize=7)
legend_top(f, ["gemma4", "qwen3vl", "qwen25vl", "gemma3_4b", "lfm25v"], ncol=3); f.tight_layout(rect=(0, 0, 1, 0.86)); save(f, "fig4_describe_rescue")

# ---- Near vs far wrong-image rate (dumbbell)
mods = [m for m in M6 if m in D["nearfar"]]
f, ax = fig(238.899, 203.785)
for i, m in enumerate(mods[::-1]):
    near, far, d, lo, hi = D["nearfar"][m]
    ax.plot([far, near], [i, i], color=COL[m], lw=1.2); ax.plot(near, i, "o", color=COL[m], ms=4); ax.plot(far, i, "o", mfc="white", mec=COL[m], ms=4)
    ax.text(1.02, i, f"{d:+.2f} [{lo:+.2f}, {hi:+.2f}]", transform=ax.get_yaxis_transform(), va="center", fontsize=5.8)
ax.set_yticks(range(len(mods))); ax.set_yticklabels([NAME[m] for m in mods[::-1]]); ax.set_xlim(0, 0.65)
ax.set_xlabel("share of answer fields naming\nanother shown photo ($N{=}5$)")
ax.text(1.02, len(mods) - 0.4, "near $-$ far [95% CI]", transform=ax.get_yaxis_transform(), fontsize=5.8, color=GREY)
ax.legend(handles=[Line2D([], [], marker="o", color=INK, ls="", ms=4, label="near list"), Line2D([], [], marker="o", mfc="white", mec=INK, ls="", ms=4, label="far list")],
          loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
f.subplots_adjust(left=0.3, right=0.66, top=0.88, bottom=0.2); save(f, "fig5_near_far")

# ---- Retention on known targets + reworded-prompt control
f, ax = fig(495.84, 175.2, 1, 2, gridspec_kw={"width_ratios": [1.15, 1]})
for m in M6: curve(ax[0], m, D["known"][m])
ax[0].set_xticks(range(1, 6)); ax[0].set_xlabel("Photos shown ($N$)"); ax[0].set_ylabel("MC joint accuracy on targets\nknown at $N{=}1$"); ax[0].set_ylim(0.3, 1.02)
ax[0].set_title("(a)  Accuracy at $N$ on targets known at $N{=}1$", loc="left")
_ys = spread([D["known"][m]["5"][0] for m in M6], 0.035)
for m, y in zip(M6, _ys): ax[0].text(5.08, y, NAME[m], color=COL[m], fontsize=5.6, va="center")
ax[0].set_xlim(0.8, 6.2)
mods = [m for m in M6 if m in D["rtm"]]
for i, m in enumerate(mods[::-1]):
    r1, l1, h1, r5, l5, h5 = D["rtm"][m]
    ax[1].plot([r5, r1], [i, i], color="#BBBBBB", lw=2.5, solid_capstyle="butt")
    ax[1].errorbar(r1, i, xerr=[[r1 - l1], [h1 - r1]], fmt="o", mfc="white", mec=COL[m], ecolor=COL[m], ms=3.5, elinewidth=0.7)
    ax[1].errorbar(r5, i, xerr=[[r5 - l5], [h5 - r5]], fmt="o", color=COL[m], ms=3.5, elinewidth=0.7)
    ax[1].text(0.3, i, NAME[m], color=COL[m], fontsize=5.8, va="center", ha="right")
ax[1].set_yticks([]); ax[1].set_xlim(0.3, 1.02); ax[1].set_xlabel("Retention of $N{=}1$-known targets")
ax[1].set_title("(b)  Reworded-prompt control", loc="left")
ax[1].legend(handles=[Line2D([], [], marker="o", mfc="white", mec=INK, ls="", ms=3.5, label="$N{=}1$, reworded"), Line2D([], [], marker="o", color=INK, ls="", ms=3.5, label="$N{=}5$, reworded (95% CI)")],
             loc="lower left", bbox_to_anchor=(0.0, 1.07), ncol=2)
f.tight_layout(w_pad=6); save(f, "figA_retention_known_targets")

# ---- Per-image cost forest
groups = [("", ["All targets"]), ("Category", ["Food", "Landmark", "Instrument", "Attire"]), ("Popularity tier", ["High", "Mid", "Low"]), ("", ["Gold letters differ"])]
f, ax = fig(267.36, 312.48); y = 0; ticks = []; labels = []
for head, rows in groups:
    if head: ax.text(-0.155, y, head, fontsize=6.8, style="italic", va="center"); y -= 1
    for r in rows:
        for j, m in enumerate(M6):
            s, lo, hi = D["forest"][r][m]; yy = y - 0.13 * (j - 2.5)
            ax.errorbar(s, yy, xerr=[[s - lo], [hi - s]], fmt=MK[m], color=COL[m], ms=2.2, elinewidth=0.7, ls="none")
            ax.plot([lo, hi], [yy, yy], LS(m), color=COL[m], lw=0.7)
        ticks.append(y); labels.append(r if r != "Gold letters differ" else "Gold letters differ\n(item vs. country)"); y -= 1.2
ax.set_yticks(ticks); ax.set_yticklabels(labels, fontsize=6.4); ax.axvline(0, color=INK, lw=0.6); ax.set_xlim(-0.155, 0.06); ax.grid(axis="y", visible=False)
ax.set_xlabel("Per-added-photo change in MC joint accuracy ($N{=}1$ to 5)"); legend_top(f, M6, ncol=3); f.tight_layout(rect=(0, 0, 1, 0.92)); save(f, "figB_per_image_cost_forest")

# ---- Wrong-pick features
feats = [("same_region", "Same world region\nas target"), ("adj", "Adjacent slot\n($|j-k|{=}1$)"), ("later", "Later slot\n($j>k$)"),
         ("first", "Image 1"), ("last", "Image 5"), ("let_adj", "Adjacent option\nletter")]
f, axs = fig(493.44, 176.16, 1, len(feats), sharey=True)
for a, (k, t) in zip(axs, feats):
    for i, m in enumerate(M6[::-1]):
        e, lo, hi, p = D["wrongpick"][k][m]
        a.errorbar(e, i, xerr=[[e - lo], [hi - e]], fmt=MK[m], color=COL[m], ms=2.4, elinewidth=0.7)
        a.plot([lo, hi], [i, i], LS(m), color=COL[m], lw=0.7)
        if p < 0.05: a.text(hi + 0.008, i, "*", color=COL[m], fontsize=7, va="center")
    if k == "same_region":
        e, lo, hi = D["wrongpick_pooled_region"]; a.errorbar(e, len(M6), xerr=[[e - lo], [hi - e]], fmt="D", color=INK, ms=3, elinewidth=0.8)
    else: a.text(0, len(M6), "not pooled", fontsize=5.5, color=GREY, ha="center", va="center")
    a.axvline(0, color=INK, lw=0.5, ls=":"); a.set_title(t, fontsize=6.8); a.set_xlim(-0.15, 0.2); a.tick_params(labelsize=5.8)
axs[0].set_yticks(range(len(M6) + 1)); axs[0].set_yticklabels([NAME[m] for m in M6[::-1]] + ["Pooled (6 models)"], fontsize=6.2)
for t, m in zip(axs[0].get_yticklabels(), M6[::-1] + ["x"]): t.set_color(COL.get(m, INK))
f.text(0.5, 0.02, "Observed $-$ expected share of wrong picks at $N{=}5$ (null: uniform over the 4 shown distractors). * Holm $p<0.05$", ha="center", fontsize=6.8)
f.tight_layout(rect=(0, 0.06, 1, 1)); save(f, "figC_wrong_pick_features")

# ---- Fig. 4 (Glance): (a) near options with five photos vs alone, (b) same image if both wrong, (c) accuracy on 884
f, ax = fig(483.093, 119.63, 1, 3, gridspec_kw={"width_ratios": [1.15, 1.05, 1]})
order = ["gemma4", "gemma3_12b", "qwen3vl", "qwen25vl", "gemma3_4b", "granite4v", "lfm25v"]
for i, m in enumerate(order[::-1]):
    if m in D["g1_five"]:
        e, lo, hi = D["g1_five"][m]; ax[0].errorbar(e, i + 0.15, xerr=[[e - lo], [hi - e]], fmt="o", color=COL[m], ms=3, elinewidth=0.7)
    e, lo, hi = D["g1"][m]; ax[0].errorbar(e, i - 0.15, xerr=[[e - lo], [hi - e]], fmt="o", mfc="white", mec=COL[m], ecolor=COL[m], ms=3, elinewidth=0.7)
ax[0].set_yticks(range(len(order))); ax[0].set_yticklabels([NAME[m] for m in order[::-1]], fontsize=6)
for t, m in zip(ax[0].get_yticklabels(), order[::-1]): t.set_color(COL[m])
ax[0].axvline(0, color=INK, lw=0.5); ax[0].set_xlabel("Accuracy lost, near vs. far options", fontsize=6.5); ax[0].set_xlim(-0.2, 0.02)
ax[0].legend(handles=[Line2D([], [], marker="o", color=INK, ls="", ms=3, label="five photos"), Line2D([], [], marker="o", mfc="white", mec=INK, ls="", ms=3, label="target alone")], loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=5.8)
ax[0].set_title("(a)", loc="left", x=-0.5)
sm = ["gemma4", "qwen3vl", "qwen25vl", "gemma3_4b", "lfm25v"]
for i, m in enumerate(sm[::-1]):
    for key, fill, dy in (("same_joint", True, 0.15), ("same_split", False, -0.15)):
        e, lo, hi = [100 * x for x in D["split"][m][key]]
        ax[1].errorbar(e, i + dy, xerr=[[e - lo], [hi - e]], fmt="o", mfc=COL[m] if fill else "white", mec=COL[m], ecolor=COL[m], ms=3, elinewidth=0.7)
ax[1].axvline(25, color=GREY, lw=0.6, ls="--"); ax[1].text(25.5, len(sm) - 0.5, "chance", fontsize=5.5, color=GREY)
ax[1].set_yticks(range(len(sm))); ax[1].set_yticklabels([NAME[m] for m in sm[::-1]], fontsize=6)
for t, m in zip(ax[1].get_yticklabels(), sm[::-1]): t.set_color(COL[m])
ax[1].set_xlim(15, 100); ax[1].set_xlabel("Same image if both wrong (%)", fontsize=6.5)
ax[1].legend(handles=[Line2D([], [], marker="o", color=INK, ls="", ms=3, label="joint prompt"), Line2D([], [], marker="o", mfc="white", mec=INK, ls="", ms=3, label="split query")], loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=5.8)
ax[1].set_title("(b)", loc="left", x=-0.5)
for m in order: curve(ax[2], m, D["acc"][m]["mc"], ms=2, lw=0.9, label=False)
_ys = spread([D["acc"][m]["mc"]["5"][0] for m in order], 0.045)
for m, y in zip(order, _ys): ax[2].text(5.15, y, NAME[m], fontsize=4.8, color=COL[m], va="center")
ax[2].set_xticks(range(1, 6)); ax[2].set_xlim(0.8, 6.6); ax[2].set_ylim(0.25, 1.0); ax[2].set_xlabel("Photos shown ($N$), 884 targets", fontsize=6.5)
ax[2].set_ylabel("MC joint accuracy", fontsize=6.5); ax[2].set_title("(c)", loc="left", x=-0.3)
f.tight_layout(w_pad=2.5); save(f, "figGlance")

# ---- One-photo option control forest (860)
f, ax = fig(241.92, 136.32); order1 = ["gemma4", "qwen3vl", "qwen25vl", "gemma3_4b", "granite4v", "lfm25v", "gemma3_12b"]
for i, m in enumerate(order1[::-1]):
    e, lo, hi = D["g1"][m]; ax.errorbar(e, i, xerr=[[e - lo], [hi - e]], fmt=MK[m], color=COL[m], ms=3, elinewidth=0.9)
    ax.text(1.03, i, f"{e:+.3f} [{lo:+.3f}, {hi:+.3f}]".replace("+", "−" if False else "+"), transform=ax.get_yaxis_transform(), fontsize=5.6, va="center")
ax.set_yticks(range(len(order1))); ax.set_yticklabels([NAME[m] for m in order1[::-1]], fontsize=6)
for t, m in zip(ax.get_yticklabels(), order1[::-1]): t.set_color(COL[m])
ax.axvline(0, color=INK, lw=0.5); ax.set_xlim(-0.18, 0.0); ax.set_xlabel("Near $-$ far options, MC joint accuracy ($N{=}1$)")
ax.text(1.03, len(order1) - 0.3, "difference [95% CI]", transform=ax.get_yaxis_transform(), fontsize=5.6, color=GREY)
f.subplots_adjust(left=0.27, right=0.62, bottom=0.22, top=0.95); save(f, "figS_g1_near_far_forest")

# ---- Split query vs original prompt (accuracy)
f, ax = fig(233.28, 123.36)
for i, m in enumerate(sm[::-1]):
    for key, hatch, dy in (("acc_joint", None, 0.18), ("acc_split", "////", -0.18)):
        e, lo, hi = D["split"][m][key]
        ax.barh(i + dy, e, height=0.34, color=COL[m] if hatch is None else "white", edgecolor=COL[m], hatch=hatch, lw=0.6)
        ax.errorbar(e, i + dy, xerr=[[e - lo], [hi - e]], fmt="none", ecolor=INK, elinewidth=0.6)
ax.set_yticks(range(len(sm))); ax.set_yticklabels([NAME[m] for m in sm[::-1]], fontsize=6)
for t, m in zip(ax.get_yticklabels(), sm[::-1]): t.set_color(COL[m])
ax.set_xlim(0, 1); ax.set_xlabel("MC joint accuracy at $N{=}5$"); ax.grid(axis="y", visible=False)
ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, fc="#BBBBBB", ec="#888888"), plt.Rectangle((0, 0), 1, 1, fc="white", ec="#888888", hatch="////")],
          labels=["Original prompt (both fields)", "Split query (one field per call)"], loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=1, fontsize=5.8)
f.tight_layout(); save(f, "figS_g2_split_vs_frozen")
