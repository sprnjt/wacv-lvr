"""Task 8 (POST-REGISTRATION / EXPLORATORY). Supplement figures in the frozen paper style (wacv-lvr/work/analysis/style.py:
colours, order large->small, serif 7.5/6.5 pt, vector PDF + PNG). Gemma-3-12B is not in the frozen palette: added as a
cool (>=7B) teal, solid line, triangle marker. Run with cr_analyses/.venv_clip (has matplotlib? -> uses .venv)."""
import sys, json
sys.path.insert(0, "wacv/wacv-lvr/work/analysis")
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import style as S
S.apply()
R = "wacv/"
PK = R + "g3_table1/dl/tar/fullpool/results/"
COL = {**S.COL, "gemma3_12b": "#3e8e5e"}; NAME = {**S.NAME, "gemma3_12b": "Gemma-3-12B"}
LS = {**S.LS, "gemma3_12b": "-"}; MARK = {**S.MARK, "gemma3_12b": "^"}
M6 = ["gemma4", "gemma3_12b", "qwen3vl", "qwen25vl", "gemma3_4b", "lfm25v"]
M5 = [m for m in M6 if m != "gemma3_12b"]

# (a) G1 forest
g1 = pd.read_csv(PK + "g1_near_far.csv"); g1 = g1[(g1.subset == "all860") & (g1.metric == "joint_ok")].set_index("model")
fig, ax = plt.subplots(figsize=(3.25, 1.75))
for y, m in enumerate(M6[::-1]):
    r = g1.loc[m]; S.errbar(ax, [r.near_minus_far], [y], None, None, COL[m]) if False else \
        ax.errorbar(r.near_minus_far, y, xerr=[[r.near_minus_far - r.lo], [r.hi - r.near_minus_far]], fmt=MARK[m], color=COL[m], ms=3.2, elinewidth=0.8)
    ax.annotate(f"{r.near_minus_far:+.3f} [{r.lo:+.3f}, {r.hi:+.3f}]", (0.02, y), xycoords=("axes fraction", "data"), fontsize=S.FS_SMALL - 0.5, color=S.MUTED, va="center")
ax.axvline(0, color=S.GREY, lw=0.7, zorder=0)
ax.set_yticks(range(len(M6))); ax.set_yticklabels([NAME[m] for m in M6[::-1]])
for t, m in zip(ax.get_yticklabels(), M6[::-1]): t.set_color(COL[m])
ax.set_xlim(-0.30, 0.03); ax.grid(axis="x"); ax.grid(axis="y", visible=False)
ax.set_xlabel("Near $-$ far options, MC joint accuracy ($N{=}1$, 860 targets)")
fig.tight_layout(pad=0.3); S.save(fig, "figS_g1_near_far_forest", bbox_inches="tight"); plt.close(fig)

# (b) full pool accuracy and illusory rate vs N
fp = pd.read_csv(R + "cr_analyses/01_fullpool_ci/fullpool_ci.csv"); fp = fp[fp.selection == "core"]
fig, axs = plt.subplots(1, 2, figsize=(3.25, 1.65))
for ax, met, yl in ((axs[0], "joint", "MC joint accuracy"), (axs[1], "illusory", "Other-photo answer rate")):
    for m in M6:
        d = fp[(fp.model == m) & (fp.metric == met)].sort_values("N")
        ax.plot(d.N, d.est, ls=LS[m], marker=MARK[m], color=COL[m], ms=2.6, lw=1.0, label=NAME[m])
        S.ci(ax, d.N, d.lo, d.hi, COL[m], alpha=0.13)
    ax.set_xticks([1, 3, 5]); ax.set_xlabel("Photos shown ($N$)"); ax.set_ylabel(yl)
axs[0].set_ylim(0.25, 0.92); axs[1].set_ylim(-0.01, 0.62)
S.panel(axs[0], "a", -0.36, 1.1); S.panel(axs[1], "b", -0.36, 1.1)
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.02), fontsize=S.FS_SMALL - 0.5, handlelength=1.8, columnspacing=0.8)
fig.tight_layout(pad=0.3, rect=(0, 0.16, 1, 1), w_pad=1.6); S.save(fig, "figS_fullpool_vs_N", bbox_inches="tight"); plt.close(fig)

# (c) G2 split query vs frozen prompt
g2 = pd.read_csv(R + "cr_analyses/02_g2_same_image/g2_same_image.csv").set_index(["model", "prompt"])
fig, ax = plt.subplots(figsize=(3.25, 1.6)); w = 0.36
for x, m in enumerate(M5):
    for k, (p, off, fc) in enumerate((("frozen", -w / 2, COL[m]), ("split", w / 2, "white"))):
        r = g2.loc[(m, p)]
        ax.bar(x + off, r.joint, w, color=fc, edgecolor=COL[m], lw=0.8, hatch=None if p == "frozen" else "////")
        ax.plot([x + off] * 2, [r.joint_lo, r.joint_hi], color=S.INK, lw=0.6)
ax.set_xticks(range(len(M5))); ax.set_xticklabels(["-\n".join(NAME[m].rsplit("-", 1)) for m in M5], fontsize=S.FS_SMALL - 0.5)
for t, m in zip(ax.get_xticklabels(), M5): t.set_color(COL[m])
ax.set_ylabel("MC joint accuracy ($N{=}5$)"); ax.set_ylim(0, 1)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(facecolor=S.MUTED, edgecolor=S.MUTED, label="Original prompt (both fields)"),
                   Patch(facecolor="white", edgecolor=S.MUTED, hatch="////", label="Split query (one field per call)")],
          loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.16), fontsize=S.FS_SMALL - 0.5)
fig.tight_layout(pad=0.3); S.save(fig, "figS_g2_split_vs_frozen", bbox_inches="tight"); plt.close(fig)

# (d) worked example img_0240
pool = pd.read_csv(R + "wacv-lvr/data/pool.csv").set_index("image_id")
IMG = R + "cr_analyses/03_clip_shown/work/images/prepared/"
g1t = {json.loads(l)["trial_id"]: json.loads(l) for l in open(R + "g3_table1/dl/tar/fullpool/trials_g1.jsonl")}
tr = next(json.loads(l) for l in open(R + "wacv-lvr/trials_main.jsonl") if '"E1_img_0240_mixed_N3_k1_P1"' in l)
raw = {}
for m in ["gemma4", "qwen3vl", "qwen25vl", "gemma3_4b", "granite4v", "lfm25v"]:
    raw[m] = next(json.loads(l)["raw"] for l in open(R + f"wacv-lvr/raw_{m}.jsonl") if '"E1_img_0240_mixed_N3_k1_P1", "mode": "mc"' in l)
raw["gemma3_12b"] = next(json.loads(l)["raw"] for l in open(R + "g3_table1/work/raw_gemma3_12b.jsonl") if '"E1_img_0240_mixed_N3_k1_P1", "mode": "mc"' in l)
def letters(r): return [l.split(":")[1].strip()[0] for l in r.splitlines() if ":" in l]
fig = plt.figure(figsize=(6.9, 2.75))
gs = fig.add_gridspec(2, 5, width_ratios=[1, 1.25, 1, 1, 1], height_ratios=[1, 1], wspace=0.08, hspace=0.35)
def show(ax, iid, title, edge):
    im = Image.open(IMG + iid + ".jpg"); w0, h0 = im.size; s = min(w0, h0)
    ax.imshow(im.crop(((w0 - s) // 2, (h0 - s) // 2, (w0 + s) // 2, (h0 + s) // 2))); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_visible(True); sp.set_color(edge); sp.set_linewidth(1.4)
    ax.set_title(title, fontsize=S.FS_SMALL, pad=2)
ax = fig.add_subplot(gs[0, 0]); show(ax, "img_0240", "Target: Tarte Tatin (France)", S.INK); S.panel(ax, "a", -0.08, 1.28)
ax = fig.add_subplot(gs[0, 1:]); ax.axis("off")
for k, lst in enumerate(("near", "far")):
    t = g1t[f"G1_img_0240_{lst}_N1_k1_P1"]
    ax.text(0.0, 0.95 - 0.42 * k, f"{lst.capitalize()} options ($N{{=}}1$, this photo only)", fontsize=S.FS_SMALL, weight="bold", va="top", transform=ax.transAxes)
    ax.text(0.0, 0.95 - 0.42 * k - 0.13, "Items: " + ", ".join(t["item_options"]) + "\nCountries: " + ", ".join(t["country_options"]),
            fontsize=S.FS_SMALL, va="top", transform=ax.transAxes, linespacing=1.3)
ax.text(0.0, 0.08, "Near: the four distractors come from the target's region (Europe); far: from other regions.", fontsize=S.FS_SMALL - 0.5, color=S.MUTED, transform=ax.transAxes)
for j, iid in enumerate(tr["order"]):
    a = fig.add_subplot(gs[1, j]); show(a, iid, f"Photo {j + 1}: {pool.loc[iid, 'item']}", S.INK if j == 0 else S.ACCENT)
    if j == 0: S.panel(a, "b", -0.08, 1.28)
ax = fig.add_subplot(gs[1, 3:]); ax.axis("off")
it, co = tr["item_options"], tr["country_options"]
ax.text(0.0, 1.0, "Asked about photo 1 ($N{=}3$, MC). Answers:", fontsize=S.FS_SMALL, va="top", transform=ax.transAxes)
for k, m in enumerate(["gemma4", "gemma3_12b", "qwen3vl", "qwen25vl", "gemma3_4b", "granite4v", "lfm25v"]):
    li, lc = letters(raw[m]); ai, ac = it["ABCDE".index(li)], co["ABCDE".index(lc)]
    ok = ai == "Tarte Tatin" and ac == "France"
    ax.text(0.0, 0.86 - 0.125 * k, f"{NAME[m]}: {ai} / {ac}" + ("  (correct)" if ok else ""), fontsize=S.FS_SMALL,
            color=S.INK if ok else S.ACCENT, va="top", transform=ax.transAxes)
credits = "; ".join(f"{pool.loc[i, 'item']}: {pool.loc[i, 'author']}, {pool.loc[i, 'license']}" for i in ["img_0240"] + tr["order"][1:])
fig.text(0.01, -0.01, "Photos (Wikimedia Commons): " + credits + ".", fontsize=S.FS_SMALL - 1, color=S.MUTED, va="top")
S.save(fig, "figS_worked_example_img0240", bbox_inches="tight"); plt.close(fig)
print("done")
