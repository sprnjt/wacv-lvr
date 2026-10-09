#!/usr/bin/env python3
"""Paper figures from results/*.csv -> results/figures/ (PDF + PNG, 300 dpi).

  python3 make_figures.py
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

OUT = Path("results/figures"); OUT.mkdir(parents=True, exist_ok=True)
MODELS = ["qwen3vl", "qwen25vl", "gemma4", "gemma3_4b", "granite4v", "lfm25v"]
NAME = {"qwen3vl": "Qwen3-VL-8B", "qwen25vl": "Qwen2.5-VL-7B", "gemma4": "Gemma 4 12B",
        "gemma3_4b": "Gemma 3 4B", "granite4v": "Granite 4.1 V 4B", "lfm25v": "LFM2.5-VL 3B"}
COLOR = {"qwen3vl": "#0072B2", "qwen25vl": "#56B4E9", "gemma4": "#E69F00",
         "gemma3_4b": "#F0E442", "granite4v": "#009E73", "lfm25v": "#CC79A7"}
MARK = {"qwen3vl": "o", "qwen25vl": "s", "gemma4": "D", "gemma3_4b": "v", "granite4v": "^", "lfm25v": "P"}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 300, "savefig.bbox": "tight"})


def save(fig, stem):
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{stem}.{ext}")
    plt.close(fig)
    print("wrote", OUT / f"{stem}.pdf")


def band(ax, d, m):
    ax.fill_between(d.N, d.lo, d.hi, color=COLOR[m], alpha=0.15, lw=0)


# --- fig 1: joint accuracy vs N, MC and open ---
acc = pd.read_csv("results/acc_by_N.csv")
acc = acc[acc.metric == "joint"]
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8), sharey=False)
for ax, mode, ttl in zip(axes, ("mc", "open"), ("multiple choice", "open answer")):
    for m in MODELS:
        d = acc[(acc.model == m) & (acc["mode"] == mode)].sort_values("N")
        ax.plot(d.N, d.acc, marker=MARK[m], ms=4, lw=1.4, color=COLOR[m], label=NAME[m])
        band(ax, d, m)
    ax.set(xlabel="N images", title=ttl, xticks=range(1, 6), ylim=(0, 1))
    ax.grid(alpha=0.25, lw=0.4)
axes[0].set_ylabel("joint accuracy")
axes[0].legend(fontsize=7, frameon=False, loc="lower left")
save(fig, "fig1_accuracy_vs_N")

# --- fig 2: error anatomy vs N (MC), all models ---
esh = pd.read_csv("results/error_shares_by_N.csv")
KINDS = [("mis_binding", "#D55E00", "mis-binding"), ("prior_drift", "#0072B2", "prior drift"),
         ("abstain", "#BBBBBB", "abstain"), ("format_error", "#000000", "format")]
fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.6), sharex=True, sharey=True)
for ax, m in zip(axes.flat, MODELS):
    d = esh[(esh.model == m) & (esh["mode"] == "mc") & (esh.N >= 2)].set_index("N").sort_index()
    bot = pd.Series(0.0, index=d.index)
    for k, c, lab in KINDS:
        ax.bar(d.index, d[k], bottom=bot, color=c, width=0.62, label=lab)
        bot += d[k]
    ax.set(title=NAME[m], xticks=d.index, ylim=(0, 1))
    ax.grid(alpha=0.25, lw=0.4, axis="y")
for ax in axes[-1]:
    ax.set_xlabel("N images")
for ax in axes[:, 0]:
    ax.set_ylabel("share of wrong fields")
handles, labels = axes.flat[0].get_legend_handles_labels()
fig.legend(handles, labels, fontsize=7, frameon=False, ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.02))
fig.subplots_adjust(bottom=0.15, hspace=0.35)
save(fig, "fig2_error_anatomy")

# --- fig 3: position profile at N=5 (MC) ---
pp = pd.read_csv("results/position_profile.csv")
fig, ax = plt.subplots(figsize=(3.6, 2.8))
for m in MODELS:
    d = pp[(pp.model == m) & (pp["mode"] == "mc")].sort_values("slot")
    ax.plot(d.slot, d.acc, marker=MARK[m], ms=4, lw=1.4, color=COLOR[m], label=NAME[m])
    ax.fill_between(d.slot, d.lo, d.hi, color=COLOR[m], alpha=0.15, lw=0)
ax.set(xlabel="slot of the asked image at N = 5", ylabel="joint accuracy", xticks=range(1, 6), ylim=(0, 1))
ax.grid(alpha=0.25, lw=0.4)
ax.legend(fontsize=6.5, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2,
          columnspacing=0.9, handlelength=1.4)
save(fig, "fig3_position_N5")

# --- fig 4: E6 describe-then-answer rescue (MC) ---
dc = pd.read_csv("results/describe_then_answer.csv")
dc = dc[dc.N != "all"].copy(); dc["N"] = dc.N.astype(int)
fig, ax = plt.subplots(figsize=(3.6, 2.8))
for m in MODELS:
    d = dc[(dc.model == m) & (dc["mode"] == "mc")].sort_values("N")
    ax.plot(d.N, d.rescue, marker=MARK[m], ms=4, lw=1.4, color=COLOR[m], label=NAME[m])
    ax.fill_between(d.N, d.lo, d.hi, color=COLOR[m], alpha=0.15, lw=0)
ax.axhline(0, color="k", lw=0.8, ls="--")
ax.set(xlabel="N images", ylabel="describe-first − direct (Δ acc)", xticks=range(1, 6))
ax.grid(alpha=0.25, lw=0.4)
ax.legend(fontsize=6.5, frameon=False, loc="lower left")
save(fig, "fig4_describe_rescue")

# --- fig 5: near vs far mis-binding at N=5 (MC) ---
nf = pd.read_csv("results/near_vs_far.csv")
nf = nf[(nf.N == 5) & (nf["mode"] == "mc") & (nf.metric == "mis_binding_rate")].set_index("model")
x = range(len(MODELS))
fig, ax = plt.subplots(figsize=(3.6, 2.8))
w = 0.36
ax.bar([i - w / 2 for i in x], [nf.loc[m, "near"] for m in MODELS], width=w, color="#D55E00", label="near-cultural list")
ax.bar([i + w / 2 for i in x], [nf.loc[m, "far"] for m in MODELS], width=w, color="#0072B2", label="far list")
ax.set_xticks(list(x), [NAME[m] for m in MODELS], rotation=25, ha="right")
ax.set(ylabel="mis-binding rate at N = 5", ylim=(0, 0.65))
ax.grid(alpha=0.25, lw=0.4, axis="y")
ax.legend(fontsize=7, frameon=False)
save(fig, "fig5_near_far")
