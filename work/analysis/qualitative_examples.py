#!/usr/bin/env python3
"""Qualitative figure: three real failures (paper Fig. figExamples) + supplementary photo credits.

  .venv/bin/python work/analysis/qualitative_examples.py            # -> $PAPER_OUT/figures/figExamples.pdf
                                                                      #    $PAPER_OUT/sec/gen_photo_credits.tex
PAPER_OUT=<paper dir> writes into the LaTeX paper (default work/analysis/out/paper). Original photos are read from $WCI (default /tmp/wci; pool.csv `file`).
The three trials are fixed below; every selection criterion is re-checked from the frozen scores (assert), so a
re-run either reproduces the figure or fails loudly.
  (a) half right   LFM2.5-VL-3B, E2 MC N=5: item right, country of another shown photo; LFM right on this target at
                   N=1; Gemma-4-12B right on the same trial (second answer line).
  (b) whole swap   LFM2.5-VL-3B, E2 MC N=5: both fields name the same other shown photo; LFM right at N=1.
  (c) collage      Qwen3-VL-8B, E9: right alone (C1) and as four separate images (C2), wrong in the 2x2 collage asked
                   by position (C3), the country taken from another panel. Collage rebuilt as make_collage.py does.
Only photos licensed public domain / CC0 / CC BY* are shown.
"""
import io
import json
import os
import sys
from pathlib import Path

import pandas as pd
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import ch                                        # noqa: E402
import make_collage as MC                        # noqa: E402
import style as S                                # noqa: E402
import matplotlib                                # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                  # noqa: E402
from matplotlib.patches import Rectangle         # noqa: E402

OUT = Path(os.environ.get("PAPER_OUT", Path(__file__).resolve().parents[2] / "work/analysis/out/paper")).expanduser()
WCI = Path(os.environ.get("WCI", "/tmp/wci"))
HALF, WHOLE, COLLAGE = "E2_img_0260_mixed_N5_k3_P1", "E2_img_0805_mixed_N5_k3_P1", "img_0618"
SMALL_M, BIG_M, COLL_M = "lfm25v", "gemma4", "qwen3vl"
GREEN = "#2e7d32"
THUMB_PX = 200

pool = pd.read_csv("data/pool.csv").set_index("image_id")
P = ch.load_pool()
TM = {t["trial_id"]: t for t in ch.read_jsonl("trials_main.jsonl")}
TC = {t["trial_id"]: t for t in ch.read_jsonl("trials_collage.jsonl")}


def licensed(i):
    lic = str(pool.loc[i, "license"])
    return lic in ("Public domain", "CC0") or lic.startswith("CC BY")


def scored(m):
    d = pd.read_csv(f"scored_{m}.csv")
    return d[d["mode"] == "mc"].set_index("trial_id")


def n1_ok(d, target):
    return bool(d[(d.target == target) & (d.N == 1) & (d.exp == "E1")].joint_ok.max())


def collage_scores(m, target):
    out = {}
    for r in ch.read_jsonl(f"work/gpu_out/raw_{m}_collage.jsonl"):
        tid = r["trial_id"]
        if r.get("mode") == "mc" and tid.startswith(f"E9_{target}_") and tid not in out:   # first row wins, as gpu_results
            out[tid] = ch.score_trial(TC[tid], "mc", r["raw"], P)
    return {c: out[f"E9_{target}_{c}"] for c in MC.CONDS}


# ---------------------------------------------------------------- select + verify
L, G = scored(SMALL_M), scored(BIG_M)
a, b = L.loc[HALF], L.loc[WHOLE]
assert a.exp == "E2" and a.N == 5 and a.item_ok and not a.country_ok and a.country_slot > 0 and a.country_slot != a.slot
assert n1_ok(L, a.target) and G.loc[HALF].joint_ok == 1
assert b.exp == "E2" and b.N == 5 and not b.item_ok and not b.country_ok and b.item_slot == b.country_slot > 0
assert n1_ok(L, b.target)
cs = collage_scores(COLL_M, COLLAGE)
c = cs["collage_pos"]
assert cs["solo"]["joint_ok"] and cs["separate"]["joint_ok"] and not c["joint_ok"]
assert c["item_ok"] and c["country_slot"] > 0                     # country borrowed from another panel
tc = TC[f"E9_{COLLAGE}_collage_pos"]
shown = TM[HALF]["order"] + TM[WHOLE]["order"] + tc["panels"]
assert all(licensed(i) for i in shown), [i for i in shown if not licensed(i)]
print("chosen trials:", HALF, f"({SMALL_M}; {BIG_M} correct)", "|", WHOLE, f"({SMALL_M})", "|",
      f"E9_{COLLAGE}_{{solo,separate,collage_pos}} ({COLL_M})")


# ---------------------------------------------------------------- images
def photo(i):
    return Image.open(WCI / pool.loc[i, "file"])


def thumb(i):
    return ImageOps.fit(photo(i).convert("RGB"), (THUMB_PX, THUMB_PX), method=Image.Resampling.LANCZOS)


def prepared(i):
    """ch.prep: flatten alpha on white, RGB, thumbnail 768, JPEG q90 (in memory)."""
    im = photo(i)
    if im.mode in ("RGBA", "LA", "P"):
        im = im.convert("RGBA")
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
        bg.alpha_composite(im)
        im = bg
    im = im.convert("RGB")
    im.thumbnail((768, 768))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=90)
    return Image.open(io.BytesIO(buf.getvalue()))


def collage(panels, only=None):
    cells = [ImageOps.pad(prepared(i), (MC.CELL, MC.CELL), method=Image.Resampling.LANCZOS, color="white")
             for i in panels]
    if only is not None:
        cells = [x if j == only else None for j, x in enumerate(cells)]
    return MC.canvas(cells).resize((300, 300), Image.Resampling.LANCZOS)


# ---------------------------------------------------------------- drawing (inches)
S.apply()
W, H = 6.875, 1.83
TH, GAP, CW = 0.47, 0.045, 0.64          # thumbnail, gap, collage side
X_TXT, X_VAL = 2.70, 3.80                 # text columns: labels, values
LINE = 0.105
fig = plt.figure(figsize=(W, H))
R = fig.canvas.get_renderer()


def ax_img(x, y, s, im):
    ax = fig.add_axes([x / W, y / H, s / W, s / H])
    ax.imshow(im, interpolation="none")
    ax.set_axis_off()
    return ax


def frame(x, y, s, color, lw=1.6):
    fig.add_artist(Rectangle((x / W, y / H), s / W, s / H, transform=fig.transFigure, fill=False,
                             ec=color, lw=lw, zorder=10, clip_on=False))


def badge(x, y, s):
    fig.text((x + 0.025) / W, (y + TH - 0.025) / H, s, fontsize=S.FS_SMALL, ha="left", va="top", color=S.INK,
             zorder=11, bbox=dict(boxstyle="square,pad=0.12", fc="white", ec="none", alpha=0.85))


def rich(x, y, parts):
    """Draw [(text, color, weight)] left to right on one baseline; returns the end x (inches)."""
    for s, col, wt in parts:
        t = fig.text(x / W, y / H, s, fontsize=S.FS_SMALL, color=col, weight=wt, ha="left", va="baseline")
        x += t.get_window_extent(R).width / fig.dpi
    return x


def answer(ans, wrong_note):
    """Coloured 'item · country' with the wrong field(s) marked."""
    parts = []
    for k, f in enumerate(("item", "country")):
        ok = ans[f"{f}_ok"]
        parts.append((ans[f"{f}_ans"], GREEN if ok else S.ACCENT, "normal" if ok else "bold"))
        if k == 0:
            parts.append(("  ·  ", S.MUTED, "normal"))
    ok = ans["item_ok"] and ans["country_ok"]
    parts.append(("   correct" if ok else f"   wrong: {wrong_note}", GREEN if ok else S.ACCENT, "normal"))
    return parts


def textblock(top, tag, asked, gold, rows):
    y = top - 0.085
    fig.text(X_TXT / W, y / H, tag, fontsize=S.FS, weight="bold", color=S.INK, ha="left", va="baseline")
    lines = [("Question", [(asked, S.INK, "normal")]), ("Gold answer", [(gold, S.INK, "normal")])] + rows
    for lab, parts in lines:
        y -= LINE
        lab_t = lab if isinstance(lab, tuple) else (lab, S.MUTED)
        fig.text(X_TXT / W, y / H, lab_t[0], fontsize=S.FS_SMALL, color=lab_t[1], ha="left", va="baseline")
        rich(X_VAL, y, parts)


def strip(top, order, asked_slot, wrong_slot):
    y = top - TH
    for k, i in enumerate(order):
        x = k * (TH + GAP)
        ax_img(x, y, TH, thumb(i))
        badge(x, y, str(k + 1))
        if k + 1 == asked_slot:
            frame(x, y, TH, S.INK)
        elif k + 1 == wrong_slot:
            frame(x, y, TH, S.ACCENT)


def gold(i):
    return f"{pool.loc[i, 'item']}  ·  {pool.loc[i, 'country']}"


def model(m, extra=""):
    return (S.NAME[m] + extra, S.INK)


# row (a): half right
top = H - 0.02
ta = TM[HALF]
strip(top, ta["order"], a.slot, a.country_slot)
textblock(top, f"(a) Half right: country from Image {a.country_slot}", f"Image {a.slot}: item? country?",
          gold(a.target),
          [(model(SMALL_M), answer(a, f"country of Image {a.country_slot}")),
           (model(BIG_M), answer(G.loc[HALF], ""))])

# row (b): whole-answer swap
top -= TH + 0.11
tb = TM[WHOLE]
strip(top, tb["order"], b.slot, b.item_slot)
textblock(top, f"(b) Whole answer of Image {b.item_slot}", f"Image {b.slot}: item? country?", gold(b.target),
          [(model(SMALL_M), answer(b, f"both fields of Image {b.item_slot}"))])

# row (c): collage
top -= TH + 0.11
s, src = tc["slot"], c["country_slot"]
y = top - CW
for x, only in ((0, None), (CW + 0.74, s - 1)):
    ax_img(x, y, CW, collage(tc["panels"], only=only))
    frame(x, y, CW, S.GREY, lw=0.5)
for x, lab in ((0, "one 2×2\ncollage\n(asked)"), (CW + 0.74, "target alone\nin the same\ncanvas")):
    fig.text((x + CW + 0.05) / W, y / H, lab, fontsize=S.FS_SMALL, color=S.MUTED, ha="left", va="bottom",
             linespacing=1.1)
half = CW * MC.CELL / MC.SIZE                                     # one panel, as a share of the canvas
mar = CW * MC.MARGIN / MC.SIZE
for j, col in ((s - 1, S.INK), (src - 1, S.ACCENT)):
    r_, c_ = divmod(j, 2)
    px = mar + c_ * CW * (MC.CELL + MC.GUT) / MC.SIZE
    py = y + CW - mar - (r_ + 1) * half - r_ * CW * MC.GUT / MC.SIZE
    frame(px, py, half, col, lw=1.3)
pos = MC.POS
textblock(top, f"(c) Collage: country from {pos[src - 1]} panel", f"{pos[s - 1].capitalize()} panel: item? country?",
          gold(tc["target"]),
          [(model(COLL_M, ", alone"), answer(cs["solo"], "")),
           (model(COLL_M, ", 4 images"), answer(cs["separate"], "")),
           (model(COLL_M, ", collage"), answer(c, f"country of {pos[src - 1]} panel"))])

S.save(fig, OUT / "figures/figExamples.pdf", png=True)

# ---------------------------------------------------------------- photo credits (supplementary)
def tex(s):
    s = str(s).replace("\\", "/")
    for ch_, rep in (("&", r"\&"), ("%", r"\%"), ("#", r"\#"), ("_", r"\_"), ("$", r"\$"), ("{", ""), ("}", "")):
        s = s.replace(ch_, rep)
    return s


def author(i):
    a = str(pool.loc[i, "author"])
    a = a.replace("Fotograf: ", "").split(" Kamera:")[0]
    return a if a and a != "nan" else "unknown author"


rows = [f"\\texttt{{{i.replace('_', chr(92) + '_')}}}: {tex(author(i))} ({tex(pool.loc[i, 'license'])})"
        for i in dict.fromkeys(shown)]
assert all(ord(ch_) < 0x250 for r in rows for ch_ in r), "non-Latin author name: transliterate before use"
(OUT / "sec/gen_photo_credits.tex").write_text(
    "% generated by work/analysis/qualitative_examples.py\n" + ";\n".join(rows) + ".\n", encoding="utf-8")
print(f"wrote {OUT / 'figures/figExamples.pdf'} and sec/gen_photo_credits.tex ({len(rows)} photos)")
