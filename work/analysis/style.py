"""Shared figure design system for every paper figure (main + supplementary).

  import sys; sys.path.insert(0, "paper/figures"); import style as S; S.apply()

Colour encodes meaning only:
  - model size class is the primary variable: >=7B = cool blues/teal, <=4B = warm red/orange/amber,
    one hue step per model (darkest = largest in each class). Line style repeats the class
    (solid >=7B, dashed <=4B) so plots survive greyscale.
  - ACCENT (magenta) is reserved for "the wrong image" / error, in every figure.
  - GREY is reserved for chance, zero lines and baselines.
Run this file to re-check colourblind safety (min CAM02-UCS dE under deutan/protan/tritan simulation)
and greyscale lightness spread.
"""
import matplotlib as mpl

MODELS = ["gemma4", "qwen3vl", "qwen25vl", "gemma3_4b", "granite4v", "lfm25v"]   # large -> small
SMALL = {"gemma3_4b", "granite4v", "lfm25v"}
NAME = {"gemma4": "Gemma-4-12B", "qwen3vl": "Qwen3-VL-8B", "qwen25vl": "Qwen2.5-VL-7B",
        "gemma3_4b": "Gemma-3-4B", "granite4v": "Granite-Vision-4B", "lfm25v": "LFM2.5-VL-3B"}
COL = {"gemma4": "#1d3f72", "qwen3vl": "#2f7fc1", "qwen25vl": "#5ab4c8",        # cool, >=7B
       "gemma3_4b": "#a33a1c", "granite4v": "#e06b26", "lfm25v": "#eab03f"}     # warm, <=4B
LARGE_C, SMALL_C = "#1d3f72", "#a33a1c"          # class-level colour (class means, class labels)
LS = {m: "--" if m in SMALL else "-" for m in MODELS}
MARK = dict(zip(MODELS, ["o", "s", "D", "o", "s", "D"]))
ACCENT = "#cc3399"                               # the wrong image / error
GREY = "#b0b0b0"                                 # chance, zero, baselines
INK, MUTED, GRID, TILE = "#222222", "#6b6b6b", "#ececec", "#e9e9e9"
FS, FS_SMALL = 7.5, 6.5                          # pt at final print size

RC = {
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "STIXGeneral"],
    "mathtext.fontset": "stix", "font.size": FS, "axes.labelsize": FS, "axes.titlesize": FS,
    "xtick.labelsize": FS_SMALL, "ytick.labelsize": FS_SMALL, "legend.fontsize": FS_SMALL,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "axes.linewidth": 0.6, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.major.pad": 1.5, "ytick.major.pad": 1.5,
    "axes.labelpad": 2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.5,
    "axes.axisbelow": True, "lines.linewidth": 1.2, "lines.markersize": 3,
    "errorbar.capsize": 0, "legend.frameon": False, "pdf.fonttype": 42, "ps.fonttype": 42,
    "savefig.dpi": 300, "figure.dpi": 150,
}


def apply():
    mpl.rcParams.update(RC)


def ci(ax, x, lo, hi, color, alpha=0.16):
    """Subtle-but-visible CI ribbon."""
    ax.fill_between(x, lo, hi, color=color, alpha=alpha, lw=0)


def errbar(ax, x, est, lo, hi, color, **kw):
    """Point + thin CI bar (est/lo/hi arrays)."""
    import numpy as np
    est, lo, hi = map(np.asarray, (est, lo, hi))
    kw = {"marker": "o", "ms": 3, "lw": 1.1, "elinewidth": 0.7, "capsize": 0, **kw}
    return ax.errorbar(x, est, yerr=[est - lo, hi - est], color=color, **kw)


def label(ax, x, y, text, color, dx=4, **kw):
    """Direct label to the right of a data point (offset in points), instead of a legend."""
    kw = {"fontsize": FS_SMALL, "va": "center", "ha": "left", **kw}
    return ax.annotate(text, (x, y), xytext=(dx, 0), textcoords="offset points", color=color,
                       annotation_clip=False, **kw)


def panel(ax_or_fig, letter, x=0.0, y=1.0):
    """Bold (a)/(b) tag at the top-left corner, in figure or axes coordinates."""
    kw = {"fontsize": FS + 0.5, "weight": "bold", "va": "top", "ha": "left", "color": INK}
    if isinstance(ax_or_fig, mpl.figure.Figure):
        return ax_or_fig.text(x, y, f"({letter})", **kw)
    return ax_or_fig.text(x, y, f"({letter})", transform=ax_or_fig.transAxes, **kw)


def save(fig, path, png=True, **kw):
    """Vector PDF with embedded TrueType fonts (+ PNG preview)."""
    from pathlib import Path
    path = Path(path)
    fig.savefig(path.with_suffix(".pdf"), **kw)
    if png:
        fig.savefig(path.with_suffix(".png"), **kw)


def _check():
    import itertools
    import numpy as np
    from colorspacious import cspace_convert, deltaE
    named = {**COL, "ACCENT": ACCENT, "GREY": GREY}
    rgb = {k: np.array([int(v[i:i + 2], 16) for i in (1, 3, 5)]) / 255 for k, v in named.items()}
    for cvd in (None, "deuteranomaly", "protanomaly", "tritanomaly"):
        sim = rgb if cvd is None else {k: np.clip(cspace_convert(v, {"name": "sRGB1+CVD", "cvd_type": cvd,
                                                                      "severity": 100}, "sRGB1"), 0, 1)
                                       for k, v in rgb.items()}
        d, a, b = min((deltaE(sim[a], sim[b], input_space="sRGB1"), a, b) for a, b in itertools.combinations(sim, 2))
        print(f"{cvd or 'normal':14s} min dE {d:5.1f} ({a} vs {b})")
        assert d >= 10, (cvd, a, b, d)            # ~10 dE = clearly distinguishable lines/markers
    L = {k: cspace_convert(v, "sRGB1", "CAM02-UCS")[0] for k, v in rgb.items()}
    print("greyscale J':", {k: round(v) for k, v in L.items()})
    # within each class, models step in lightness so greyscale keeps the order (line style separates classes)
    for cls in (MODELS[:3], MODELS[3:]):
        js = [L[m] for m in cls]
        assert all(b - a >= 9 for a, b in zip(js, js[1:])), (cls, js)


if __name__ == "__main__":
    _check()
    print("palette OK")
