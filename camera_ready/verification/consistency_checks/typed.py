import re, glob, math
from decimal import Decimal, ROUND_HALF_UP, ROUND_HALF_EVEN
import pandas as pd
H = "work/cc/hf/results/"
s = open("work/cc/supp.tex").read()
SRC = {
 "tab:fmt": ["camera_ready/06_format_errors/*.csv"],
 "tab:per-image-cost": ["main_study/exploratory/f1_*.csv", "main_study/exploratory/f15_*.csv"],
 "tab:confounds": ["main_study/exploratory/*.csv"],
 "tab:slottier": ["camera_ready/05_slot_tier/*.csv"],
 "tab:s8full": ["camera_ready/01_fullpool_ci/*.csv", "full_pool/*.csv"],
 "tab:s8hyp": ["full_pool/h*.csv", "full_pool/scaling_summary.csv", "camera_ready/04_holm_mc/*.csv"],
 "tab:s8open": ["full_pool/*.csv", "camera_ready/01_fullpool_ci/*.csv"],
 "tab:clipshown": ["camera_ready/03_clip_shown/*.csv", "main_study/h3_near_regression.csv"],
 "tab:s8g1": ["full_pool/g1_near_far.csv"],
 "tab:s8g2": ["camera_ready/02_g2_same_image/*.csv", "full_pool/g2_split.csv"],
}
def table(lab):
    i = s.index(r"\label{" + lab + "}"); a = s.rfind(r"\begin{tabular}", 0, i); e = s.index(r"\end{tabular}", a)
    return s[a:e]
def nums(t):
    t = t.replace("$-$", "-").replace("−", "-"); t = re.sub(r"\\(textbf|mathbf|underline)\{([^}]*)\}", r"\2", t)
    t = re.sub(r"\\[A-Za-z]+", " ", t); t = t.replace("$", " ").replace("{", " ").replace("}", " ")
    t = re.sub(r"[A-Za-z][A-Za-z0-9.-]*\d[A-Za-z0-9.-]*", " ", t)   # model names with digits
    return [x for x in re.findall(r"(?<![\w.])-?\d+(?:\.\d+)?(?![\w])", t)]
def rnd(v, d, mode):
    q = Decimal(1).scaleb(-d); return Decimal(repr(v)).quantize(q, rounding=mode)
for lab, globs in SRC.items():
    vals = set()
    for g in globs:
        for f in glob.glob(H + g):
            df = pd.read_csv(f)
            for c in df.select_dtypes("number").columns:
                for v in df[c].dropna():
                    vals.add(float(v)); vals.add(float(v) * 100)
    vals = [v for v in vals if math.isfinite(v)]
    typed = nums(table(lab)); unmatched = []; double = []
    for x in typed:
        d = len(x.split(".")[1]) if "." in x else 0
        X = Decimal(x)
        if d == 0 and abs(float(x)) in (1, 2, 3, 4, 5, 95): continue       # N values, 95% CI headers
        hit = any(rnd(v, d, ROUND_HALF_UP) == X or rnd(v, d, ROUND_HALF_EVEN) == X or rnd(-v, d, ROUND_HALF_UP) == X for v in vals)
        if not hit:
            dbl = [v for v in vals if rnd(float(rnd(v, d + 1, ROUND_HALF_UP)), d, ROUND_HALF_UP) == X]
            (double if dbl else unmatched).append((x, [round(v, d + 3) for v in dbl[:2]]))
    print(f"{lab:18s} {len(typed):3d} numbers | unmatched: {len(unmatched)} {[u[0] for u in unmatched][:12]} | double-rounded: {double[:6]}")
