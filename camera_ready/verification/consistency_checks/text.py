import re, glob, math
from decimal import Decimal, ROUND_HALF_UP, ROUND_HALF_EVEN
import pandas as pd
vals = set()
for f in glob.glob("work/cc/hf/results/**/*.csv", recursive=True) + glob.glob("work/cc/hf/figures/*.csv") + glob.glob("work/cc/hf/models/*/probes/results/*.csv"):
    df = pd.read_csv(f)
    for c in df.select_dtypes("number").columns:
        for v in df[c].dropna():
            if math.isfinite(v): vals.update({float(v), float(v) * 100, -float(v)})
def rnd(v, d, m): return Decimal(repr(v)).quantize(Decimal(1).scaleb(-d), rounding=m)
for name in ("main.tex", "supp.tex"):
    s = open("work/cc/" + name).read()
    s = re.sub(r"\\begin\{(table\*?|tabular)\}.*?\\end\{\1\}", " ", s, flags=re.S)        # tables done separately
    s = re.sub(r"(?m)%.*$", "", s)
    t = s.replace("$-$", "-").replace("−", "-")
    t = re.sub(r"\\(cite|cref|Cref|ref|label|gpu|gpuif|ssec|stab|sfig|mtab|mfig|msec|href|url|includegraphics)\*?\{[^}]*\}", " ", t)
    t = re.sub(r"[A-Za-z][A-Za-z0-9._-]*\d[A-Za-z0-9._-]*", " ", t)
    found = [(m.group(0), t[max(0, m.start()-50):m.end()+20].replace("\n", " ")) for m in re.finditer(r"(?<![\w.])-?\d+\.\d{2,3}(?![\d])", t)]
    bad = []
    for x, ctx in found:
        d = len(x.split(".")[1]); X = Decimal(x)
        if any(rnd(v, d, ROUND_HALF_UP) == X or rnd(v, d, ROUND_HALF_EVEN) == X for v in vals): continue
        dbl = [v for v in vals if rnd(float(rnd(v, d + 1, ROUND_HALF_UP)), d, ROUND_HALF_UP) == X]
        bad.append((x, "double-rounded from %s" % round(dbl[0], d + 3) if dbl else "no source", ctx))
    print(f"== {name}: {len(found)} typed decimals outside tables; {len(bad)} not matched")
    for b in bad: print("   ", b[0], "|", b[1], "|", b[2])
