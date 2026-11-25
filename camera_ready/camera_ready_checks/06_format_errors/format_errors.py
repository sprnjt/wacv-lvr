"""Task 6. Share of format-error replies (item_err or country_err == 'format_error', i.e. a field the frozen
parser could not read) per model x mode x N, from the frozen scored_<m>.csv. Recomputation of frozen data (descriptive).
Selections: core = analyze.core_sets (Table 1 trials, E1/E2 P1 mixed); main = all kind=='main' trials.
Granite MC joint acc at N=5 on all vs parsed-only replies (core, per-target mean as in Table 1, and pooled trials).
Run from cr_analyses/_frozen."""
import sys; sys.path.insert(0, ".")
import pandas as pd
import analyze as A
MODELS = ["gemma4", "qwen3vl", "qwen25vl", "gemma3_4b", "granite4v", "lfm25v"]
df = A.load(MODELS)
df["fmt"] = ((df.item_err == "format_error") | (df.country_err == "format_error")).astype(int)
core, _ = A.core_sets(df)
rows = []
for sel, d in (("core", core), ("main", df[df.kind == "main"])):
    t = d.groupby(["model", "mode", "N"]).agg(fmt_share=("fmt", "mean"), fmt_n=("fmt", "sum"), trials=("fmt", "size")).reset_index()
    rows.append(t.assign(selection=sel))
out = pd.concat(rows); out.to_csv("../06_format_errors/format_errors.csv", index=False)
w = out[out.selection == "core"].pivot_table(index=["mode", "model"], columns="N", values="fmt_share")
print("core selection, share of replies with a format error\n", w.round(3).to_string())
print("main selection totals:\n", out[out.selection == "main"].groupby(["model", "mode"])[["fmt_n", "trials"]].sum().to_string())
g = core[(core.model == "granite4v") & (core["mode"] == "mc") & (core.N == 5)]
p = g[g.fmt == 0]
print(f"\ngranite MC N=5 core: all {g.joint_ok.mean():.3f} (pooled) / {g.groupby('target').joint_ok.mean().mean():.3f} (per-target);"
      f" parsed-only {p.joint_ok.mean():.3f} (pooled) / {p.groupby('target').joint_ok.mean().mean():.3f} (per-target);"
      f" trials {len(g)} -> {len(p)}")
