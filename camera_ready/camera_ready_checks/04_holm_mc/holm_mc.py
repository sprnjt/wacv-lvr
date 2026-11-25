"""Task 4. (a) Holm over the 24 MC cells of the frozen results/illusory_conjunctions.csv (open cells dropped);
p_perm values are the frozen ones, only the Holm family changes (frozen analyze.holm). Recomputation of frozen numbers.
(b) Collage (E9) ICR cells, from frozen work/gpu_out/results/collage_<m>.csv (what gpu_results.py reads):
cells above baseline (obs > baseline), nominal p<0.05, and Holm over all 18 C2-C4 cells. Run from cr_analyses/_frozen."""
import sys; sys.path.insert(0, ".")
import pandas as pd, numpy as np
import analyze as A
F = "../../wacv-lvr/"
d = pd.read_csv(F + "results/illusory_conjunctions.csv")
mc = d[d["mode"] == "mc"].copy()
mc["p_holm_mc_only"] = A.holm(mc.p_perm)
mc["above_baseline"] = mc.excess > 0
mc.to_csv("../04_holm_mc/icr_holm_mc_only.csv", index=False)
print(f"MC cells: {len(mc)}; above baseline: {mc.above_baseline.sum()}; nominal p<0.05: {(mc.p_perm < .05).sum()}; "
      f"Holm(MC-only)<0.05: {(mc.p_holm_mc_only < .05).sum()}; min p_perm {mc.p_perm.min():.4f}; "
      f"min Holm(MC) {mc.p_holm_mc_only.min():.3f}; min frozen Holm(all 48) {mc.p_holm.min():.3f}")
print(mc[["model", "N", "wrong_joint", "icr_observed", "icr_independence_baseline", "excess", "p_perm", "p_holm", "p_holm_mc_only"]].round(3).to_string(index=False))
rows = []
for m in ["gemma4", "qwen3vl", "qwen25vl", "gemma3_4b", "granite4v", "lfm25v"]:
    c = pd.read_csv(f"{F}work/gpu_out/results/collage_{m}.csv"); c = c[c.table == "icr"]
    for cond, g in c.groupby("condition"):
        v = g.set_index("metric").value
        rows.append(dict(model=m, condition=cond, wrong=int(g.n.iloc[0]), icr=v["icr"], baseline=v["icr_baseline"], p_perm=v["icr_p_perm"]))
e9 = pd.DataFrame(rows); e9["p_holm"] = A.holm(e9.p_perm); e9["above_baseline"] = e9.icr > e9.baseline
e9.to_csv("../04_holm_mc/collage_icr_holm.csv", index=False)
print(f"\ncollage cells: {len(e9)}; above baseline: {e9.above_baseline.sum()}; nominal p<0.05: {(e9.p_perm < .05).sum()}; "
      f"Holm<0.05: {(e9.p_holm < .05).sum()}; min p {e9.p_perm.min():.3f}; min Holm {e9.p_holm.min():.3f}")
print(e9.round(3).to_string(index=False))
