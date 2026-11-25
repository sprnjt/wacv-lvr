"""POST-REGISTRATION / EXPLORATORY. Table 1 row for Gemma-3-12B on the 160 frozen targets,
using the frozen analyze.py functions (load, core_sets, scaling, wide) unchanged.
Each model is run in its own process-fresh RNG state (seed 0), as gemma3_4b was in the frozen run."""
import sys, importlib
sys.path.insert(0, ".")
m = sys.argv[1]
import analyze as A
df = A.load([m]); core, bal = A.core_sets(df)
acc, summ = A.scaling(core)
acc.to_csv(f"../acc_by_N_{m}.csv", index=False); summ.to_csv(f"../scaling_summary_{m}.csv", index=False)
j = acc[acc.metric == "joint"]
for mode in ("mc", "open"):
    g = j[j["mode"] == mode].set_index("N"); s = summ[summ["mode"] == mode].iloc[0]
    print(f"[{m} {mode}] targets={int(g.targets.iloc[0])} Acc@1={g.acc[1]:.3f} [{g.lo[1]:.3f},{g.hi[1]:.3f}] "
          f"Acc@5={g.acc[5]:.3f} [{g.lo[5]:.3f},{g.hi[5]:.3f}] D1->5={g.acc[5]-g.acc[1]:+.3f} "
          f"drop1->2={s.initial_drop:+.3f} [{s.drop_lo:+.3f},{s.drop_hi:+.3f}] slope={s.slope_per_image:+.3f} [{s.slope_lo:+.3f},{s.slope_hi:+.3f}]")
    print("   joint by N:", " ".join(f"N{n}={g.acc[n]:.3f}" for n in range(1, 6)))
W = A.wide(core[core["mode"] == "mc"], "joint_ok")
ok1 = W[1] == 1
print(f"[{m} mc] targets fully correct at N=1: {ok1.sum()}; their mean joint acc at N=5: {W.loc[ok1, 5].mean():.3f}"
      f"  (N=1 trials/target: {core[(core['mode']=='mc')&(core.N==1)].groupby('target').size().value_counts().to_dict()})")
