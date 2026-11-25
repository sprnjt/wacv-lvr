import os, sys
sys.path.insert(0, "work/fp884/work/analysis"); os.chdir("work/fp884")
import novel as NV
df = NV.load()[0]; core, bal = NV.core_sets(df); NV.tier_slope(core); print("done")
