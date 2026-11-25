import os, sys
sys.path.insert(0, "work/fp884/work/analysis"); os.chdir("work/fp884")
import novel as NV
print(NV.region_pooled().round(4).to_string(index=False))
