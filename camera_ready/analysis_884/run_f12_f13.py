import os, sys
sys.path.insert(0, "work/fp884/work/analysis"); os.chdir("work/fp884")
import novel as NV
df, core, bal = None, None, None
r = NV.load(); df = r[0]
core, bal = NV.core_sets(df)
NV.rtm_control(df, core); NV.letter_copy(core); NV.letters(core)
print("done")
