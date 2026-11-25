import os, sys
sys.path.insert(0, "work/fp884/work/analysis"); os.chdir("work/fp884")
import novel as NV
df, pool = NV.load()[0], None
core, bal = NV.core_sets(df)
for name, fn in [("per_image_cost", lambda: NV.per_image_cost(core)), ("known", lambda: NV.conditional_on_known(core, bal)),
                 ("category", lambda: NV.category(core)), ("tier", lambda: NV.tier_slope(core)),
                 ("letters", lambda: NV.letters(core)), ("letter_copy", lambda: NV.letter_copy(core))]:
    try: fn(); print("ok", name, flush=True)
    except Exception as e: print("FAIL", name, type(e).__name__, str(e)[:200], flush=True)
