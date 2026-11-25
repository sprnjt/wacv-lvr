"""Phase 06: build trials_g2_160.jsonl (1,600) and trials_g2_full.jsonl (8,840) from the E2 trials, and log three rendered prompts to RUN_LOG.md."""
import os, sys, json
from pathlib import Path
H = Path.home(); W = H/"CultureHaystack_handoff"/"wacv-lvr"; F = H/"CultureHaystack_handoff"/"fullpool"
sys.path.insert(0, str(W)); sys.path.insert(0, str(F/"ext")); os.chdir(F)
import ch, ch_g2
rd = lambda p: [json.loads(l) for l in open(p) if l.strip()]
def conv(src):
    out = []
    for t in src:
        for ask in ("item", "country"):
            d = dict(t); d["ask"] = ask; d["exp"] = "G2"; d["trial_id"] = "G2_" + t["trial_id"] + "_Q" + ask
            out.append(d)
    return out
frozen = [t for t in rd(W/"trials_main.jsonl") if t["exp"] == "E2"]
full = [t for t in rd(F/"trials_full.jsonl") if t["exp"] == "E2"]
g160, gfull = conv(frozen), conv(full)
ch.write_jsonl(str(F/"trials_g2_160.jsonl"), g160); ch.write_jsonl(str(F/"trials_g2_full.jsonl"), gfull)
print("E2 frozen", len(frozen), "E2 full", len(full), "g2_160", len(g160), "g2_full", len(gfull), flush=True)
near = next(t for t in rd(F/"trials_full.jsonl") if t["exp"] == "E3" and t["list"] == "near" and t["N"] == 5)
samples = [("item", g160[0]), ("country", g160[1]), ("item, near list (E3 source)", conv([near])[0])]
with open(F/"RUN_LOG.md", "a") as f:
    f.write("- phase 06: G2 built (1600 / 8840 trials). Rendered prompts follow (images shown as <id>).\n")
    for label, t in samples:
        txt = "".join(v if k == "text" else f" <{v}>" for k, v in ch_g2.make_parts_g2(t, "mc"))
        f.write(f"  - G2 sample ({label}), trial {t['trial_id']}:\n" + "\n".join("    | " + ln for ln in txt.split("\n")) + "\n")
print("logged 3 rendered prompts")
