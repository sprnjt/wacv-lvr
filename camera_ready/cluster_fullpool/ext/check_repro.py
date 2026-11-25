"""Phase 05: compare logs/repro_<model>.jsonl against the frozen raw_<model>.jsonl (cwd = fullpool/). Prints k/n identical; exit 1 if below 18/20."""
import sys, json
from pathlib import Path
m = sys.argv[1]
W = Path.home()/"CultureHaystack_handoff"/"wacv-lvr"
rd = lambda p: [json.loads(l) for l in open(p) if l.strip()]
frozen = {(r["trial_id"], r["mode"]): r["raw"] for r in rd(W/f"raw_{m}.jsonl")}
rs = rd(f"logs/repro_{m}.jsonl")
same = sum(frozen.get((r["trial_id"], r["mode"])) == r["raw"] for r in rs)
print(f"REPRO {m}: {same}/{len(rs)} identical to frozen")
for r in rs:
    if frozen.get((r["trial_id"], r["mode"])) != r["raw"]:
        print("DIFF", r["trial_id"], r["mode"], "| new:", r["raw"][:80].replace("\n"," "), "| frozen:", str(frozen.get((r["trial_id"], r["mode"])))[:80].replace("\n"," "))
sys.exit(0 if same >= 18 and len(rs) == 20 else 1)
