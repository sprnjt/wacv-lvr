"""Phase 05: build trials_g1.jsonl (one-image near vs far options, N=1) from the full targets. Reads wacv-lvr read-only, writes fullpool/."""
import os, sys, csv, json, hashlib
from pathlib import Path
H = Path.home(); W = H/"CultureHaystack_handoff"/"wacv-lvr"; F = H/"CultureHaystack_handoff"/"fullpool"
sys.path.insert(0, str(W)); os.chdir(W)
import ch
pool, _, _ = ch.load_data()
L = json.load(open(F/"data"/"lists.json"))
rows = [r for r in csv.DictReader(open(F/"data"/"targets.csv")) if r["split"] == "main"]
T = []
for x in rows:
    if x["near_ok"] == "1":
        for lst in ("near", "far"):
            T.append(ch.mk(pool, L, "G1", x["image_id"], lst, 1, 1))
ch.write_jsonl(str(F/"trials_g1.jsonl"), T)
print("g1 trials", len(T), "sha", hashlib.sha256("\n".join(json.dumps(t) for t in T).encode()).hexdigest())
frozen = {r["image_id"] for r in ch.read_jsonl(str(W/"trials_main.jsonl")) if False} or {json.loads(l)["target"] for l in open(W/"trials_main.jsonl") if l.strip()}
print("160-target subset", sum(t["target"] in frozen for t in T))
print("expected sha 2b1d04cb46828e211e0429ca3a627fc4d06d91ecc7d3f7e302c115ad2cd75c14, subset 304")
