"""Phase 03: build the full-pool targets, lists and trials exactly as docs/03 section 1. Reads wacv-lvr read-only, writes only fullpool/."""
import os, sys, csv, json, random, hashlib
from pathlib import Path
H = Path.home(); W = H/"CultureHaystack_handoff"/"wacv-lvr"; F = H/"CultureHaystack_handoff"/"fullpool"
sys.path.insert(0, str(W)); os.chdir(W)
import ch
pool, targets, lists = ch.load_data()
print("frozen", len(pool), len(targets), len(lists), flush=True)
pil = {x["image_id"] for x in targets if x["split"] == "pilot"}
main = [x for x in targets if x["split"] == "main"]
mid = {x["image_id"] for x in main}
L = dict(lists)
rows = [dict(x) for x in main]
for cat in ch.CATS:
    new = sorted(i for i in pool if pool[i]["category"] == cat and i not in pil and i not in mid)
    random.Random(f"{ch.SEED}-full-{cat}").shuffle(new)
    for j, i in enumerate(new):
        L[i] = {k: ch.pick_distractors(pool, i, k) for k in ("mixed", "near", "far")}
        r = pool[i]
        rows.append(dict(image_id=i, split="main", category=cat, tier=r["tier"], region=r["region"],
                         country=r["country"], near_ok=str(int(L[i]["near"] is not None)), idx=str(40 + j)))
T = ch.main_trials(pool, rows, L)
pilot_rows = [dict(x) for x in targets if x["split"] == "pilot"]
allrows = pilot_rows + rows
(F/"data").mkdir(exist_ok=True)
fields = list(targets[0].keys())
with open(F/"data"/"targets.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
    for r in allrows: w.writerow({k: r[k] for k in fields})
with open(F/"data"/"lists.json", "w") as f: json.dump(L, f)
ch.write_jsonl(str(F/"trials_full.jsonl"), T)
print("targets main", len(rows), "near_ok", sum(r["near_ok"] == "1" for r in rows), "lists", len(L), flush=True)
print("sha lists", hashlib.sha256(json.dumps(L).encode()).hexdigest(), flush=True)
print("trials", len(T), "sha", hashlib.sha256("\n".join(json.dumps(t) for t in T).encode()).hexdigest(), flush=True)
from collections import Counter
print("by exp", dict(Counter(t["exp"] for t in T)), flush=True)
fro = {}
for line in open(W/"trials_main.jsonl"):
    t = json.loads(line); fro[t["trial_id"]] = t
cur = {t["trial_id"]: t for t in T}
miss = [k for k in fro if k not in cur]; diff = [k for k in fro if k in cur and cur[k] != fro[k]]
print("frozen trials", len(fro), "missing", len(miss), "different", len(diff), flush=True)
print("new cats", dict(Counter(r["category"] for r in rows[len(main):])), flush=True)
