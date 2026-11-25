#!/bin/bash
# merge Kaggle shard outputs -> cluster-style raw files, then score and analyze with the original code
set -e
G=work/gscore; F=$G/home/CultureHaystack_handoff/fullpool; PY=python3
"$PY" -I - <<'PY'
import json,glob
F="work/gscore/home/CultureHaystack_handoff/fullpool"
rows=[json.loads(l) for p in sorted(glob.glob("work/kag/out*/raw_granite4v_shard*.jsonl")) for l in open(p) if l.strip()]
seen=set(); out={"core":[],"g1":[],"g2":[]}
for r in rows:
    k=(r["task"],r["trial_id"])
    if k in seen: continue
    seen.add(k); out[r["task"]].append(dict(trial_id=r["trial_id"],mode="mc",raw=r["raw"]))
for task,fn in (("core","raw_granite4v.jsonl"),("g1","raw_granite4v_g1.jsonl"),("g2","raw_granite4v_g2_160.jsonl")):
    open(f"{F}/{fn}","w").write("".join(json.dumps(x)+"\n" for x in out[task]))
print({k:len(v) for k,v in out.items()}, "expected core 7956 g1 1720 g2 1600")
PY
cd $F && PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$G/home/CultureHaystack_handoff/wacv-lvr" "$PY" -c "import ch; ch.score_file('granite4v','trials_full.jsonl'); print('scored')"
HOME=$G/home PYTHONDONTWRITEBYTECODE=1 "$PY" ext/analyze_ext_granite.py g1 g2
