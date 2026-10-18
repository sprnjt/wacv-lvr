#!/usr/bin/env python3
"""Smoke / reproduction checks used by run_all.sh (CPU only). Exit 1 = do not start the full job.

  python3 runpod/check.py causal FILE              rows written; self-control |delta| ~ 0
  python3 runpod/check.py mc FILE TRIALS           prints replies; fails if every reply is a format error
  python3 runpod/check.py attn FILE TRIALS         one finite, positive mass per shown image
  python3 runpod/check.py repro MODEL FILE         replies vs frozen raw_MODEL.jsonl (warns, never fails)
"""
import math, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import ch

kind = sys.argv[1]
path = sys.argv[3] if kind == "repro" else sys.argv[2]
rs = ch.read_jsonl(path) if Path(path).exists() else []
if not rs:
    sys.exit(f"FAIL {kind}: no rows in {path}")

if kind == "causal":
    pat = [r for r in rs if r.get("delta_logodds") is not None]
    print(f"{len(rs)} rows, {len(rs) - len(pat)} wrong-donor gate failures")
    for r in pat:
        print(f"  {r['case_id']} L{r['layer']} {r['intervention']}: base {r['baseline_logodds']:+.3f} delta {r['delta_logodds']:+.4f}")
    ctrl = [abs(r["delta_logodds"]) for r in pat if r["intervention"] == "self_to_target_control"]
    if not pat:
        print("WARN: every smoke case failed the gate; patch path not exercised (full run will still gate)")
    elif max(ctrl) > 0.05:
        sys.exit(f"FAIL causal: self-control |delta| up to {max(ctrl):.4f} (should be ~0)")
elif kind == "mc":
    trials = {t["trial_id"]: t for t in ch.read_jsonl(sys.argv[3])}
    bad = 0
    for r in rs:
        p = ch.parse(r["raw"], "mc", trials[r["trial_id"]])
        bad += p["item"] is None or p["country"] is None
        print(f"--- {r['trial_id']} parsed={p}\n{r['raw']}")
    if bad == len(rs):
        sys.exit("FAIL mc: every reply is a format error")
elif kind == "attn":
    trials = {t["trial_id"]: t for t in ch.read_jsonl(sys.argv[3])}
    for r in rs:
        a = r["attn"]
        print(r["trial_id"], r["mode"], [round(x, 3) for x in a])
        if len(a) != trials[r["trial_id"]]["N"] or not all(math.isfinite(x) and x >= 0 for x in a) or sum(a) <= 0:
            sys.exit(f"FAIL attn: bad row {r['trial_id']}")
elif kind == "repro":
    model = sys.argv[2]
    frozen = {(r["trial_id"], r["mode"]): r["raw"] for r in ch.read_jsonl(f"raw_{model}.jsonl")}
    same = sum(frozen.get((r["trial_id"], r["mode"])) == r["raw"] for r in rs)
    print(f"REPRO {model}: {same}/{len(rs)} replies identical to frozen raw_{model}.jsonl")
    if same < 0.9 * len(rs):
        print("WARNING: environment drift (library/image versions). Paired deltas vs frozen E2/E6 are "
              "confounded by it; report this number next to them.")
print(f"OK {kind}")
