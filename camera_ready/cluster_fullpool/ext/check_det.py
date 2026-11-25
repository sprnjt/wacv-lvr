"""G3 determinism check: two runs of the same 20 calls must give identical replies (no frozen file exists for gemma3_12b). Exit 0 if 20/20 equal."""
import sys, json
rd = lambda p: {(r["trial_id"], r["mode"]): r["raw"] for r in (json.loads(l) for l in open(p) if l.strip())}
a, b = rd("logs/repro_gemma3_12b_a.jsonl"), rd("logs/repro_gemma3_12b_b.jsonl")
same = sum(a.get(k) == v for k, v in b.items())
print(f"DETERMINISM gemma3_12b: {same}/{len(b)} identical between two runs")
sys.exit(0 if same == len(b) == 20 else 1)
