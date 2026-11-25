"""Generate jobs/queue_jobs.json (priority = list order). Rebuilt after the run folder was lost (see RUN_LOG). Re-run to append blocks; state is kept by job name."""
import json, datetime
from pathlib import Path
F = Path.home()/"CultureHaystack_handoff"/"fullpool"
M5 = ["lfm25v", "gemma3_4b", "qwen25vl", "qwen3vl", "gemma4"]          # smallest first
GB = dict(gemma3_4b=10, lfm25v=8, qwen25vl=18, qwen3vl=19, gemma4=26, gemma3_12b=26)
RM = "python ../wacv-lvr/run_models.py"
J = []
def add(name, cmd="", deps=(), gpu=0, f=None, rows=None, **kw):
    d = dict(name=name, cmd=cmd, deps=list(deps), gpu_gb=gpu)
    if f: d.update(expect_file=f, expect_rows=rows)
    d.update(kw); J.append(d)
# P0: reproduction check first (>=18/20, else the model is stopped and its G1 output marked invalid)
for m in ["lfm25v", "gemma4", "gemma3_4b", "qwen25vl", "qwen3vl"]:
    add(f"p0_{m}", f"bash jobs/p0.sh {m}; grep -q '^exit=0$' logs/p0_{m}.out || {{ echo 'P0 failed or incomplete for {m}; its G1 output is INVALID' > results/INVALID_g1_{m}.txt; echo \"- $(date '+%F %T') [P0] {m} FAILED the reproduction check; G1 output for {m} marked invalid\" >> RUN_LOG.md; exit 3; }}; grep REPRO logs/p0_{m}.out | sed 's/^/- P0 /' >> RUN_LOG.md", [], GB[m], perm_rc=3)
for m in ["qwen25vl", "qwen3vl", "gemma4", "gemma3_4b", "lfm25v"]:
    add(f"g1_{m}", f"bash jobs/g1.sh {m}", [f"p0_{m}"], GB[m], f"raw_{m}_g1.jsonl", 1720)
add("g3_det", "python ext/run_ext.py gemma3_12b --trials ../wacv-lvr/trials_main.jsonl --repeat --out logs/repro_gemma3_12b_a.jsonl && python ext/run_ext.py gemma3_12b --trials ../wacv-lvr/trials_main.jsonl --repeat --out logs/repro_gemma3_12b_b.jsonl && (python ext/check_det.py > logs/g3_det.out 2>&1; rc=$?; cat logs/g3_det.out >> RUN_LOG.md; [ $rc -eq 0 ] || exit 3)", [], GB["gemma3_12b"], perm_rc=3)
add("g3_main", "python ext/run_ext.py gemma3_12b --trials ../wacv-lvr/trials_main.jsonl --out raw_gemma3_12b_main.jsonl", ["g3_det"], GB["gemma3_12b"], "raw_gemma3_12b_main.jsonl", 6096)
add("g3_g1", "python ext/run_ext.py gemma3_12b --trials trials_g1.jsonl --modes mc --out raw_gemma3_12b_g1.jsonl", ["g3_det"], GB["gemma3_12b"], "raw_gemma3_12b_g1.jsonl", 1720)
# G2 on the 160 targets, 5 models (Granite excluded): 5 x 1,600 = 8,000 rows
for m in M5:
    add(f"g2_160_{m}", f"python ext/run_g2.py {m} --trials trials_g2_160.jsonl --out raw_{m}_g2_160.jsonl", [f"p0_{m}"], GB[m], f"raw_{m}_g2_160.jsonl", 1600)
G1ALL = [f"g1_{m}" for m in M5]
add("up_chunk01_g1_g3", "python ext/upload_chunk.py chunk01_g1_g3 " + " ".join(f"raw_{m}_g1.jsonl" for m in M5) + " raw_gemma3_12b_g1.jsonl raw_gemma3_12b_main.jsonl trials_g1.jsonl data " + " ".join(f"logs/repro_{m}.jsonl logs/p0_{m}.out" for m in M5) + " logs/repro_gemma3_12b_a.jsonl logs/repro_gemma3_12b_b.jsonl ext jobs RUN_LOG.md", G1ALL + ["g3_main", "g3_g1"], noenv=True, perm_rc=4)
add("up_chunk02_g2_160", "python ext/upload_chunk.py chunk02_g2_160 " + " ".join(f"raw_{m}_g2_160.jsonl" for m in M5) + " trials_g2_160.jsonl RUN_LOG.md", [f"g2_160_{m}" for m in M5], noenv=True, perm_rc=4)
add("clip_sim", "python ../wacv-lvr/clip_sim.py", [], 0, "clip_sim.csv", 2)
for m in M5:
    add(f"f1_mc_{m}", f"{RM} {m} --trials trials_full.jsonl --modes mc --out raw_{m}.jsonl", [f"g1_{m}"], GB[m], f"raw_{m}.jsonl", 16720)
    add(f"score_mc_{m}", f"python ../wacv-lvr/ch.py score {m} --trials trials_full.jsonl", [f"f1_mc_{m}"])
    add(f"up_f1mc_{m}", f"python ext/upload_chunk.py f1_mc_{m} raw_{m}.jsonl scored_{m}.csv RUN_LOG.md", [f"score_mc_{m}"], noenv=True, perm_rc=4)
for m in M5:
    add(f"f1_open_{m}", f"{RM} {m} --trials trials_full.jsonl --modes open --out raw_{m}.jsonl", [f"f1_mc_{k}" for k in M5], GB[m], f"raw_{m}.jsonl", 33440)
    add(f"score_{m}", f"python ../wacv-lvr/ch.py score {m} --trials trials_full.jsonl", [f"f1_open_{m}"])
    add(f"up_f1open_{m}", f"python ext/upload_chunk.py f1_open_{m} raw_{m}.jsonl scored_{m}.csv review_{m}.csv RUN_LOG.md", [f"score_{m}"], noenv=True, perm_rc=4)
for m in M5:
    add(f"g2_full_{m}", f"python ext/run_g2.py {m} --trials trials_g2_full.jsonl --out raw_{m}_g2_full.jsonl", [f"f1_open_{k}" for k in M5], GB[m], f"raw_{m}_g2_full.jsonl", 8840)
add("g3_f1", "python ext/run_ext.py gemma3_12b --trials trials_full.jsonl --modes mc --out raw_gemma3_12b.jsonl", ["g3_det"] + [f"g2_full_{m}" for m in M5], GB["gemma3_12b"], "raw_gemma3_12b.jsonl", 16720)
ALLG = G1ALL + ["g3_main", "g3_g1"] + [f"g2_160_{m}" for m in M5]
add("analyze_g1_g2_g3", "python ext/analyze_ext.py g1 g2 g3", ALLG)
add("up_chunk03_results_g1_g2_g3", "python ext/upload_chunk.py chunk03_results_g1_g2_g3 results RUN_LOG.md", ["analyze_g1_g2_g3"], noenv=True, perm_rc=4)
add("analyze_f1", "python ext/analyze_ext.py f1 && python ../wacv-lvr/analyze.py " + " ".join(M5), [f"score_{m}" for m in M5] + ["clip_sim"])
add("up_chunk04_f1_results", "python ext/upload_chunk.py chunk04_f1_results results RUN_LOG.md", ["analyze_f1"], noenv=True, perm_rc=4)
nb = datetime.datetime(2026, 11, 17, 18, 0).timestamp()
add("package_v1", "python ext/package_return.py v1", [], noenv=True, perm_rc=4, not_before=nb)
add("package_final", "python ext/package_return.py final", ["analyze_g1_g2_g3", "analyze_f1", "g3_f1"] + [f"g2_full_{m}" for m in M5], noenv=True, perm_rc=4)
(F/"jobs").mkdir(exist_ok=True)
json.dump(J, open(F/"jobs"/"queue_jobs.json", "w"), indent=1)
print(len(J), "jobs")
