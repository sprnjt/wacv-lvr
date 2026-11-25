"""Phase 05-09 helper: every 30 min upload a resumable checkpoint (raw outputs, trial files, queue state, job graph, scripts, RUN_LOG) to a NEW prefix of the HF bucket. Never overwrites or deletes. Run: python ext/checkpoint_loop.py"""
import os, sys, time, glob, subprocess, datetime
os.chdir(os.path.expanduser("~/CultureHaystack_handoff/fullpool"))
PERIOD = 1800
while True:
    paths = sorted(glob.glob("raw_*.jsonl") + glob.glob("scored_*.csv") + glob.glob("trials_*.jsonl") + glob.glob("review_*.csv") + [f for f in glob.glob("data/*") if os.path.isfile(f) and not os.path.islink(f)]) + [p for p in ["ext", "jobs", "logs/queue_state.json", "logs/queue.log", "RUN_LOG.md", "results", "env.sh"] if os.path.exists(p)]
    paths = [os.path.abspath(p) if os.path.isdir(p) else p for p in paths]
    cmd = "unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE; . $HOME/.culturehaystack_secrets; python ext/upload_chunk.py ckpt " + " ".join(paths)
    r = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True)
    open("logs/checkpoint_loop.log", "a").write(f"{datetime.datetime.now():%F %T} rc={r.returncode} {r.stdout[-200:].strip()} {r.stderr[-200:].strip()}\n")
    time.sleep(PERIOD)
