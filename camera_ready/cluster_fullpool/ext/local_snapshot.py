"""Every 10 min copy the resumable state (raw outputs, scored files, queue state, run log, results) to ~/ch_backup/cur, keeping the previous generation in ~/ch_backup/prev. Lives on the persistent home volume, so a new pod can resume exactly. Run: python ext/local_snapshot.py"""
import os, glob, shutil, time
H = os.path.expanduser("~"); F = H + "/CultureHaystack_handoff/fullpool"; BK = H + "/ch_backup"
os.chdir(F)
def snapshot():
    tmp = BK + "/tmp"; shutil.rmtree(tmp, ignore_errors=True); os.makedirs(tmp)
    files = glob.glob("raw_*.jsonl") + glob.glob("scored_*.csv") + glob.glob("review_*.csv") + glob.glob("trials_*.jsonl") + ["logs/queue_state.json", "RUN_LOG.md", "clip_sim.csv", "judge_overrides.csv"] + glob.glob("results/*") + glob.glob("data/*.csv") + glob.glob("data/*.json")
    for f in files:
        if os.path.isfile(f) and not os.path.islink(f):
            d = tmp + "/" + f.replace("/", "__"); shutil.copy2(f, d)
    open(tmp + "/SNAPSHOT_TIME.txt", "w").write(time.strftime("%F %T") + "\n")
    shutil.rmtree(BK + "/prev", ignore_errors=True)
    if os.path.isdir(BK + "/cur"): os.rename(BK + "/cur", BK + "/prev")
    os.rename(tmp, BK + "/cur")
if __name__ == "__main__":
    while True:
        try: snapshot()
        except Exception as e: open("logs/local_snapshot.log", "a").write(time.strftime("%F %T") + " " + repr(e) + "\n")
        time.sleep(600)
