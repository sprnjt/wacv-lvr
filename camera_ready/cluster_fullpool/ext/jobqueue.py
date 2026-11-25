"""Dependency job queue for a node WITHOUT a scheduler (phases 04-09). Jobs are in jobs/queue_jobs.json (re-read every loop, so jobs can be appended).
A job starts when all its deps are done and a GPU slot + memory are free (strict priority = list order for GPU jobs). Jobs are resumable (run_models skips finished
calls), so a failed job is retried; known failure classes are fixed automatically and every fix is logged to RUN_LOG.md. Never edits prompts, seeds, decoding or wacv-lvr/."""
import json, os, re, subprocess, time, datetime, traceback
from pathlib import Path
F = Path.home()/"CultureHaystack_handoff"/"fullpool"; os.chdir(F)
(F/"logs"/"jobs").mkdir(parents=True, exist_ok=True)
STATE = F/"logs"/"queue_state.json"
TOTAL, MARGIN, MAX_GPU_JOBS, MAX_CPU_JOBS, MAX_ATT = 40.0, 1.5, 2, 2, 5

def ts(): return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
def log(m):
    with open(F/"logs"/"queue.log", "a") as f: f.write(f"[{ts()}] {m}\n")
def runlog(m):
    log(m)
    with open(F/"RUN_LOG.md", "a") as f: f.write(f"- {ts()} [queue] {m}\n")
def load_state():
    try: return json.load(open(STATE))
    except Exception: return {}
def rows(p):
    try:
        with open(p, "rb") as f: return sum(1 for _ in f)
    except FileNotFoundError: return 0
def gpu_used():
    try:
        o = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=30).stdout
        return float(o.strip().split()[0]) / 1024
    except Exception: return 0.0
def gpu_procs():
    o = subprocess.run("pgrep -f '^python .*(run_models|run_ext|run_g2).py' | wc -l", shell=True, capture_output=True, text=True).stdout.strip()
    return int(o or 0)
def alive(pid):
    try: os.kill(pid, 0); return True
    except Exception: return False
def repair(path):
    p = Path(path)
    if not p.exists(): return False
    data = p.read_bytes().split(b"\n"); keep = list(data)
    while keep and not keep[-1].strip(): keep.pop()
    if not keep: return False
    try: json.loads(keep[-1]); return False
    except Exception:
        keep.pop(); p.write_bytes(b"\n".join(keep) + (b"\n" if keep else b"")); return True
def tail(n):
    try: return (F/"logs"/"jobs"/f"{n}.log").read_text(errors="replace")[-4000:]
    except Exception: return ""

def finish(j, s, rc):
    n = j["name"]
    ok = rc == 0 and (not j.get("expect_file") or rows(j["expect_file"]) >= j.get("expect_rows", 1))
    if ok:
        s["status"] = "done"
        if j.get("expect_file") and s.get("started"):
            mins = max((time.time() - s["started"]) / 60, 1e-6); s["rate"] = round((rows(j["expect_file"]) - s.get("rows0", 0)) / mins, 1)
            runlog(f"{n} done, measured {s['rate']} calls/min this run ({rows(j['expect_file']) - s.get('rows0', 0)} calls in {mins:.0f} min)")
        else:
            runlog(f"{n} done")
        return
    if rc == j.get("perm_rc", -99):
        s["status"] = "failed"; runlog(f"{n} STOPPED by its own check (rc={rc}); dependents blocked, no code changes made"); return
    s["attempts"] = s.get("attempts", 0) + 1
    t = tail(n); fixes = []
    if re.search(r"out of memory", t, re.I):
        s["attempts"] -= 1; s["next"] = time.time() + 300; fixes.append("CUDA OOM (GPU shared with other users) -> retry in 5 min, not counted as a failed attempt")
    if j.get("expect_file") and repair(j["expect_file"]): fixes.append("dropped a truncated last line from " + j["expect_file"])
    if re.search(r"JSONDecodeError", t) and j.get("expect_file") and repair(j["expect_file"]): fixes.append("repaired jsonl")
    if rc == 0: fixes.append(f"exited 0 but only {rows(j.get('expect_file',''))}/{j.get('expect_rows')} rows -> resume")
    if s["attempts"] >= MAX_ATT:
        s["status"] = "failed"; runlog(f"{n} FAILED after {s['attempts']} attempts (rc={rc}); needs a look. Last log: " + t[-300:].replace(chr(10), " | "))
    else:
        s["status"] = "pending"; s["next"] = time.time() + 120
        runlog(f"{n} rc={rc}, attempt {s['attempts']}/{MAX_ATT}: " + ("; ".join(fixes) or "retry (resumable)"))

def launch(j, s):
    n = j["name"]
    pre = "unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE; . $HOME/.culturehaystack_secrets; " if j.get("noenv") else "source env.sh; "
    sh = f"cd {F}; {pre}( {j['cmd']} ); echo $? > logs/jobs/{n}.exit"
    p = subprocess.Popen(["bash", "-c", sh], stdout=open(F/"logs"/"jobs"/f"{n}.log", "a"), stderr=subprocess.STDOUT, start_new_session=True)
    s.update(status="running", pid=p.pid, started=time.time(), rows0=rows(j["expect_file"]) if j.get("expect_file") else 0); log(f"start {n} pid {p.pid}")

def step(S):
    jobs = json.load(open(F/"jobs"/"queue_jobs.json"))
    for j in jobs:
        n = j["name"]; s = S.setdefault(n, {"status": "pending", "attempts": 0, "next": 0})
        if s["status"] == "done": continue
        if j.get("external"):
            if rows(j["expect_file"]) >= j["expect_rows"]: s["status"] = "done"; runlog(f"{n} done (started outside the queue)")
            continue
        if s["status"] == "running":
            ex = F/"logs"/"jobs"/f"{n}.exit"
            if ex.exists():
                rc = int(ex.read_text().strip() or 1); ex.unlink(); finish(j, s, rc)
            elif not alive(s.get("pid", 0)):
                finish(j, s, 1)
    running = [j for j in jobs if S[j["name"]]["status"] == "running"]
    est = max(gpu_used(), sum(S[j["name"]].get("gpu_override", j.get("gpu_gb", 0)) for j in running))
    gp = gpu_procs(); cpu_run = sum(1 for j in running if not j.get("gpu_gb"))
    gpu_blocked = False
    for j in jobs:
        n = j["name"]; s = S[n]
        if s["status"] != "pending" or j.get("external") or time.time() < s["next"] or time.time() < j.get("not_before", 0): continue
        deps = j.get("deps", [])
        if any(S.get(d, {}).get("status") in ("failed", "blocked") for d in deps):
            s["status"] = "blocked"; runlog(f"{n} blocked (a dependency failed)"); continue
        if not all(S.get(d, {}).get("status") == "done" for d in deps): continue
        need = s.get("gpu_override", j.get("gpu_gb", 0))
        if need > 0:
            if gpu_blocked: continue
            if gp >= MAX_GPU_JOBS or est + need > TOTAL - MARGIN:
                continue  # backfill: a later job that fits may run while a bigger one waits for memory (other tenants share the GPU)
            launch(j, s); est += need; gp += 1
        else:
            if cpu_run >= MAX_CPU_JOBS: continue
            launch(j, s); cpu_run += 1

def status_file(S):
    jobs = json.load(open(F/"jobs"/"queue_jobs.json"))
    lines = [f"{ts()}  gpu_used={gpu_used():.1f}GB gpu_procs={gpu_procs()}"]
    for j in jobs:
        s = S.get(j["name"], {}); r = f"{rows(j['expect_file'])}/{j['expect_rows']}" if j.get("expect_file") else ""
        sp = ""
        if s.get("status") == "running" and j.get("expect_file") and s.get("started"):
            sp = f" {(rows(j['expect_file']) - s.get('rows0', 0)) / max((time.time() - s['started']) / 60, 1e-6):.1f}/min"
        elif s.get("rate") is not None:
            sp = f" {s['rate']}/min"
        lines.append(f"{j['name']:28s} {s.get('status','pending'):8s} att={s.get('attempts',0)} {r}{sp}")
    (F/"logs"/"queue_status.txt").write_text("\n".join(lines) + "\n")

def main():
    S = load_state(); log("queue started")
    while True:
        try:
            step(S); json.dump(S, open(STATE, "w"), indent=1); status_file(S)
        except Exception:
            log("loop error: " + traceback.format_exc()[-600:])
        time.sleep(30)

if __name__ == "__main__":
    main()
