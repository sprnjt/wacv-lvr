"""Phase 09: frozen check, report, scrub, secret scan, archive + manifest, upload to a NEW prefix culturehaystack/fullpool_return_<date>_<tag>/ of the temporary bucket.
Usage: python ext/package_return.py <tag>   (exit 5 = secret hit, 4 = auth failure)"""
# Progress checkpoints and the final return were copied to a temporary Hugging Face bucket that existed only
# during the run (now deleted). All final data is in the Hugging Face dataset repository
# shinjinii/culturehaystack-results. The bucket address is read from the CH_TEMP_BUCKET environment variable.
import os, re, sys, json, shutil, tarfile, hashlib, getpass, socket, subprocess, datetime
from pathlib import Path
H = Path.home(); F = H/"CultureHaystack_handoff"/"fullpool"; os.chdir(F)
tag = sys.argv[1]; date = datetime.datetime.utcnow().strftime("%Y%m%d")
r = subprocess.run("cd ~/CultureHaystack_handoff && sha256sum -c FROZEN_SHA256.txt", shell=True, capture_output=True, text=True)
(F/"frozen_check.txt").write_text(r.stdout + r.stderr)
bad = [l for l in r.stdout.splitlines() if not l.endswith(": OK")]
print("frozen check: ok" if not bad and r.returncode == 0 else f"FROZEN CHECK PROBLEM: {bad[:3]}", flush=True)
subprocess.run([str(H/"envs"/"ch"/"bin"/"python"), "ext/make_report.py"])
stage = H/"upload_stage"/f"return_{date}_{tag}"; shutil.rmtree(stage, ignore_errors=True); body = stage/"fullpool"; body.mkdir(parents=True)
TEXT = {".jsonl", ".csv", ".md", ".out", ".log", ".txt", ".json", ".py", ".sh", ".yaml", ".exit"}
subs = [(str(H), "<PROJ>"), (re.compile(r"proxy-[A-Za-z0-9.-]+"), "<proxy>"), (re.compile(r"\bscn\d+(-\w+)?\b"), "<host>"), (getpass.getuser(), "<user>"), (socket.gethostname(), "<host>"), (re.compile(r"\bisea\d*\b"), "<user>")]
def put(src, rel):
    if "__pycache__" in rel or rel.endswith(".pyc") or rel.startswith("images"): return
    dst = body/rel; dst.parent.mkdir(parents=True, exist_ok=True)
    if src.suffix in TEXT:
        t = src.read_text(errors="replace")
        for a, b in subs: t = t.replace(a, b) if isinstance(a, str) else a.sub(b, t)
        dst.write_text(t)
    else: shutil.copy2(src, dst)
items = ["data/targets.csv", "data/lists.json", "data/judge_log.jsonl", "judge_overrides.csv", "clip_sim.csv", "RUN_LOG.md", "REPORT.md", "frozen_check.txt", "env.sh", "results", "ext", "jobs", "logs"]
items += sorted(p.name for p in F.glob("trials_*.jsonl")) + sorted(p.name for p in F.glob("raw_*.jsonl")) + sorted(p.name for p in F.glob("scored_*.csv")) + sorted(p.name for p in F.glob("review_*.csv"))
for it in items:
    s = F/it
    if s.is_dir():
        for f in sorted(s.rglob("*")):
            if f.is_file(): put(f, str(f.relative_to(F)))
    elif s.exists(): put(s, it)
key = ""
try: key = json.load(open(H/".kaggle"/"kaggle.json"))["key"]
except Exception: pass
pat = re.compile(r"hf_[A-Za-z0-9]{20,}")
for f in body.rglob("*"):
    if f.is_file() and f.suffix in TEXT:
        t = f.read_text(errors="replace")
        if pat.search(t) or (key and key in t): print("SECRET HIT in", f); shutil.rmtree(stage); sys.exit(5)
manifest = []
for f in sorted(body.rglob("*")):
    if f.is_file(): manifest.append(hashlib.sha256(f.read_bytes()).hexdigest() + "  " + str(f.relative_to(body)))
up = stage/"upload"; up.mkdir()
(up/"MANIFEST.sha256").write_text("\n".join(manifest) + "\n")
arc = up/f"fullpool_return_{date}_{tag}.tar.gz"
with tarfile.open(arc, "w:gz") as t: t.add(body, arcname="fullpool")
for n in ("REPORT.md", "RUN_LOG.md"): shutil.copy2(body/n, up/n)
print("archive", arc.name, round(arc.stat().st_size/1e6, 1), "MB;", len(manifest), "files", flush=True)
prefix = f"culturehaystack/fullpool_return_{date}_{tag}"
env = dict(os.environ); env.pop("HF_HUB_OFFLINE", None); env.pop("TRANSFORMERS_OFFLINE", None)
r = subprocess.run([sys.executable, "-c", "import sys; sys.path.insert(0, sys.argv[1]); import huggingface_hub as h; print(h.__version__); h.sync_bucket(sys.argv[2], sys.argv[3])", str(H/"hfup_lib"), str(up), f"{os.environ['CH_TEMP_BUCKET']}/{prefix}"], capture_output=True, text=True, env=env)
print(r.stdout[-1000:], r.stderr[-600:])
if r.returncode != 0: sys.exit(4 if re.search(r"401|403|Unauthorized|Forbidden", r.stdout + r.stderr) else 1)
with open(F/"RUN_LOG.md", "a") as f: f.write(f"- {datetime.datetime.now():%Y-%m-%d %H:%M} [upload] FINAL package {tag} -> hf bucket {prefix}\n")
print("PACKAGE_UPLOAD_OK", prefix)
