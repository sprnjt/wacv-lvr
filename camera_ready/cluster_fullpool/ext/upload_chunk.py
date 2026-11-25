"""Stage, scrub and upload finished result files to the temporary HF bucket under a NEW prefix each time (never overwrites, never deletes).
Usage (cwd = fullpool/): python ext/upload_chunk.py <chunk_name> <path> [<path> ...]    exit 4 = auth failure, 5 = secret scan hit."""
# Progress checkpoints and the final return were copied to a temporary Hugging Face bucket that existed only
# during the run (now deleted). All final data is in the Hugging Face dataset repository
# shinjinii/culturehaystack-results. The bucket address is read from the CH_TEMP_BUCKET environment variable.
import os, re, sys, shutil, getpass, socket, subprocess, datetime, json
from pathlib import Path
H = Path.home(); F = H/"CultureHaystack_handoff"/"fullpool"; os.chdir(F)
name, paths = sys.argv[1], sys.argv[2:]
stamp = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M")
prefix = f"culturehaystack/fullpool_progress_{stamp}_{name}"
stage = H/"upload_stage"/f"{stamp}_{name}"; stage.mkdir(parents=True)
TEXT = {".jsonl", ".csv", ".md", ".out", ".log", ".txt", ".json", ".py", ".sh", ".yaml"}
BAD = ("secret", "kaggle", ".culturehaystack", "images/", "images_prepared")
subs = [(str(H), "<PROJ>"), (re.compile(r"proxy-[A-Za-z0-9.-]+"), "<proxy>"), (re.compile(r"\bscn\d+(-\w+)?\b"), "<host>"),
        (getpass.getuser(), "<user>"), (socket.gethostname(), "<host>"), (re.compile(r"\bisea\d*\b"), "<user>")]
def put(src, rel):
    if any(b in rel for b in BAD):
        print("SKIP", rel); return
    dst = stage/rel; dst.parent.mkdir(parents=True, exist_ok=True)
    if src.suffix in TEXT:
        t = src.read_text(errors="replace")
        for a, b in subs:
            t = t.replace(a, b) if isinstance(a, str) else a.sub(b, t)
        dst.write_text(t)
    else:
        shutil.copy2(src, dst)
for p in paths:
    s = Path(p)
    if s.is_dir():
        for f in sorted(s.rglob("*")):
            if f.is_file() and str(f.resolve()).startswith(str(F.resolve())): put(f, str(f.relative_to(F)) if f.is_absolute() else str(f))
    elif s.exists():
        put(s, str(s.relative_to(F)) if s.is_absolute() else p)
    else:
        print("MISSING", p)
key = ""
try: key = json.load(open(H/".kaggle"/"kaggle.json"))["key"]
except Exception: pass
pat = re.compile(r"hf_[A-Za-z0-9]{20,}")
for f in stage.rglob("*"):
    if f.is_file() and f.suffix in TEXT:
        t = f.read_text(errors="replace")
        if pat.search(t) or (key and key in t):
            print("SECRET HIT in", f); shutil.rmtree(stage); sys.exit(5)
n = sum(1 for f in stage.rglob("*") if f.is_file()); size = sum(f.stat().st_size for f in stage.rglob("*") if f.is_file())
print(f"staged {n} files, {size/1e6:.1f} MB -> {prefix}", flush=True)
env = dict(os.environ); env.pop("HF_HUB_OFFLINE", None); env.pop("TRANSFORMERS_OFFLINE", None)
r = subprocess.run([sys.executable, "-c", "import sys; sys.path.insert(0, sys.argv[1]); import huggingface_hub as h; print(h.__version__); h.sync_bucket(sys.argv[2], sys.argv[3])", str(H/"hfup_lib"), str(stage), f"{os.environ['CH_TEMP_BUCKET']}/{prefix}"],
                   capture_output=True, text=True, env=env)
print(r.stdout[-1500:], r.stderr[-800:])
if r.returncode != 0:
    sys.exit(4 if re.search(r"401|403|Unauthorized|Forbidden", r.stdout + r.stderr) else 1)
with open(F/"RUN_LOG.md", "a") as f:
    f.write(f"- {datetime.datetime.now():%Y-%m-%d %H:%M} [upload] chunk {name}: {n} files, {size/1e6:.1f} MB -> hf bucket {prefix}\n")
print("UPLOAD_OK", prefix)
