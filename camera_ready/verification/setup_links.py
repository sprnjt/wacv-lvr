"""Recreate the working-directory shortcuts used by post_registration/camera_ready_analyses/_frozen, _fullpool and
03_clip_shown/work. Hugging Face does not store symlinks, so they are listed in links.json (link -> target, both
relative to the package root). Run once after download:  python tools/setup_links.py"""
import json, os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
n = 0
for e in json.loads((ROOT / "tools/links.json").read_text()):
    link, target = ROOT / e["link"], ROOT / e["target"]
    if not target.exists():
        raise SystemExit(f"missing target: {e['target']}")
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink() or link.exists():
        continue
    link.symlink_to(os.path.relpath(target, link.parent))
    n += 1
print(f"created {n} links")
