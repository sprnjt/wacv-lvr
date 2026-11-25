"""Phase 02: unpack the uploaded Kaggle zip, run the byte-identical version check (temp dir only), prepare 890 images, link into fullpool/."""
import os, sys, glob, filecmp, tempfile, contextlib, io, zipfile
from pathlib import Path
H = Path.home(); W = H/"CultureHaystack_handoff"/"wacv-lvr"; F = H/"CultureHaystack_handoff"/"fullpool"
z = H/"archive (8).zip"
print("ZIP", z.exists(), z.stat().st_size if z.exists() else 0, flush=True)
dst = H/"dataset_upload"; dst.mkdir(exist_ok=True)
with zipfile.ZipFile(z) as zf:
    print("ZIP_ENTRIES", len(zf.namelist()), zf.namelist()[:3], flush=True)
    zf.extractall(dst)
meta = glob.glob(str(dst/"**"/"metadata.csv"), recursive=True)
print("METADATA", meta, flush=True)
if not meta: sys.exit("no metadata.csv")
root = Path(meta[0]).parent
sys.path.insert(0, str(W)); os.chdir(W)
import ch
d = Path(tempfile.mkdtemp())
with contextlib.redirect_stdout(io.StringIO()):
    ch.build_data(str(meta[0]), out=d)
ok = filecmp.cmp(d/"pool.csv", "data/pool.csv", shallow=False) and filecmp.cmp(d/"lists.json", "data/lists.json", shallow=False)
print("VERSION_CHECK", "PASS" if ok else "FAIL", flush=True)
if not ok: sys.exit(1)
out = H/"images_prepared"
ch.prep(str(root), out=str(out))
n = len(list(out.glob("img_*.jpg")))
print("PREPARED_IMAGES", n, flush=True)
(F/"images").mkdir(exist_ok=True)
lk = F/"images"/"prepared"
if not lk.exists(): lk.symlink_to(out)
print("DONE", flush=True)
