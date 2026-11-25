"""Phase 02 helper: Kaggle version check (temp dir only), image prep, model snapshots. Reads wacv-lvr read-only."""
import os, sys, glob, filecmp, tempfile, contextlib, io, json
from pathlib import Path
H=Path.home(); W=H/"CultureHaystack_handoff"/"wacv-lvr"; F=H/"CultureHaystack_handoff"/"fullpool"
sys.path.insert(0,str(W)); os.chdir(W)
import ch
meta=glob.glob(str(H/"dataset"/"**"/"metadata.csv"),recursive=True); print("metadata:",meta)
root=Path(meta[0]).parent
d=Path(tempfile.mkdtemp())
with contextlib.redirect_stdout(io.StringIO()): ch.build_data(str(meta[0]),out=d)
ok=filecmp.cmp(d/"pool.csv","data/pool.csv",shallow=False) and filecmp.cmp(d/"lists.json","data/lists.json",shallow=False)
print("VERSION_CHECK", "PASS" if ok else "FAIL")
if not ok: sys.exit(1)
out=H/"images_prepared"; ch.prep(str(root),out=str(out))
n=len(list(out.glob("img_*.jpg"))); print("PREPARED_IMAGES",n)
os.chdir(F); (F/"images").mkdir(exist_ok=True)
lk=F/"images"/"prepared"
if not lk.exists(): lk.symlink_to(out)
os.environ["HF_HOME"]=str(H/"hf")
from huggingface_hub import snapshot_download, model_info
revs=[]
for r in ["Qwen/Qwen3-VL-8B-Instruct","Qwen/Qwen2.5-VL-7B-Instruct","google/gemma-3-4b-it","ibm-granite/granite-vision-4.1-4b","LiquidAI/LFM2.5-VL-3B","google/gemma-3-12b-it","google/gemma-4-12B-it","openai/clip-vit-base-patch32"]:
    try:
        p=snapshot_download(r); revs.append(f"{r} {Path(p).name}"); print("MODEL_OK",r,flush=True)
    except Exception as e:
        print("MODEL_FAIL",r,type(e).__name__,str(e)[:150],flush=True)
(F/"logs"/"model_revisions.txt").write_text("\n".join(revs)+"\n")
print("NOT_DOWNLOADED: google/gemma-4-31B-it (judge, ~62GB, does not fit 40GB GPU; pending decision)")
