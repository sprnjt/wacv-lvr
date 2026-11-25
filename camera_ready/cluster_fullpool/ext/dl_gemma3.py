"""Phase 02 helper: download Gemma-3-4B and 12B first (post-registration run). Run in background with HF token sourced."""
import os
os.environ["HF_HOME"]=os.path.expanduser("~/hf")
from huggingface_hub import snapshot_download
for r in ["google/gemma-3-4b-it","google/gemma-3-12b-it"]:
    try:
        p=snapshot_download(r)
        print("MODEL_OK",r,p.split("/")[-1],flush=True)
    except Exception as e:
        print("MODEL_FAIL",r,type(e).__name__,str(e)[:200],flush=True)
