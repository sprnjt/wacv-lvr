"""Phase 02 helper: download the two Qwen models (post-registration run). HF token comes from the environment."""
import os
os.environ["HF_HOME"]=os.path.expanduser("~/hf")
from huggingface_hub import snapshot_download
for r in ["Qwen/Qwen3-VL-8B-Instruct","Qwen/Qwen2.5-VL-7B-Instruct"]:
    try:
        p=snapshot_download(r)
        print("MODEL_OK",r,p.split("/")[-1],flush=True)
    except Exception as e:
        print("MODEL_FAIL",r,type(e).__name__,str(e)[:200],flush=True)
