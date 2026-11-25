"""Phase 02 helper: download the remaining models (Granite, LFM, Gemma-4-12B, CLIP). Token from environment."""
import os
os.environ["HF_HOME"]=os.path.expanduser("~/hf")
from huggingface_hub import snapshot_download
for r in ["ibm-granite/granite-vision-4.1-4b","LiquidAI/LFM2.5-VL-3B","google/gemma-4-12B-it","openai/clip-vit-base-patch32"]:
    try:
        p=snapshot_download(r)
        print("MODEL_OK",r,p.split("/")[-1],flush=True)
    except Exception as e:
        print("MODEL_FAIL",r,type(e).__name__,str(e)[:200],flush=True)
