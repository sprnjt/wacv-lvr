"""Phase 02 helper: download the Kaggle dataset with kagglehub using ~/.kaggle/kaggle.json (key never printed)."""
import json, os
k = json.load(open(os.path.expanduser("~/.kaggle/kaggle.json")))
os.environ["KAGGLE_USERNAME"] = k["username"]
os.environ["KAGGLE_KEY"] = k["key"]
os.environ["KAGGLEHUB_CACHE"] = os.path.expanduser("~/kagglehub")
import kagglehub
try:
    p = kagglehub.dataset_download("shinjinichakraborty/world-cultural-items-1k")
    print("KAGGLE_OK", p, flush=True)
except Exception as e:
    print("KAGGLE_FAIL", type(e).__name__, str(e)[:300], flush=True)
