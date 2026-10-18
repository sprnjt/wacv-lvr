#!/usr/bin/env bash
# One-time pod setup (~15-25 min, mostly downloads). From the repo root on the pod:
#   export HF_TOKEN=hf_...  KAGGLE_USERNAME=...  KAGGLE_KEY=...
#   bash runpod/setup.sh 2>&1 | tee logs/setup.log
# Idempotent: re-running skips what is already there.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs runpod/state
export HF_HOME=${HF_HOME:-/workspace/hf}      # on the volume: survives pod restarts
: "${HF_TOKEN:?export HF_TOKEN=hf_... (a token whose account accepted the google/gemma-3-4b-it license)}"

python3 -c 'import sys; assert sys.version_info >= (3, 10), "transformers 5.x needs Python >= 3.10"'

# --- torch: keep the template's CUDA build when recent enough -------------------------------
if ! python3 -c 'import torch, sys; v = tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2]); sys.exit(0 if v >= (2, 6) and torch.cuda.is_available() else 1)' 2>/dev/null; then
  echo "installing torch (cu128)"; pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu128
fi
# Versions tested CPU-side for this run (work/cpu_check.py). The main study used "transformers 5.x"
# (exact minor not recorded); run_all.sh's repro step measures any drift against the frozen raw files.
# Deliberately NOT installed: causal_conv1d / kernels (LFM ran on the reference conv path).
pip install -q "torchvision==0.24.1" --index-url https://download.pytorch.org/whl/cu128 || true   # LFM/Qwen processors need it
pip install -q "transformers==5.20.1" "accelerate==1.15.0" pillow pandas numpy scipy statsmodels matplotlib kaggle

# --- README host note: the system cuDNN in LD_LIBRARY_PATH can shadow the wheel's and crash torch.
# Use `env -u LD_LIBRARY_PATH` when CUDA works without it (it normally does on RunPod).
if env -u LD_LIBRARY_PATH python3 -c 'import torch; assert torch.cuda.is_available(); torch.ones(1).cuda()' 2>/dev/null; then
  echo 'RUN="env -u LD_LIBRARY_PATH"' > runpod/state/env.sh
elif python3 -c 'import torch; assert torch.cuda.is_available(); torch.ones(1).cuda()'; then
  echo 'RUN=""' > runpod/state/env.sh
else
  echo "FATAL: CUDA not usable from torch"; exit 1
fi
python3 -c 'import torch, transformers; print("torch", torch.__version__, "cuda", torch.version.cuda, "|", torch.cuda.get_device_name(0), "| transformers", transformers.__version__)'

# --- frozen study data (data/ and results/ are not in git; they ship as zips) ----------------
[ -f data/pool.csv ] || unzip -q -n data.zip
[ -d results ] || unzip -q -n results.zip

# --- dataset -> images/prepared ---------------------------------------------------------------
if [ ! -f dataset/metadata.csv ]; then
  if [ -z "${KAGGLE_USERNAME:-}${KAGGLE_API_TOKEN:-}" ] && [ ! -f ~/.kaggle/kaggle.json ]; then
    echo "FATAL: no Kaggle credentials (export KAGGLE_USERNAME + KAGGLE_KEY), or scp dataset/ here"; exit 1
  fi
  kaggle datasets download -d shinjinichakraborty/world-cultural-items-1k -p dataset
  unzip -q -o dataset/world-cultural-items-1k.zip -d dataset
fi
# The Kaggle dataset must be the version the study was built from: rebuild data/ in a scratch dir
# (same filters + exclude.txt) and require an identical pool.
python3 - <<'EOF'
import ch, filecmp, tempfile, contextlib, io
from pathlib import Path
d = Path(tempfile.mkdtemp())
with contextlib.redirect_stdout(io.StringIO()):
    ch.build_data("dataset/metadata.csv", out=d)
assert filecmp.cmp(d / "pool.csv", "data/pool.csv", shallow=False), "dataset version differs from data/pool.csv"
assert filecmp.cmp(d / "lists.json", "data/lists.json", shallow=False), "distractor lists differ"
print("dataset matches frozen data/pool.csv + lists.json")
EOF
python3 ch.py selftest --meta dataset/metadata.csv | tail -1
if [ "$(ls images/prepared/*.jpg 2>/dev/null | wc -l)" -ne 890 ]; then
  python3 ch.py prep --src dataset | tail -1
fi
[ "$(ls images/prepared/*.jpg | wc -l)" -eq 890 ] || { echo "FATAL: expected 890 prepared images"; exit 1; }
python3 make_interventions.py
if git rev-parse --git-dir >/dev/null 2>&1 && ! git diff --quiet -- trials_interventions.jsonl; then
  echo "WARNING: trials_interventions.jsonl changed vs git"
fi
# E9 collage (exploratory): trials ship with the repo (deterministic); build images/collage from images/prepared.
# Non-fatal: a failure here only makes the E9 steps fail; every other job is unaffected.
{ [ -f trials_collage.jsonl ] || python3 make_collage.py trials; } && python3 make_collage.py images ||
  echo "WARNING: E9 collage build failed (E9 steps will fail; other jobs unaffected)"

# --- models: LFM now (first job); the rest are prefetched in the background by run_all.sh -----
python3 - <<'EOF'
from huggingface_hub import model_info, snapshot_download
try:
    model_info("google/gemma-3-4b-it")
    print("gemma-3-4b-it: access OK")
except Exception as e:
    print("WARNING: no access to google/gemma-3-4b-it (job 4a will fail):", type(e).__name__)
snapshot_download("LiquidAI/LFM2.5-VL-3B")
print("LFM2.5-VL-3B downloaded")
EOF
df -h "${HF_HOME}" | tail -1
echo "SETUP OK"
