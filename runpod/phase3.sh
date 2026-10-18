#!/bin/bash
# Phase 3 (post-freeze): collage experiment (E9) on the two remaining >=7B models. Fresh pod -> setup -> repro + E9.
cd /workspace/wacv-lvr
export HF_TOKEN=$(tr '\0' '\n' < /proc/1/environ | sed -n 's/^HF_TOKEN=//p')
export HF_HOME=/workspace/hf PIP_BREAK_SYSTEM_PACKAGES=1
mkdir -p dataset logs runpod/state; touch runpod/state/prefetch.done   # skip run_all.sh prefetch of models not needed here
nohup python3 -c 'from huggingface_hub import snapshot_download as s
for r in ("Qwen/Qwen2.5-VL-7B-Instruct","google/gemma-4-12B-it"): s(r); print("ok", r, flush=True)' > logs/prefetch3.log 2>&1 &
if [ ! -f dataset/metadata.csv ]; then
  curl -sL -o /workspace/wci.zip https://www.kaggle.com/api/v1/datasets/download/shinjinichakraborty/world-cultural-items-1k && unzip -q -o /workspace/wci.zip -d dataset
fi
echo "[$(date +%T)] setup"; bash runpod/setup.sh > logs/setup3.log 2>&1 || { echo "SETUP FAILED"; tail -n 5 logs/setup3.log; exit 1; }
python3 make_collage.py images > logs/collage_images.log 2>&1; tail -n 1 logs/collage_images.log
wait
echo "[$(date +%T)] lanes"
JOBS="repro:qwen25vl collage:qwen25vl repro:gemma4 collage:gemma4" bash runpod/run_all.sh > logs/lane3.log 2>&1
grep -h REPRO logs/repro_qwen25vl.log logs/repro_gemma4.log
echo "[$(date +%T)] PHASE3 END"
