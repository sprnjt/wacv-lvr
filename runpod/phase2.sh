#!/bin/bash
# Phase 2 (after the pod restart that injected the hf_token secret): reinstall, gemma3_4b jobs, collect + HF upload.
# The token is read from PID 1's environment inside the pod and never printed.
cd /workspace/wacv-lvr
export HF_TOKEN=$(tr '\0' '\n' < /proc/1/environ | sed -n 's/^HF_TOKEN=//p')
export HF_HOME=/workspace/hf HF_REPO=${HF_REPO:-HiLEx-DLA/culturehaystack-outputs} PIP_BREAK_SYSTEM_PACKAGES=1
echo "[$(date +%T)] setup"; bash runpod/setup.sh > logs/setup2.log 2>&1 || { echo "SETUP FAILED"; tail -n 5 logs/setup2.log; exit 1; }
echo "[$(date +%T)] gemma download"
python3 -c 'from huggingface_hub import snapshot_download; snapshot_download("google/gemma-3-4b-it"); print("gemma ok")' > logs/gemma_dl.log 2>&1 || { echo "GEMMA DOWNLOAD FAILED"; tail -n 3 logs/gemma_dl.log; exit 1; }
echo "[$(date +%T)] lanes"
JOBS="interventions:gemma3_4b" bash runpod/run_all.sh > logs/laneG1.log 2>&1 &
sleep 60
JOBS="collage:gemma3_4b" bash runpod/run_all.sh > logs/laneG2.log 2>&1 &
wait
echo "[$(date +%T)] collect"; bash runpod/collect.sh > logs/collect2.log 2>&1; tail -n 3 logs/collect2.log
echo "[$(date +%T)] PHASE2 END"
