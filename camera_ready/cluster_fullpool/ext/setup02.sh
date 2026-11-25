#!/bin/bash
# Phase 02 (setup): conda env, Kaggle data + version check, prepared images, model weights. No GPU. Run in background.
set -uo pipefail
P=$HOME/CultureHaystack_handoff; F=$P/fullpool; cd $F
source $HOME/.culturehaystack_secrets
export HF_HOME=$HOME/hf PYTHONUNBUFFERED=1
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a $F/logs/setup02.log; }
log "env"
if [ ! -x $HOME/envs/ch/bin/python ]; then $HOME/miniconda3/bin/conda create -y -p $HOME/envs/ch python=3.11 >>$F/logs/setup02.log 2>&1; fi
PY=$HOME/envs/ch/bin/python
$PY -m pip install -q torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu126 >>$F/logs/setup02.log 2>&1 || log "torch install FAILED"
$PY -m pip install -q transformers==5.20.1 accelerate==1.15.0 pillow pandas numpy scipy statsmodels matplotlib huggingface_hub >>$F/logs/setup02.log 2>&1 || log "pip install FAILED"
$PY -m pip freeze | grep -iE 'torch|transformers|accelerate|pillow|huggingface' > $F/logs/env.txt; nvidia-smi >> $F/logs/env.txt
log "env done"
log "dataset"
mkdir -p $HOME/dataset && cd $HOME/dataset
[ -f world-cultural-items-1k.zip ] || curl -sL -o world-cultural-items-1k.zip https://www.kaggle.com/api/v1/datasets/download/shinjinichakraborty/world-cultural-items-1k
unzip -q -o world-cultural-items-1k.zip -d . ; ls | head >> $F/logs/setup02.log
cd $F; $PY $F/ext/setup02.py 2>&1 | tee -a $F/logs/setup02.log
log "setup02 finished"
