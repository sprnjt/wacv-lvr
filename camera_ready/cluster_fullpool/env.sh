# Sourced by every job script. Offline model use, pinned env. Phase 04.
export HF_HOME=$HOME/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export PYTHONPATH=$HOME/CultureHaystack_handoff/wacv-lvr:$HOME/CultureHaystack_handoff/fullpool/ext
export PATH=$HOME/envs/ch/bin:$PATH
export PYTHONDONTWRITEBYTECODE=1
