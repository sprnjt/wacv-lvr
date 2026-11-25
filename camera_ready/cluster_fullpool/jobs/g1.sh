#!/bin/bash
# G1 (MC only) for one model on trials_g1.jsonl. Resumable. Usage: g1.sh <model_key>
set -uo pipefail
M=$1
source $HOME/CultureHaystack_handoff/fullpool/env.sh
cd $HOME/CultureHaystack_handoff/fullpool
python ../wacv-lvr/run_models.py $M --trials trials_g1.jsonl --modes mc --out raw_${M}_g1.jsonl > logs/g1_$M.out 2>&1
echo "exit=$?" >> logs/g1_$M.out
