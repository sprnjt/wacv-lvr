#!/bin/bash
# P0 reproduction check for one model: 10 frozen main trials x 2 modes, then compare. Usage: p0.sh <model_key>
set -uo pipefail
M=$1
source $HOME/CultureHaystack_handoff/fullpool/env.sh
cd $HOME/CultureHaystack_handoff/fullpool
rm -f logs/repro_$M.jsonl
python ../wacv-lvr/run_models.py $M --trials ../wacv-lvr/trials_main.jsonl --repeat --out logs/repro_$M.jsonl > logs/p0_$M.out 2>&1
python ext/check_repro.py $M >> logs/p0_$M.out 2>&1; echo "exit=$?" >> logs/p0_$M.out
