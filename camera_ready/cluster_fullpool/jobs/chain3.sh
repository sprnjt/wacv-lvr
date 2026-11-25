#!/bin/bash
# Chain 3 (G3): wait until all G1/chain2 jobs are done (GPU room for the 24 GB model), then determinism check, frozen protocol, then G1.
cd $HOME/CultureHaystack_handoff/fullpool
source env.sh
while pgrep -f 'jobs/g1.sh|chain2.sh' > /dev/null; do sleep 30; done
rm -f logs/repro_gemma3_12b_a.jsonl logs/repro_gemma3_12b_b.jsonl
python ext/run_ext.py gemma3_12b --trials ../wacv-lvr/trials_main.jsonl --repeat --out logs/repro_gemma3_12b_a.jsonl > logs/g3_det_a.out 2>&1
python ext/run_ext.py gemma3_12b --trials ../wacv-lvr/trials_main.jsonl --repeat --out logs/repro_gemma3_12b_b.jsonl > logs/g3_det_b.out 2>&1
python ext/check_det.py > logs/g3_det.out 2>&1; rc=$?
cat logs/g3_det.out >> RUN_LOG.md
if [ $rc -ne 0 ]; then echo "G3 stopped: gemma3_12b not deterministic" >> RUN_LOG.md; exit 1; fi
python ext/run_ext.py gemma3_12b --trials ../wacv-lvr/trials_main.jsonl --out raw_gemma3_12b_main.jsonl > logs/g3_main.out 2>&1
python ext/run_ext.py gemma3_12b --trials trials_g1.jsonl --modes mc --out raw_gemma3_12b_g1.jsonl > logs/g3_g1.out 2>&1
echo "chain3 done" >> logs/chain3.out
