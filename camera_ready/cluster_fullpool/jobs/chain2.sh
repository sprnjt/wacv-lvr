#!/bin/bash
# Chain 2: wait for LFM P0, run Gemma-4-12B P0, then G1 for each model whose P0 passed (exit=0). No code changes on failure.
cd $HOME/CultureHaystack_handoff/fullpool
while ! grep -q '^exit=' logs/p0_lfm25v.out 2>/dev/null; do sleep 20; done
jobs/p0.sh gemma4
for m in lfm25v gemma4; do
  if grep -q '^exit=0' logs/p0_$m.out; then jobs/g1.sh $m; else echo "P0 failed for $m: not run" >> RUN_LOG.md; fi
done
