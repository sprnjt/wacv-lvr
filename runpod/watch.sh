#!/bin/bash
# Local watcher (runs on the Mac). Polls the pod every 2 min and EXITS with a reason as soon as something
# needs the lead's attention, so the lead (Claude) is woken, fixes it, and relaunches the watcher.
# Exit reasons: FAIL (new failed step), TRACEBACK / OOM / OFFLOAD in a live log, STALL (no output growth
# for STALL_MIN while a GPU job runs), IDLE (no GPU job running but work remains), MILESTONE, SSH.
STALL_MIN=${STALL_MIN:-12}
REMOTE='cd /workspace/wacv-lvr
echo "FAILS $(grep -h FAIL logs/lane*.log 2>/dev/null | sort -u | wc -l)"
echo "PROCS $(pgrep -fc "[r]un_models.py|[c]ausal_probe.py run")"
echo "ROWS $(cat raw_*_interventions.jsonl raw_*_collage.jsonl work/smoke/*.jsonl causal_lfm25v.jsonl causal_qwen3vl.jsonl attn_lfm25v.jsonl 2>/dev/null | wc -l)"
for f in $(ls -t logs/*.log | head -n 6); do tail -c 3000 "$f" | grep -q -E "Traceback|CUDA out of memory|OutOfMemoryError" && echo "ERRLOG $f"; tail -c 3000 "$f" | grep -q "offloaded to the cpu" && echo "OFFLOAD $f"; done
for s in causal_qwen3vl E9_qwen3vl attn_lfm25v E9_gemma3_4b E7_gemma3_4b repro_qwen25vl E9_qwen25vl repro_gemma4 E9_gemma4; do [ -f runpod/state/$s.done ] && echo "DONE $s"; done
nvidia-smi --query-gpu=memory.used --format=csv,noheader'
snap() { ssh -o ConnectTimeout=20 -o BatchMode=yes ${HOST:-wacvpod} "$REMOTE" 2>&1; }
s=$(snap) || { echo "SSH failed: $s"; exit 2; }
fails0=$(echo "$s" | awk '/^FAILS/{print $2}'); rows=$(echo "$s" | awk '/^ROWS/{print $2}'); done0=$(echo "$s" | grep -c '^DONE'); last=$SECONDS
echo "start: $(echo "$s" | tr '\n' ' ')"
while true; do
  sleep 120
  s=$(snap) || { sleep 60; s=$(snap) || { echo "SSH failed twice: $s"; exit 2; }; }
  f=$(echo "$s" | awk '/^FAILS/{print $2}'); r=$(echo "$s" | awk '/^ROWS/{print $2}'); p=$(echo "$s" | awk '/^PROCS/{print $2}'); d=$(echo "$s" | grep -c '^DONE')
  [ "$f" -gt "$fails0" ] && { echo "FAIL: new failed step"; echo "$s"; exit 1; }
  echo "$s" | grep -q '^ERRLOG' && { echo "TRACEBACK/OOM in live log"; echo "$s"; exit 1; }
  echo "$s" | grep -q '^OFFLOAD' && { echo "OFFLOAD: a model was loaded partly on CPU (slow)"; echo "$s"; exit 1; }
  [ "$d" -gt "$done0" ] && { echo "MILESTONE: a tracked job finished"; echo "$s"; exit 0; }
  if [ "$r" -gt "$rows" ]; then rows=$r; last=$SECONDS; fi
  [ "$p" -gt 0 ] && [ $((SECONDS - last)) -gt $((STALL_MIN * 60)) ] && { echo "STALL: no new output rows for ${STALL_MIN} min"; echo "$s"; exit 1; }
  [ "$p" -eq 0 ] && { echo "IDLE: no GPU job running"; echo "$s"; exit 0; }
done
