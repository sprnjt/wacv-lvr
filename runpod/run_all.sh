#!/usr/bin/env bash
# All GPU jobs, in priority order, each preceded by a 2-3 item smoke test. Resumable: re-run the same
# command after any crash/restart; finished steps are skipped (runpod/state/*.done) and every job
# script itself resumes from its output file.
#
#   nohup bash runpod/run_all.sh > logs/run_all.log 2>&1 &
#   tail -f logs/run_all.log                     # one line per step; details in logs/<step>.log
#
# Knobs (env):  STOP_AFTER=3   run jobs 1..3 only (cut line)
#               MAX_HOURS=11   do not START a new GPU step after this many hours of this invocation
#               AUTO_STOP=1    run collect.sh, then stop the pod (billing stops; volume is kept)
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs runpod/state work/smoke
export HF_HOME=${HF_HOME:-/workspace/hf} PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
RUN="env -u LD_LIBRARY_PATH"; [ -f runpod/state/env.sh ] && source runpod/state/env.sh
STOP_AFTER=${STOP_AFTER:-5}; MAX_HOURS=${MAX_HOURS:-11}
GPU="$RUN python3"
TIMEOUT=$(command -v timeout >/dev/null && echo "timeout 4h")   # a hung step cannot eat the budget

{ date; nvidia-smi; python3 -m pip freeze | grep -iE '^(torch|transformers|accelerate|pillow|huggingface)'; git rev-parse HEAD 2>/dev/null; } > logs/env.txt 2>&1

# Background prefetch of the later models while the LFM jobs run (hub file locks make this safe).
if [ ! -f runpod/state/prefetch.done ]; then
  nohup python3 -c '
from huggingface_hub import snapshot_download
for r in ("Qwen/Qwen3-VL-8B-Instruct", "google/gemma-3-4b-it", "ibm-granite/granite-vision-4.1-4b"):
    try: snapshot_download(r); print("ok", r, flush=True)
    except Exception as e: print("FAILED", r, e, flush=True)
' > logs/prefetch.log 2>&1 && touch runpod/state/prefetch.done &
fi

# step NAME CMD... : run once, log to logs/NAME.log, mark done; never aborts the script.
step() {
  local name=$1; shift
  if [ -f "runpod/state/$name.done" ]; then echo "[skip] $name"; return 0; fi
  if [ "$SECONDS" -gt $((MAX_HOURS * 3600)) ]; then echo "[time] MAX_HOURS reached, not starting $name"; return 1; fi
  echo "[$(date +%T)] start $name"
  local t0=$SECONDS
  $TIMEOUT "$@" >> "logs/$name.log" 2>&1
  local rc=$?
  if [ $rc -eq 0 ]; then
    touch "runpod/state/$name.done"; echo "[$(date +%T)] done  $name ($(( (SECONDS - t0) / 60 )) min)"
  else
    echo "[$(date +%T)] FAIL  $name (exit $rc) -> logs/$name.log"; return 1
  fi
}

# A failed check forgets its smoke run, so the next invocation re-smokes after a fix.
unsmoke() { rm -f "runpod/state/$1.done" "$2"; return 1; }

# Two items of a JSONL file (head, or one per experiment for the interventions file).
python3 - <<'EOF'
import ch
I = ch.read_jsonl("trials_interventions.jsonl")
ch.write_jsonl("work/smoke/iv_trials.jsonl", [next(t for t in I if t["exp"] == "E8"), next(t for t in I if t["exp"] == "E7" and t["N"] == 5)])
ch.write_jsonl("work/smoke/attn_trials.jsonl", [t for t in ch.read_jsonl("trials_main.jsonl") if t["exp"] == "E2"][:2])
ch.write_jsonl("work/smoke/e9_trials.jsonl", ch.read_jsonl("trials_collage.jsonl")[:4])   # one target, all 4 conditions
EOF

causal() {   # causal MODEL
  local m=$1
  step "select_$m" python3 causal_probe.py select "$m" --limit 128 || return 1
  head -3 "causal_cases_$m.jsonl" > "work/smoke/cases_$m.jsonl"
  step "smoke_causal_$m" $GPU causal_probe.py run "$m" --cases "work/smoke/cases_$m.jsonl" --out "work/smoke/causal_$m.jsonl" &&
  { step "check_causal_$m" python3 runpod/check.py causal "work/smoke/causal_$m.jsonl" || unsmoke "smoke_causal_$m" "work/smoke/causal_$m.jsonl"; } &&
  step "causal_$m" $GPU causal_probe.py run "$m" &&
  step "summary_causal_$m" python3 causal_probe.py summarize "$m"
}

repro() {    # repro MODEL : first 10 main trials x 2 modes again, compared with the frozen raw file
  local m=$1
  step "repro_$m" $GPU run_models.py "$m" --trials trials_main.jsonl --repeat --out "work/smoke/repro_$m.jsonl" &&
  python3 runpod/check.py repro "$m" "work/smoke/repro_$m.jsonl" | tee -a "logs/repro_$m.log" | grep REPRO
}

interventions() {   # interventions MODEL : E8 then E7, MC only, into raw_MODEL_interventions.jsonl
  local m=$1 out="raw_$1_interventions.jsonl"
  repro "$m"
  step "smoke_iv_$m" $GPU run_models.py "$m" --trials work/smoke/iv_trials.jsonl --modes mc --out "work/smoke/iv_$m.jsonl" &&
  { step "check_iv_$m" python3 runpod/check.py mc "work/smoke/iv_$m.jsonl" work/smoke/iv_trials.jsonl || unsmoke "smoke_iv_$m" "work/smoke/iv_$m.jsonl"; } &&
  step "E8_$m" $GPU run_models.py "$m" --trials trials_interventions.jsonl --exps E8 --modes mc --out "$out" &&
  step "E7_$m" $GPU run_models.py "$m" --trials trials_interventions.jsonl --exps E7 --modes mc --out "$out"
  [ -f "$out" ] && python3 analyze_interventions.py "$m" --raw "$out" >> "logs/E7_$m.log" 2>&1
  return 0
}

collage() {   # collage MODEL : E9 (4 conditions x 160 targets), MC only, into raw_MODEL_collage.jsonl
  local m=$1 out="raw_$1_collage.jsonl"
  step "smoke_E9_$m" $GPU run_models.py "$m" --trials work/smoke/e9_trials.jsonl --modes mc --out "work/smoke/e9_$m.jsonl" &&
  { step "check_E9_$m" python3 runpod/check.py mc "work/smoke/e9_$m.jsonl" work/smoke/e9_trials.jsonl || unsmoke "smoke_E9_$m" "work/smoke/e9_$m.jsonl"; } &&
  step "E9_$m" $GPU run_models.py "$m" --trials trials_collage.jsonl --modes mc --out "$out"
  [ -f "$out" ] && python3 analyze_collage.py "$m" --raw "$out" >> "logs/E9_$m.log" 2>&1
  return 0
}

attention() { attention_lfm; }
attention_lfm() {
  step smoke_attn_lfm25v $GPU run_models.py lfm25v --trials work/smoke/attn_trials.jsonl --exps E2 --attn --out work/smoke/attn_lfm25v.jsonl &&
  { step check_attn_lfm25v python3 runpod/check.py attn work/smoke/attn_lfm25v.jsonl work/smoke/attn_trials.jsonl || unsmoke smoke_attn_lfm25v work/smoke/attn_lfm25v.jsonl; } &&
  step attn_lfm25v $GPU run_models.py lfm25v --trials trials_main.jsonl --exps E2 --attn     # -> attn_lfm25v.jsonl (1,600 rows)
}

echo "=== run_all start $(date)  STOP_AFTER=$STOP_AFTER MAX_HOURS=$MAX_HOURS"
# Parallel lanes: JOBS="causal:lfm25v interventions:lfm25v" runs just those jobs, in order (one lane).
# Launch several lanes with disjoint JOBS to share the GPU (see runpod/lanes.sh).
if [ -n "${JOBS:-}" ]; then
  for j in $JOBS; do "${j%%:*}" "${j#*:}"; done
  echo "=== lane end $(date) ($(( SECONDS / 60 )) min)"; exit 0
fi
[ "$STOP_AFTER" -ge 1 ] && causal lfm25v                                     # job 1
[ "$STOP_AFTER" -ge 2 ] && interventions lfm25v                              # job 2
[ "$STOP_AFTER" -ge 3 ] && causal qwen3vl                                    # job 3
[ "$STOP_AFTER" -ge 4 ] && for m in lfm25v gemma3_4b granite4v qwen3vl; do collage "$m"; done   # E9, after cut line A
[ "$STOP_AFTER" -ge 4 ] && { interventions gemma3_4b; interventions granite4v; }   # job 4
[ "$STOP_AFTER" -ge 5 ] && attention_lfm                                     # job 5

# --- CPU post-processing ------------------------------------------------------------------------
if [ -f attn_lfm25v.jsonl ] && [ "$(wc -l < attn_lfm25v.jsonl)" -eq 1600 ]; then
  # Separate table: analyze.py main() would overwrite the hand-edited results/summary.md.
  python3 -c 'import analyze; analyze.attention(analyze.load(["lfm25v"]), ["lfm25v"]).to_csv("results/attention_capture_lfm25v.csv", index=False)' > logs/analyze.log 2>&1
  python3 verify_results.py > logs/verify.log 2>&1; tail -1 logs/verify.log     # read-only audit
fi
echo "=== run_all end $(date) ($(( SECONDS / 60 )) min)"
grep -h "FAIL" logs/run_all.log 2>/dev/null | sort -u || true
if [ "${AUTO_STOP:-0}" = 1 ]; then
  bash runpod/collect.sh && runpodctl stop pod "$RUNPOD_POD_ID"
fi
