#!/usr/bin/env bash
# Pack every NEW output of this GPU session (never the frozen raw_/scored_/attn_ main-study files).
#   bash runpod/collect.sh        -> /workspace/gpu_outputs_<timestamp>.tar.gz   (OUT_DIR=... to change)
# Optional: with HF_TOKEN (write access) and HF_REPO (e.g. HiLEx-DLA/culturehaystack-outputs) set, also upload
# the output files (no images/, dataset/ or secrets) to that private HF dataset under gpu_outputs_<timestamp>/.
# The upload can never fail the collection, and the token is read from the environment only.
set -uo pipefail
cd "$(dirname "$0")/.."
stamp=gpu_outputs_$(date +%Y%m%d_%H%M)
out=${OUT_DIR:-/workspace}/$stamp.tar.gz
files=$(ls -d causal_lfm25v.jsonl causal_qwen3vl.jsonl causal_cases_*.jsonl raw_*_interventions.jsonl \
  raw_*_collage.jsonl attn_lfm25v.jsonl trials_interventions.jsonl trials_collage.jsonl results/causal_*.csv \
  results/interventions_*.csv results/collage_*.csv results/attention_capture_lfm25v.csv logs work/smoke runpod/state 2>/dev/null)
tar czf "$out" $files
rc=$?
ls -lh "$out"; tar tzf "$out" | grep -v '^logs/\|^work/smoke/\|^runpod/state/'

if [ -n "${HF_TOKEN:-}" ] && [ -n "${HF_REPO:-}" ]; then
  UP_TIMEOUT=$(command -v timeout >/dev/null && echo "timeout 30m")   # a hung upload must not block AUTO_STOP
  STAMP=$stamp $UP_TIMEOUT python3 - <<'EOF' || echo "WARNING: HF upload failed or timed out (the tarball above is unaffected)"
import glob, os, sys
tok, repo = os.environ["HF_TOKEN"], os.environ["HF_REPO"]
pats = ["raw_*_interventions.jsonl", "raw_*_collage.jsonl", "causal_lfm25v.jsonl", "causal_qwen3vl.jsonl",
        "causal_cases_*.jsonl", "attn_lfm25v.jsonl", "results/*.csv", "logs/**/*"]
files = sorted({f for p in pats for f in glob.glob(p, recursive=True) if os.path.isfile(f)})
bad = [f for f in files if f.startswith(("images/", "dataset/")) or tok.encode() in open(f, "rb").read()]
if bad:
    sys.exit(f"HF upload skipped: {len(bad)} file(s) are images/dataset or contain the token, e.g. {bad[0]}")
print(f"HF upload: {len(files)} files -> dataset {repo}/{os.environ['STAMP']}/", flush=True)
try:
    from huggingface_hub import HfApi
    api = HfApi(token=tok)
    api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
    api.upload_folder(repo_id=repo, repo_type="dataset", folder_path=".", path_in_repo=os.environ["STAMP"],
                      allow_patterns=files, commit_message=os.environ["STAMP"])
    print("HF upload OK")
except Exception as e:
    sys.exit(f"HF upload failed: {type(e).__name__}: {str(e).replace(tok, '***')[:300]}")
EOF
fi
exit $rc
