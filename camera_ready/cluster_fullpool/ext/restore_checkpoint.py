"""Resume on a NEW pod: python restore_checkpoint.py <bucket prefix of a ckpt upload> (e.g. culturehaystack/fullpool_progress_20261115_0400_ckpt).
Downloads the checkpoint into ~/CultureHaystack_handoff/fullpool, then: pip env + models + images must exist (see setup scripts in ext/ and jobs/), then run: bash jobs/restart_queue.sh. Finished jobs and finished rows are skipped, so work continues exactly where it stopped. Needs HF_TOKEN. wacv-lvr/ comes from the original package, never from the checkpoint."""
# Progress checkpoints and the final return were copied to a temporary Hugging Face bucket that existed only
# during the run (now deleted). All final data is in the Hugging Face dataset repository
# shinjinii/culturehaystack-results. The bucket address is read from the CH_TEMP_BUCKET environment variable.
import sys, os, huggingface_hub as h
dst = os.path.expanduser("~/CultureHaystack_handoff/fullpool"); os.makedirs(dst, exist_ok=True)
h.sync_bucket(os.environ["CH_TEMP_BUCKET"] + "/" + sys.argv[1], dst)
print("restored to", dst, "- next: bash jobs/restart_queue.sh")
