"""G2 runner (phase 06): patches the prompt via ch_g2, registers gemma3_12b, then run_models.main (MC only, resumable).
Usage (cwd = fullpool/): python ext/run_g2.py <model> --trials trials_g2_160.jsonl --out raw_<model>_g2_160.jsonl"""
import sys
import ch_g2  # noqa: F401  (patches ch.make_parts)
import run_models
run_models.MODELS["gemma3_12b"] = dict(hf_id="google/gemma-3-12b-it", family="gemma")
a = sys.argv[1:]
run_models.main(a[0], a[a.index("--trials") + 1], modes=("mc",), out_path=a[a.index("--out") + 1])
