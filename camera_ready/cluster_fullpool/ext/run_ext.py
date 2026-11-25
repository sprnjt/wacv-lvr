"""G3 wrapper (phase 05): register gemma3_12b without touching run_models.py, then run its main() with the same CLI as run_models.
Usage (cwd = fullpool/): python ext/run_ext.py gemma3_12b --trials FILE [--modes mc,open] [--out FILE] [--repeat]"""
import sys
import run_models
run_models.MODELS["gemma3_12b"] = dict(hf_id="google/gemma-3-12b-it", family="gemma")
args = sys.argv[1:]
trials = args[args.index("--trials") + 1] if "--trials" in args else "trials_pilot.jsonl"
exps = args[args.index("--exps") + 1].split(",") if "--exps" in args else None
modes = tuple(args[args.index("--modes") + 1].split(",")) if "--modes" in args else ("open", "mc")
out = args[args.index("--out") + 1] if "--out" in args else None
run_models.main(args[0], trials, repeat="--repeat" in args, attn=False, exps=exps, modes=modes, out_path=out)
