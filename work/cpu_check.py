#!/usr/bin/env python3
"""CPU-only check of the GPU code paths with TINY random-weight models built from the real configs
(real processors/tokenizers/chat templates; no weights downloaded; fake images of real-ish sizes).

  cd <scratch dir with data/ symlinked>; python <repo>/work/cpu_check.py <repo>

Checks: image spans == N for LFM (tiled) and qwen3vl; generate_attn runs and is finite; causal
probe run/summarize end-to-end incl. resume, self-control delta == 0, wrong-donor gate path.
"""
import json, sys, os
from pathlib import Path
REPO = Path(sys.argv[1]).resolve()
# it writes fake images/prepared/*.jpg into the cwd: never let that be the repo (a run once left 890 there)
if Path.cwd().resolve() == REPO or REPO in Path.cwd().resolve().parents:
    sys.exit("run from a scratch dir outside the repo (see docstring)")
sys.path.insert(0, str(REPO))
import torch
from PIL import Image
import transformers
from transformers import AutoConfig, AutoProcessor
import ch, run_models, causal_probe

torch.manual_seed(0)
trials = {t["trial_id"]: t for t in ch.read_jsonl(REPO / "trials_main.jsonl")}
pool = ch.load_pool()
Path("images/prepared").mkdir(parents=True, exist_ok=True)
sizes = [(768, 576), (576, 768), (768, 768), (768, 512), (640, 768)]   # 768x768 / 640x768 tile in LFM
for n, i in enumerate(sorted(pool)):
    if not Path(f"images/prepared/{i}.jpg").exists():
        Image.effect_noise(sizes[n % 5], 40 + n % 50).convert("RGB").save(f"images/prepared/{i}.jpg")


def tiny(name):
    hf = run_models.MODELS[name]["hf_id"]
    cfg = AutoConfig.from_pretrained(hf)
    t, v = cfg.text_config, cfg.vision_config
    if name == "lfm25v":
        t.num_hidden_layers, t.layer_types = 4, ["conv", "full_attention", "conv", "full_attention"]
        t.hidden_size, t.intermediate_size, t.block_ff_dim = 64, 128, 128
        t.num_attention_heads, t.num_key_value_heads, t.conv_dim = 4, 2, 64
        v.hidden_size, v.intermediate_size, v.num_hidden_layers, v.num_attention_heads = 32, 64, 2, 2
        cfg.projector_hidden_size = 64
    else:
        t.num_hidden_layers, t.hidden_size, t.intermediate_size = 6, 64, 128
        t.num_attention_heads, t.num_key_value_heads, t.head_dim = 4, 2, 16
        t.rope_parameters = dict(t.rope_parameters or {}, mrope_section=[2, 3, 3])
        v.depth, v.hidden_size, v.intermediate_size, v.num_heads, v.out_hidden_size = 4, 32, 64, 2, 64
        v.deepstack_visual_indexes = [1, 2, 3]
    model = getattr(transformers, run_models.MODELS[name]["arch"])(cfg).eval().float()
    return model, AutoProcessor.from_pretrained(hf)


for name in ("lfm25v", "qwen3vl"):
    model, proc = tiny(name)
    run_models.load = lambda _n, m=model, p=proc: (m, p)
    fam = run_models.MODELS[name]["family"]
    # 1. spans: every E2 N=5 trial of a few targets, plus the matching N=1
    for tid in [k for k in trials if k.startswith("E2_") and k.endswith("_P1")][:15]:
        x = run_models.prep_inputs(model, proc, fam, trials[tid], "mc")
        sp = run_models.image_spans(x["input_ids"][0].tolist(), run_models.image_token_ids(proc))
        assert len(sp) == 5, (name, tid, len(sp))
        n1 = trials[tid.replace("E2_", "E1_").split("_N5")[0] + "_N1_k1_P1"]
        y = run_models.prep_inputs(model, proc, fam, n1, "mc")
        sp1 = run_models.image_spans(y["input_ids"][0].tolist(), run_models.image_token_ids(proc))
        k = trials[tid]["order"].index(trials[tid]["target"])
        assert len(sp1) == 1 and sp1[0][1] - sp1[0][0] == sp[k][1] - sp[k][0], (name, tid)
    print(name, "spans OK (5 per N=5 trial; N=1 target span length == N=5 target span length)")
    # 2. attention probe path (eager) on two trials, both modes, through run_models.main
    model.set_attn_implementation("eager")
    small = [trials[k] for k in list(trials) if k.startswith("E2_")][:2]
    ch.write_jsonl("smoke_trials.jsonl", small)
    gen = lambda t, mode, mx: run_models.generate_attn(model, proc, fam, t, mode, 4)
    Path(f"attn_smoke_{name}.jsonl").unlink(missing_ok=True)
    run_models.main(name, "smoke_trials.jsonl", attn=True, out_path=f"attn_smoke_{name}.jsonl", gen=gen)
    rr = ch.read_jsonl(f"attn_smoke_{name}.jsonl")
    assert len(rr) == 4 and all(len(r["attn"]) == 5 and sum(r["attn"]) > 0 for r in rr)
    print(name, "attention probe OK", [round(sum(r["attn"]), 3) for r in rr])
    model.set_attn_implementation("sdpa")
    # 3. causal probe end to end on 3 cases (incl. resume)
    cases = ch.read_jsonl(REPO / f"causal_cases_{name}.jsonl")[:3]
    ch.write_jsonl("smoke_cases.jsonl", cases)
    out = f"causal_smoke_{name}.jsonl"
    Path(out).unlink(missing_ok=True)
    # random weights fail the gate; force it open once to exercise the patch path, then test the gate
    real_parse = ch.parse
    ch.parse = lambda raw, mode, tr: {f: pool[c["wrong"]][f] for c in cases[:1] for f in ("item", "country")}
    causal_probe.run_cases(name, str(REPO / "trials_main.jsonl"), "smoke_cases.jsonl", out, "auto")
    n_first = len(ch.read_jsonl(out))
    causal_probe.run_cases(name, str(REPO / "trials_main.jsonl"), "smoke_cases.jsonl", out, "auto")  # resume: no-op
    assert len(ch.read_jsonl(out)) == n_first, "resume rewrote rows"
    ch.parse = real_parse
    rs = ch.read_jsonl(out)
    pat = [r for r in rs if r["intervention"] != "wrong_donor_gate_fail"]
    layers = sorted({r["layer"] for r in pat})
    assert pat and all(abs(r["delta_logodds"]) < 1e-4 for r in pat if r["intervention"] == "self_to_target_control")
    assert any(abs(r["delta_logodds"]) > 1e-4 for r in pat if r["intervention"] == "target_clean_to_target")
    if name == "qwen3vl":
        assert min(layers) == 3, layers      # deepstack layers 0-2 skipped
    assert all(abs(r["delta_logodds"]) > 0 for r in pat if r["intervention"] == "target_clean_to_target" and r["layer"] > 0), "null layer in grid"
    causal_probe.summarize(out, f"causal_smoke_{name}.csv")
    print(name, "causal OK; layers", layers, "rows", len(rs),
          "self-control max |delta|", max(abs(r["delta_logodds"]) for r in pat if r["intervention"] == "self_to_target_control"))
print("CPU CHECK OK")
