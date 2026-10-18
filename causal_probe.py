#!/usr/bin/env python3
"""Exploratory activation-patching probe for CultureHaystack.

This deliberately lives outside the frozen main-study pipeline.  It tests a
specific causal prediction: replacing the target image's multi-image hidden
state with the same image's clean N=1 state should increase the likelihood of
the gold MC answer; putting a named wrong image's clean state in the target
slot should do the converse.

Typical use (run the selection once, then resume layer jobs freely):

  python causal_probe.py select lfm25v --limit 128
  env -u LD_LIBRARY_PATH python causal_probe.py run lfm25v            # --layers auto (default: 0,6,13,20,26 for LFM)
  python causal_probe.py summarize lfm25v

The default model is LFM because it is the only model with pre-registered H3a
context attraction.  qwen3vl is supported as a robust-model replication:

  python causal_probe.py select qwen3vl --limit 128
  env -u LD_LIBRARY_PATH python causal_probe.py run qwen3vl           # auto skips qwen3vl's deepstack layers 0-2

All outputs are new `causal_*` files and never alter raw/scored main-study
records.  Treat results as post-freeze exploratory.
"""
import argparse
import csv
import json
from pathlib import Path

import ch
import run_models


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def jsonl_append(path, record):
    with Path(path).open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def n1_index(trials):
    """Map image id to its matched E1/N=1 trial (same target/options/list)."""
    return {t["target"]: t for t in trials
            if t["exp"] == "E1" and t["N"] == 1 and t.get("variant") == "P1"}


def option_letter(options, value):
    return ch.LETTERS[options.index(value)] if value in options else None


def select_cases(model, trials_path, output, limit):
    """Select E2/N=5 MC mis-binding fields whose target is correct (that field) in its own E1/N=1 run.

    The wrong-image donor is NOT taken from an existing E1 trial: most named wrong images are
    distractors with no N=1 trial of their own (that requirement left qwen3vl with 7 cases). `run`
    builds it as the target's N=1 prompt with the wrong image swapped in, and gates each case on the
    model naming the wrong image's label for that field when it is shown alone."""
    trials = rows(trials_path)
    by_id = {t["trial_id"]: t for t in trials}
    n1 = n1_index(trials)
    pool = ch.load_pool()
    scored = []
    with open(f"scored_{model}.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["exp"] == "E2" and r["mode"] == "mc" and r["variant"] == "P1" and r["N"] == "5":
                scored.append(r)
    # N=1 is used only as a clean donor; require correctness on the scored field.
    n1_scored = {(r["trial_id"], r["mode"]): r for r in csv.DictReader(open(f"scored_{model}.csv", encoding="utf-8"))
                 if r["exp"] == "E1" and r["N"] == "1" and r["mode"] == "mc"}
    candidates = []
    for r in scored:
        tr = by_id[r["trial_id"]]
        for field in ("item", "country"):
            wrong = r[f"{field}_src"]
            if r[f"{field}_err"] != "mis_binding" or not wrong or wrong == tr["target"]:
                continue
            target_clean = n1.get(tr["target"])
            if not target_clean:
                continue
            # The source image must be recognized when it is alone. This avoids
            # confusing a failed donor representation with a causal rescue.
            # (The wrong-image donor is gated the same way at run time.)
            if n1_scored.get((target_clean["trial_id"], "mc"), {}).get(f"{field}_ok") != "1":
                continue
            gold = tr["target"]
            gold_letter = option_letter(tr[f"{field}_options"], pool[gold][field])
            wrong_letter = option_letter(tr[f"{field}_options"], pool[wrong][field])
            if not gold_letter or not wrong_letter:
                continue
            candidates.append(dict(case_id=f"{r['trial_id']}::{field}", model=model,
                                   trial_id=r["trial_id"], field=field, target=gold, wrong=wrong,
                                   target_clean_id=target_clean["trial_id"],
                                   gold_letter=gold_letter, wrong_letter=wrong_letter))
    # Deterministic, target-balanced prefix: no random post-hoc sampling.
    candidates.sort(key=lambda x: (x["target"], x["field"], x["trial_id"]))
    selected, seen = [], set()
    while candidates and len(selected) < limit:
        remaining = []
        for c in candidates:
            if c["target"] not in seen and len(selected) < limit:
                selected.append(c); seen.add(c["target"])
            else:
                remaining.append(c)
        if len(remaining) == len(candidates):
            selected.extend(remaining[:limit - len(selected)]); break
        candidates = remaining
    Path(output).write_text("".join(json.dumps(x) + "\n" for x in selected), encoding="utf-8")
    print(f"wrote {len(selected)} cases to {output}")


def language_layers(model):
    """Find the decoder block list without baking a model-family-specific path."""
    for path in ("model.language_model.layers", "language_model.layers", "model.layers", "layers"):
        obj = model
        try:
            for bit in path.split("."):
                obj = getattr(obj, bit)
            if hasattr(obj, "__len__") and len(obj):
                return obj
        except AttributeError:
            pass
    raise RuntimeError("cannot locate decoder layers; add this architecture's path to language_layers()")


def spans_for(inputs, processor):
    ids = inputs["input_ids"][0].tolist()
    spans = run_models.image_spans(ids, run_models.image_token_ids(processor))
    if not spans:
        raise RuntimeError("no image-token spans found")
    return spans


def tensor_from_output(out):
    return out[0] if isinstance(out, tuple) else out


def with_hidden(out, hidden):
    if isinstance(out, tuple):
        return (hidden,) + out[1:]
    return hidden


def capture_layers(model, inputs, layers):
    """Return {layer: post-block states} from ONE forward pass. No gradients are retained."""
    import torch
    captured, blocks = {}, language_layers(model)
    def make(layer):
        def hook(_, __, out):
            if layer in captured:
                raise RuntimeError(f"layer {layer} ran twice in one forward")
            captured[layer] = tensor_from_output(out).detach().cpu()
        return hook
    handles = [blocks[layer].register_forward_hook(make(layer)) for layer in layers]
    try:
        with torch.inference_mode():
            model(**inputs, use_cache=False)
    finally:
        for h in handles:
            h.remove()
    if set(captured) != set(layers):
        raise RuntimeError(f"captured layers {sorted(captured)}, expected {sorted(layers)}")
    return captured


def patch_layer(model, inputs, layer, recipient_span, donor_hidden, donor_span):
    """Run a forward pass, patching one image span after `layer`."""
    import torch
    rs, re = recipient_span; ds, de = donor_span
    if re - rs != de - ds:
        raise RuntimeError(f"incompatible image spans: recipient={re-rs}, donor={de-ds}")
    donor = donor_hidden[:, ds:de]
    def hook(_, __, out):
        hidden = tensor_from_output(out)
        # clone prevents an in-place edit of a tensor retained by an architecture's cache.
        patched = hidden.clone()
        patched[:, rs:re] = donor.to(device=patched.device, dtype=patched.dtype)
        return with_hidden(out, patched)
    handle = language_layers(model)[layer].register_forward_hook(hook)
    try:
        with torch.inference_mode():
            return model(**inputs, use_cache=False).logits
    finally:
        handle.remove()


def answer_tokens(processor, trial, item_letter, country_letter):
    # This is a forced, format-fixed MC readout.  The exact same prefix is used
    # for gold and named-wrong alternatives, so their log-likelihood difference
    # is the causal estimand rather than free-generation formatting behavior.
    text = f"Item: {item_letter}\nCountry: {country_letter}"
    return processor.tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids


def logprob(model, inputs, layer=None, recipient_span=None, donor_hidden=None, donor_span=None, tokens=None):
    """Teacher-force a short MC answer, optionally patching every forward pass."""
    import torch
    base_ids = inputs["input_ids"]
    full_ids = torch.cat([base_ids, tokens.to(base_ids.device)], dim=1)
    full = dict(inputs)
    full["input_ids"] = full_ids
    # Every per-token tensor must grow with input_ids (qwen3vl's mm_token_type_ids drives M-RoPE;
    # appended answer tokens are text = 0).
    for key, fill in (("attention_mask", 1), ("mm_token_type_ids", 0), ("token_type_ids", 0)):
        if key in full:
            v = full[key]
            full[key] = torch.cat([v, torch.full((1, tokens.shape[1]), fill, device=v.device, dtype=v.dtype)], dim=1)
    if layer is None:
        with torch.inference_mode():
            logits = model(**full, use_cache=False).logits
    else:
        logits = patch_layer(model, full, layer, recipient_span, donor_hidden, donor_span)
    start = base_ids.shape[1]
    # logits at position start+i-1 predicts appended token i.
    lp = torch.log_softmax(logits[0, start - 1:start - 1 + tokens.shape[1]].float(), dim=-1)
    return float(lp.gather(1, tokens[0].to(lp.device).unsqueeze(1)).sum().cpu())


INTERVENTIONS = ("self_to_target_control", "target_clean_to_target", "wrong_clean_to_target")


def wrong_donor_messages(target_clean, wrong):
    """The target's own N=1 MC prompt with the wrong image swapped in, resized to the target's pixel
    size: identical text, and the same image-token grid as the recipient's target span (images of
    different aspect ratios otherwise yield different span lengths and cannot be patched)."""
    from PIL import Image
    msgs = run_models.build_messages(target_clean, "mc")
    imgs = [p for p in msgs[0]["content"] if p["type"] == "image"]
    assert len(imgs) == 1
    size = imgs[0]["image"].size
    imgs[0]["image"] = Image.open(f"images/prepared/{wrong}.jpg").convert("RGB").resize(size, Image.BICUBIC)
    return msgs


def run_cases(model_name, trials_path, cases_path, output, layers):
    import time
    import torch
    trials = {t["trial_id"]: t for t in rows(trials_path)}
    cases = rows(cases_path)
    prev = rows(output) if Path(output).exists() else []
    done = {(r["case_id"], r["layer"], r["intervention"]) for r in prev}
    gate_failed = {r["case_id"] for r in prev if r["intervention"] == "wrong_donor_gate_fail"}
    model, processor = run_models.load(model_name)
    model.eval()
    fam = run_models.MODELS[model_name]["family"]
    n_layers = len(language_layers(model))
    # Only attention layers move information from the image span to the answer positions (LFM's
    # conv layers mix 3 neighbouring tokens), so the last useful patch site is just before the last
    # attention layer: 26 for LFM2.5-VL-3B (attention at 27, then conv 28-29), n-2 for qwen3vl.
    text_cfg = getattr(model.config, "text_config", model.config)
    types = getattr(text_cfg, "layer_types", None) or ["full_attention"] * n_layers
    hi = max(i for i, t in enumerate(types[:n_layers]) if "attention" in t) - 1
    if layers == "auto":
        # qwen3vl adds deepstack visual features to the residual stream after its first
        # len(deepstack_visual_indexes) decoder layers; a patch there is partly overwritten by the
        # recipient image's own features, so start the grid after them (0 for LFM).
        lo = len(getattr(getattr(model.config, "vision_config", None), "deepstack_visual_indexes", None) or [])
        layers = sorted({lo + round(i * (hi - lo) / 4) for i in range(5)})
        print("auto-selected layers:", ",".join(map(str, layers)))
    else:
        layers = [int(x) for x in layers.split(",")]
    if any(x < 0 or x > hi for x in layers):
        raise ValueError(f"layer indices must be in [0, {hi}]: a patch after the last attention layer "
                         f"cannot reach the answer positions (delta == 0 by construction)")
    pool = ch.load_pool()
    for num, case in enumerate(cases, 1):
        cid, t0 = case["case_id"], time.time()
        if cid in gate_failed or all((cid, l, iv) in done for l in layers for iv in INTERVENTIONS):
            continue
        rec, target_clean = trials[case["trial_id"]], trials[case["target_clean_id"]]
        recipient = run_models.prep_inputs(model, processor, fam, rec, "mc")
        target_in = run_models.prep_inputs(model, processor, fam, target_clean, "mc")
        wrong_in = run_models.prep_inputs(model, processor, fam, target_clean, "mc",
                                          messages=wrong_donor_messages(target_clean, case["wrong"]))
        rec_spans, target_spans, wrong_spans = spans_for(recipient, processor), spans_for(target_in, processor), spans_for(wrong_in, processor)
        if len(rec_spans) != rec["N"] or len(target_spans) != 1 or len(wrong_spans) != 1:
            raise RuntimeError(f"{cid}: image-span count mismatch {len(rec_spans)}/{len(target_spans)}/{len(wrong_spans)}")
        target_pos = rec["order"].index(case["target"])
        # Gate: the swapped-in wrong image must be named correctly (this field) when shown alone.
        with torch.inference_mode():
            g = model.generate(**wrong_in, max_new_tokens=run_models.MAX_NEW_TOKENS, do_sample=False)
        gate_raw = processor.batch_decode(g[:, wrong_in["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()
        if ch.parse(gate_raw, "mc", target_clean)[case["field"]] != pool[case["wrong"]][case["field"]]:
            jsonl_append(output, dict(**case, layer=None, intervention="wrong_donor_gate_fail", wrong_donor_raw=gate_raw))
            continue
        gold_i = option_letter(rec["item_options"], pool[case["target"]]["item"])
        gold_c = option_letter(rec["country_options"], pool[case["target"]]["country"])
        wrong_i = gold_i if case["field"] == "country" else case["wrong_letter"]
        wrong_c = gold_c if case["field"] == "item" else case["wrong_letter"]
        gold_tok, wrong_tok = answer_tokens(processor, rec, gold_i, gold_c), answer_tokens(processor, rec, wrong_i, wrong_c)
        # Unpatched baseline is layer-independent: compute once per case.
        base = logprob(model, recipient, tokens=gold_tok) - logprob(model, recipient, tokens=wrong_tok)
        # Clean donor states for every layer from one forward each. Self-to-self is a negative
        # control: same hook and write path, must leave the log-odds unchanged up to noise.
        hid = {"self_to_target_control": (capture_layers(model, recipient, layers), rec_spans[target_pos]),
               "target_clean_to_target": (capture_layers(model, target_in, layers), target_spans[0]),
               "wrong_clean_to_target": (capture_layers(model, wrong_in, layers), wrong_spans[0])}
        for layer in layers:
            for intervention in INTERVENTIONS:
                if (cid, layer, intervention) in done:
                    continue
                donor, donor_span = hid[intervention][0][layer], hid[intervention][1]
                gold_lp = logprob(model, recipient, layer, rec_spans[target_pos], donor, donor_span, gold_tok)
                wrong_lp = logprob(model, recipient, layer, rec_spans[target_pos], donor, donor_span, wrong_tok)
                jsonl_append(output, dict(**case, layer=layer, intervention=intervention,
                                          baseline_logodds=base, patched_logodds=gold_lp - wrong_lp,
                                          delta_logodds=(gold_lp - wrong_lp) - base, wrong_donor_raw=gate_raw))
        print(f"{num}/{len(cases)} cases ({time.time() - t0:.1f}s)", flush=True)
    print("wrote", output)


def summarize(input_path, output):
    import numpy as np
    rs = rows(input_path)
    groups = {}
    for r in rs:
        if r.get("delta_logodds") is None:      # wrong_donor_gate_fail rows
            continue
        groups.setdefault((r["model"], r["layer"], r["intervention"]), []).append(float(r["delta_logodds"]))
    rng = np.random.default_rng(20261017)
    with open(output, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["model", "layer", "intervention", "n", "mean_delta_logodds", "ci_lo", "ci_hi", "rescue_share"])
        w.writeheader()
        for (model, layer, intervention), x in sorted(groups.items()):
            a = np.asarray(x); boot = np.asarray([rng.choice(a, len(a), replace=True).mean() for _ in range(5000)])
            w.writerow(dict(model=model, layer=layer, intervention=intervention, n=len(a),
                            mean_delta_logodds=a.mean(), ci_lo=np.quantile(boot, .025), ci_hi=np.quantile(boot, .975),
                            rescue_share=(a > 0).mean()))
    print(f"wrote {output}; {sum(r['intervention'] == 'wrong_donor_gate_fail' for r in rs)} cases failed the wrong-donor gate")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command", required=True)
    for name in ("select", "run", "summarize"):
        q = sub.add_parser(name); q.add_argument("model")
    s = sub.choices["select"]; s.add_argument("--trials", default="trials_main.jsonl"); s.add_argument("--out"); s.add_argument("--limit", type=int, default=128)
    r = sub.choices["run"]; r.add_argument("--trials", default="trials_main.jsonl"); r.add_argument("--cases"); r.add_argument("--out"); r.add_argument("--layers", default="auto", help="comma-separated indices, or auto (default)")
    z = sub.choices["summarize"]; z.add_argument("--in", dest="input"); z.add_argument("--out")
    a = p.parse_args()
    cases = getattr(a, "cases", None) or f"causal_cases_{a.model}.jsonl"
    if a.command == "select":
        select_cases(a.model, a.trials, a.out or cases, a.limit)
    elif a.command == "run":
        run_cases(a.model, a.trials, cases, a.out or f"causal_{a.model}.jsonl", a.layers)
    else:
        summarize(a.input or f"causal_{a.model}.jsonl", a.out or f"results/causal_{a.model}.csv")


if __name__ == "__main__":
    main()
