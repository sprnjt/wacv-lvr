#!/usr/bin/env python3
"""Exploratory activation-patching probe for CultureHaystack.

This deliberately lives outside the frozen main-study pipeline.  It tests a
specific causal prediction: replacing the target image's multi-image hidden
state with the same image's clean N=1 state should increase the likelihood of
the gold MC answer; putting a named wrong image's clean state in the target
slot should do the converse.

Typical use (run the selection once, then resume layer jobs freely):

  python causal_probe.py select lfm25v --limit 128
  env -u LD_LIBRARY_PATH python causal_probe.py run lfm25v --layers 0,4,8,12
  python causal_probe.py summarize lfm25v

The default model is LFM because it is the only model with pre-registered H3a
context attraction.  qwen3vl is supported as a robust-model replication:

  python causal_probe.py select qwen3vl --limit 128
  env -u LD_LIBRARY_PATH python causal_probe.py run qwen3vl --layers 0,8,16,24

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
    """Select error fields with a clean N=1 source for target and wrong image."""
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
            target_clean, wrong_clean = n1.get(tr["target"]), n1.get(wrong)
            if not target_clean or not wrong_clean:
                continue
            # The source image must be recognized when it is alone. This avoids
            # confusing a failed donor representation with a causal rescue.
            if n1_scored.get((target_clean["trial_id"], "mc"), {}).get(f"{field}_ok") != "1":
                continue
            if n1_scored.get((wrong_clean["trial_id"], "mc"), {}).get(f"{field}_ok") != "1":
                continue
            gold = tr["target"]
            gold_letter = option_letter(tr[f"{field}_options"], pool[gold][field])
            wrong_letter = option_letter(tr[f"{field}_options"], pool[wrong][field])
            if not gold_letter or not wrong_letter:
                continue
            candidates.append(dict(case_id=f"{r['trial_id']}::{field}", model=model,
                                   trial_id=r["trial_id"], field=field, target=gold, wrong=wrong,
                                   target_clean_id=target_clean["trial_id"], wrong_clean_id=wrong_clean["trial_id"],
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


def capture_layer(model, inputs, layer):
    """Return post-block states. No gradients are retained."""
    import torch
    captured = []
    def hook(_, __, out):
        captured.append(tensor_from_output(out).detach().cpu())
    handle = language_layers(model)[layer].register_forward_hook(hook)
    try:
        with torch.inference_mode():
            model(**inputs, use_cache=False)
    finally:
        handle.remove()
    if len(captured) != 1:
        raise RuntimeError(f"expected one layer capture, received {len(captured)}")
    return captured[0]


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
    if "attention_mask" in full:
        full["attention_mask"] = torch.cat([full["attention_mask"],
                                             torch.ones((1, tokens.shape[1]), device=base_ids.device,
                                                        dtype=full["attention_mask"].dtype)], dim=1)
    if layer is None:
        with torch.inference_mode():
            logits = model(**full, use_cache=False).logits
    else:
        logits = patch_layer(model, full, layer, recipient_span, donor_hidden, donor_span)
    start = base_ids.shape[1]
    # logits at position start+i-1 predicts appended token i.
    lp = torch.log_softmax(logits[0, start - 1:start - 1 + tokens.shape[1]].float(), dim=-1)
    return float(lp.gather(1, tokens[0].to(lp.device).unsqueeze(1)).sum().cpu())


def run_cases(model_name, trials_path, cases_path, output, layers):
    trials = {t["trial_id"]: t for t in rows(trials_path)}
    cases = rows(cases_path)
    done = {(r["case_id"], r["layer"], r["intervention"])
            for r in rows(output)} if Path(output).exists() else set()
    model, processor = run_models.load(model_name)
    model.eval()
    if layers == "auto":
        n = len(language_layers(model))
        layers = sorted({0, n // 4, n // 2, (3 * n) // 4, n - 1})
        print("auto-selected layers:", ",".join(map(str, layers)))
    else:
        layers = [int(x) for x in layers.split(",")]
    n_layers = len(language_layers(model))
    if any(x < 0 or x >= n_layers for x in layers):
        raise ValueError(f"layer indices must be in [0, {n_layers - 1}]")
    pool = ch.load_pool()
    for num, case in enumerate(cases, 1):
        rec = trials[case["trial_id"]]
        target_clean, wrong_clean = trials[case["target_clean_id"]], trials[case["wrong_clean_id"]]
        recipient = run_models.prep_inputs(model, processor, run_models.MODELS[model_name]["family"], rec, "mc")
        target_in = run_models.prep_inputs(model, processor, run_models.MODELS[model_name]["family"], target_clean, "mc")
        wrong_in = run_models.prep_inputs(model, processor, run_models.MODELS[model_name]["family"], wrong_clean, "mc")
        rec_spans, target_spans, wrong_spans = spans_for(recipient, processor), spans_for(target_in, processor), spans_for(wrong_in, processor)
        target_pos, wrong_pos = rec["order"].index(case["target"]), rec["order"].index(case["wrong"])
        gold_i = option_letter(rec["item_options"], pool[case["target"]]["item"])
        gold_c = option_letter(rec["country_options"], pool[case["target"]]["country"])
        wrong_i = gold_i if case["field"] == "country" else case["wrong_letter"]
        wrong_c = gold_c if case["field"] == "item" else case["wrong_letter"]
        gold_tok, wrong_tok = answer_tokens(processor, rec, gold_i, gold_c), answer_tokens(processor, rec, wrong_i, wrong_c)
        for layer in layers:
            # Capture clean donor states at precisely the layer being replaced.
            target_hidden = capture_layer(model, target_in, layer)
            wrong_hidden = capture_layer(model, wrong_in, layer)
            # Self-to-self is a negative control: it exercises the identical
            # hook and write path but must leave the forced-answer log-odds
            # unchanged up to numerical noise.
            recipient_hidden = capture_layer(model, recipient, layer)
            for intervention, donor, donor_span in (
                ("self_to_target_control", recipient_hidden, rec_spans[target_pos]),
                ("target_clean_to_target", target_hidden, target_spans[0]),
                ("wrong_clean_to_target", wrong_hidden, wrong_spans[0]),
            ):
                key = (case["case_id"], layer, intervention)
                if key in done:
                    continue
                gold_lp = logprob(model, recipient, layer, rec_spans[target_pos], donor, donor_span, gold_tok)
                wrong_lp = logprob(model, recipient, layer, rec_spans[target_pos], donor, donor_span, wrong_tok)
                base_gold = logprob(model, recipient, tokens=gold_tok)
                base_wrong = logprob(model, recipient, tokens=wrong_tok)
                jsonl_append(output, dict(**case, layer=layer, intervention=intervention,
                                          baseline_logodds=base_gold - base_wrong,
                                          patched_logodds=gold_lp - wrong_lp,
                                          delta_logodds=(gold_lp - wrong_lp) - (base_gold - base_wrong)))
        if num % 10 == 0:
            print(f"{num}/{len(cases)} cases", flush=True)
    print("wrote", output)


def summarize(input_path, output):
    import numpy as np
    rs = rows(input_path)
    groups = {}
    for r in rs:
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
    print("wrote", output)


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
