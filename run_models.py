#!/usr/bin/env python3
"""Run ONE model on a trials file (greedy decoding, one fresh request per trial and answer mode).

  python run_models.py qwen3vl --trials trials_pilot.jsonl
  python run_models.py gemma4  --trials trials_pilot.jsonl
  python run_models.py qwen3vl --trials trials_pilot.jsonl --repeat      # first 10 main trials again
  python run_models.py qwen3vl --trials trials_main.jsonl                # full study (resumes if interrupted)
  python run_models.py qwen3vl --trials trials_main.jsonl --exps E2 --attn   # LVR probe: answer-token
      attention mass per shown image -> attn_<name>.jsonl (also filtered to the given experiments)

Writes raw_<name>.jsonl (or raw_<name>_repeat.jsonl / attn_<name>.jsonl). Written from the two model
cards but NOT run on a GPU here: check the loading and chat-template lines on first use and print one
raw output before a long run. P4 (describe-then-answer) trials get 256 new tokens; all others 64.
"""
import json, sys
from pathlib import Path
import ch

MODELS = {
    "qwen3vl":    dict(hf_id="Qwen/Qwen3-VL-8B-Instruct", family="qwen", arch="Qwen3VLForConditionalGeneration"),
    "qwen25vl":   dict(hf_id="Qwen/Qwen2.5-VL-7B-Instruct", family="qwen", arch="Qwen2_5_VLForConditionalGeneration"),
    "gemma4":     dict(hf_id="google/gemma-4-12B-it", family="gemma"),
    # gemma-3 repos are GATED: run `hf auth login` and accept the license on the model page first
    "gemma3_4b":  dict(hf_id="google/gemma-3-4b-it", family="gemma"),
    "granite4v":  dict(hf_id="ibm-granite/granite-vision-4.1-4b", family="granite"),
    "lfm25v":     dict(hf_id="LiquidAI/LFM2.5-VL-3B", family="lfm", arch="Lfm2VlForConditionalGeneration"),
}
MAX_NEW_TOKENS = 64          # raise only if Gemma's thinking cannot be switched off
TEMPLATE_KWARGS = {"qwen": {}, "gemma": {}, "granite": {}, "lfm": {}}
# Gemma 4's chat template has enable_thinking, default false (verified on the model card): {} is correct.


def load(name):
    import transformers
    from transformers import AutoProcessor
    cfg = MODELS[name]
    processor = AutoProcessor.from_pretrained(cfg["hf_id"])
    cls = getattr(transformers, cfg["arch"]) if cfg.get("arch") else transformers.AutoModelForMultimodalLM
    model = cls.from_pretrained(cfg["hf_id"], dtype="auto", device_map="auto")
    return model, processor


def build_messages(trial, mode, images=True, image_dir="images/prepared"):
    from PIL import Image
    content = []
    for kind, val in ch.make_parts(trial, mode, images):
        if kind == "text":
            content.append({"type": "text", "text": val})
        else:
            content.append({"type": "image", "image": Image.open(f"{image_dir}/{val}.jpg").convert("RGB")})
    return [{"role": "user", "content": content}]


def prep_inputs(model, processor, family, trial, mode, images=True):
    return processor.apply_chat_template(
        build_messages(trial, mode, images), tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt", **TEMPLATE_KWARGS[family]).to(model.device)


def generate(model, processor, family, trial, mode, max_new=MAX_NEW_TOKENS):
    inputs = prep_inputs(model, processor, family, trial, mode, images=(trial["kind"] != "leak"))
    out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False)
    new = out[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(new, skip_special_tokens=True)[0].strip()


# ---------- attention capture (LVR probe) ----------
def image_token_ids(processor):
    """Token ids that stand for image content in the templated input."""
    cands = set()
    v = getattr(processor, "image_token_id", None)
    if v is not None:
        cands.add(v)
    tok = processor.tokenizer
    for s in ("<|image_pad|>", "<image_soft_token>", "<image>"):
        i = tok.convert_tokens_to_ids(s)
        if i is not None and i >= 0 and i != tok.unk_token_id:
            cands.add(i)
    return cands


def image_spans(ids, cands):
    """Contiguous runs of image tokens, one run per image, in prompt order."""
    out, s = [], None
    for i, x in enumerate(ids):
        if x in cands and s is None:
            s = i
        elif x not in cands and s is not None:
            out.append((s, i))
            s = None
    if s is not None:
        out.append((s, len(ids)))
    return out


def generate_attn(model, processor, family, trial, mode, max_new=MAX_NEW_TOKENS):
    """Greedy generation plus answer-token attention mass per shown image (in image order)."""
    import torch
    inputs = prep_inputs(model, processor, family, trial, mode, images=(trial["kind"] != "leak"))
    out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False,
                         output_attentions=True, return_dict_in_generate=True)
    new = out.sequences[:, inputs["input_ids"].shape[1]:]
    raw = processor.batch_decode(new, skip_special_tokens=True)[0].strip()
    ids = inputs["input_ids"][0].tolist()
    spans = image_spans(ids, image_token_ids(processor))
    mass = torch.zeros(len(ids))
    for k, step in enumerate(out.attentions or ()):        # one entry per generated token
        if not step:
            raise RuntimeError("empty attentions: model is not in eager attention mode")
        # newest query row ([..., -1, :]: step 0's prefill matrix row for the first generated token),
        # averaged over heads and layers, prompt positions only. Sliding-window layers (gemma4)
        # return only the trailing window — left-pad to the step's source length so all layers align.
        src = len(ids) + k
        rows = []
        for l in step:
            r = l[0, :, -1, :].float().mean(0)
            if r.numel() < src:
                r = torch.nn.functional.pad(r, (src - r.numel(), 0))
            rows.append(r[: len(ids)])
        mass += torch.stack(rows).mean(0).cpu()
    per_img = [float(mass[s:e].sum()) for s, e in spans]
    return raw, per_img


def main(name, trials_path, repeat=False, attn=False, exps=None, gen=None):
    trials = ch.read_jsonl(trials_path)
    if repeat:
        trials = [t for t in trials if t["kind"] == "main"][:10]
    if exps:
        trials = [t for t in trials if t["exp"] in exps]
    out_path = Path(f"raw_{name}{'_repeat' if repeat else ''}.jsonl") if not attn else Path(f"attn_{name}.jsonl")
    done = {(r["trial_id"], r["mode"]) for r in ch.read_jsonl(out_path)} if out_path.exists() else set()
    if gen is None:
        model, processor = load(name)
        fam = MODELS[name]["family"]
        if attn:
            model.set_attn_implementation("eager")   # sdpa/flash return no attention weights
        gen = (lambda t, mode, mx: generate_attn(model, processor, fam, t, mode, mx)) if attn else \
              (lambda t, mode, mx: generate(model, processor, fam, t, mode, mx))
    with open(out_path, "a", encoding="utf-8") as f:
        for n, t in enumerate(trials):
            mx = 256 if t.get("variant") == "P4" else MAX_NEW_TOKENS
            for mode in ("open", "mc"):
                if (t["trial_id"], mode) in done:
                    continue
                r = gen(t, mode, mx)
                rec = dict(trial_id=t["trial_id"], mode=mode, raw=r[0], attn=r[1]) if attn \
                    else dict(trial_id=t["trial_id"], mode=mode, raw=r)
                f.write(json.dumps(rec) + "\n")
                f.flush()
            if n % 100 == 0:
                print(f"{n}/{len(trials)} trials", flush=True)
    print("wrote", out_path)


if __name__ == "__main__":
    args = sys.argv[1:]
    trials = args[args.index("--trials") + 1] if "--trials" in args else "trials_pilot.jsonl"
    exps = args[args.index("--exps") + 1].split(",") if "--exps" in args else None
    main(args[0], trials, repeat="--repeat" in args, attn="--attn" in args, exps=exps)