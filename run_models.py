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
import json, re, sys
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
    if name in ("qwen3vl", "qwen25vl"):
        patch_qwen3vl_patch_embed(name)
    model = cls.from_pretrained(cfg["hf_id"], dtype="auto", device_map="auto")
    return model, processor


def patch_qwen3vl_patch_embed(name="qwen3vl"):
    """Qwen3-VL's patch embed is a Conv3d with kernel == stride, i.e. a linear map over each flattened patch.
    On torch 2.9.1 + A40 the bf16 Conv3d path takes ~6-50 s per call (post-freeze RunPod run); the identical
    F.linear is ~25x faster. Same math, bf16 summation order differs; run_all.sh's repro step measures drift."""
    import torch.nn.functional as F
    if name == "qwen25vl":       # same Conv3d patch embed (bias=False); post-freeze E9 extension only
        from transformers.models.qwen2_5_vl import modeling_qwen2_5_vl as M
        cls = M.Qwen2_5_VisionPatchEmbed
    else:
        from transformers.models.qwen3_vl import modeling_qwen3_vl as M
        cls = M.Qwen3VLVisionPatchEmbed

    def forward(self, hidden_states):
        w = self.proj.weight                                   # (embed, C, T, P, P)
        x = hidden_states.to(w.dtype).reshape(-1, w[0].numel())  # (n_patches, C*T*P*P), same order as the view
        return F.linear(x, w.reshape(w.shape[0], -1), self.proj.bias)
    cls.forward = forward


def build_messages(trial, mode, images=True, image_dir="images/prepared"):
    from PIL import Image
    content = []
    for kind, val in ch.make_parts(trial, mode, images):
        if kind == "text":
            content.append({"type": "text", "text": val})
        else:
            # E9 names files under images/ ("collage/<id>.png"); every other trial names a prepared image id
            path = Path(image_dir).parent / val if "/" in val else f"{image_dir}/{val}.jpg"
            content.append({"type": "image", "image": Image.open(path).convert("RGB")})
    return [{"role": "user", "content": content}]


def prep_inputs(model, processor, family, trial, mode, images=True, messages=None):
    return processor.apply_chat_template(
        messages or build_messages(trial, mode, images), tokenize=True, add_generation_prompt=True,
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
    # LFM2-VL tiles large images: <|img_row_R_col_C|> <image>*256 ... <|img_thumbnail|> <image>*k.
    # Count the tile markers as image tokens so one image stays one contiguous span.
    cands |= {i for s, i in tok.get_vocab().items() if re.fullmatch(r"<\|img_(row_\d+_col_\d+|thumbnail)\|>", s)}
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
    # Do not silently write an unusable probe row.  This is especially important
    # for LFM, whose multimodal tokenizer differs from the Qwen/Gemma tokenizers.
    if len(spans) != trial["N"]:
        raise RuntimeError(
            f"expected {trial['N']} image-token spans, found {len(spans)} "
            f"for {trial['trial_id']} ({family})"
        )
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
    if not torch.isfinite(mass).all() or any(x < 0 for x in per_img) or sum(per_img) <= 0:
        raise RuntimeError(f"invalid attention mass for {trial['trial_id']} ({family})")
    return raw, per_img


def main(name, trials_path, repeat=False, attn=False, exps=None, modes=("open", "mc"), out_path=None, gen=None):
    trials = ch.read_jsonl(trials_path)
    if repeat:
        trials = [t for t in trials if t["kind"] == "main"][:10]
    if exps:
        trials = [t for t in trials if t["exp"] in exps]
    if out_path is None and not attn and Path(trials_path).name not in ("trials_main.jsonl", "trials_pilot.jsonl"):
        # would append new trial ids to the frozen raw_<name>.jsonl
        raise SystemExit(f"--out is required for {trials_path} (raw_{name}.jsonl is a frozen main-study file)")
    out_path = Path(out_path) if out_path else (Path(f"raw_{name}{'_repeat' if repeat else ''}.jsonl") if not attn else Path(f"attn_{name}.jsonl"))
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
            mx = 256 if t.get("variant") in ("P4", "P5") else MAX_NEW_TOKENS   # P5 (E7) is describe-first too
            for mode in modes:
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
    modes = tuple(args[args.index("--modes") + 1].split(",")) if "--modes" in args else ("open", "mc")
    if set(modes) - {"open", "mc"}:
        raise ValueError("--modes accepts comma-separated open and/or mc")
    out = args[args.index("--out") + 1] if "--out" in args else None
    main(args[0], trials, repeat="--repeat" in args, attn="--attn" in args, exps=exps, modes=modes, out_path=out)
