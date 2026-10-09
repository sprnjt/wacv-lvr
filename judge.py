#!/usr/bin/env python3
"""LLM judge for open-answer review rows (gemma-4-31B-it, text-only, greedy decoding).

  env -u LD_LIBRARY_PATH python3 judge.py [review_qwen3vl.csv ...]   # default: all review_*.csv

Labels each unresolved review row correct/incorrect and appends to judge_overrides.csv
(schema: model,trial_id,mode,field,label), which `ch.py score` then applies per model.
The judge sees only the gold answer and the model's answer string — never the images and
never which model produced the answer. Every prompt/verdict is logged to
data/judge_log.jsonl for the pre-registration record. Resume-safe: rows already present in
judge_overrides.csv are skipped; unparsed verdicts are logged but not written (rerun retries).
"""
import csv, glob, json, sys
from pathlib import Path

JUDGE_ID = "google/gemma-4-31B-it"   # instruction-tuned; pinned for the pre-registration
FIELD_DESC = {"item": "item (a dish, drink, landmark, instrument, garment or other object)",
              "country": "country of origin"}

PROMPT = """You are checking answers from a visual quiz benchmark. A vision model was shown photographs and asked to name an {field_desc}.

Gold answer: {gold}
Model's answer: {answer}

Does the model's answer name the same {field} as the gold answer?
Allow synonyms, translations, alternate spellings, and well-known equivalents (e.g. "UK" or "Britain" for "United Kingdom", "Holland" for "Netherlands").
Do NOT accept broader or narrower categories (e.g. "bagpipe" for "gaita asturiana", "pasta" for "lasagna"), a different specific item, or a different country — including a historical predecessor of the gold country.

Reply with exactly one word: CORRECT or INCORRECT"""


def load_judge():
    import transformers
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(JUDGE_ID)
    model = transformers.AutoModelForMultimodalLM.from_pretrained(JUDGE_ID, dtype="auto", device_map="auto")
    return model, processor


def verdict(model, processor, field, gold, answer):
    q = PROMPT.format(field_desc=FIELD_DESC[field], field=field, gold=gold, answer=answer)
    msgs = [{"role": "user", "content": [{"type": "text", "text": q}]}]
    inputs = processor.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                           return_dict=True, return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=8, do_sample=False)
    raw = processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()
    word = raw.split()[0].strip(" .").upper() if raw.split() else ""
    return raw, {"CORRECT": "correct", "INCORRECT": "incorrect"}.get(word)


def main(paths):
    ov_path = Path("judge_overrides.csv")
    seen = {(r["model"], r["trial_id"], r["mode"], r["field"])
            for r in csv.DictReader(open(ov_path))} if ov_path.exists() else set()
    rows = []
    for p in paths:
        for r in csv.DictReader(open(p)):
            if (r["model"], r["trial_id"], r["mode"], r["field"]) not in seen:
                rows.append(r)
    print(f"{len(rows)} rows to judge ({len(seen)} already done)")
    if not rows:
        return
    model, processor = load_judge()
    fresh = not ov_path.exists()
    with open(ov_path, "a", newline="") as ovf, open("data/judge_log.jsonl", "a") as logf:
        w = csv.DictWriter(ovf, fieldnames=["model", "trial_id", "mode", "field", "label"])
        if fresh:
            w.writeheader()
        for n, r in enumerate(rows):
            raw, label = verdict(model, processor, r["field"], r["gold"], r["answer"])
            logf.write(json.dumps(dict(model=r["model"], trial_id=r["trial_id"], field=r["field"],
                                       gold=r["gold"], answer=r["answer"], judge=JUDGE_ID,
                                       judge_raw=raw, label=label)) + "\n")
            logf.flush()
            if label:
                w.writerow(dict(model=r["model"], trial_id=r["trial_id"], mode=r["mode"],
                                field=r["field"], label=label))
                ovf.flush()
            else:
                print(f"unparsed verdict, skipped: {raw!r} ({r['trial_id']} {r['field']})")
            if n % 100 == 0:
                print(f"{n}/{len(rows)}", flush=True)
    print(f"done -> {ov_path}")


if __name__ == "__main__":
    main(sys.argv[1:] or sorted(glob.glob("review_*.csv")))
