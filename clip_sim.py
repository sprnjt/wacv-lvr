#!/usr/bin/env python3
"""CLIP image similarity between each target and its distractor lists (the visual-similarity control for H3).

  python clip_sim.py          -> clip_sim.csv (columns: target, list, mean_sim)

Needs torch + transformers and images/prepared/. Not run here; check it on first use.
mean_sim is the mean over all 4 distractors of a list (an approximation for the N = 2 trials, which use d1 only).
"""
import csv
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
import ch

CLIP = "openai/clip-vit-base-patch32"


def main():
    pool, targets, lists = ch.load_data()
    ids = sorted({i for t, ls in lists.items() for d in ls.values() if d for i in [t] + d})
    model, proc = CLIPModel.from_pretrained(CLIP).eval(), CLIPProcessor.from_pretrained(CLIP)
    feats = {}
    with torch.no_grad():
        for b in range(0, len(ids), 32):
            chunk = ids[b:b + 32]
            ims = [Image.open(f"images/prepared/{i}.jpg").convert("RGB") for i in chunk]
            out = model.get_image_features(**proc(images=ims, return_tensors="pt"))
            out = out if torch.is_tensor(out) else out.pooler_output
            out = torch.nn.functional.normalize(out, dim=-1)
            feats.update(dict(zip(chunk, out)))
    with open("clip_sim.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["target", "list", "mean_sim"])
        for t, ls in lists.items():
            for name, d in ls.items():
                if d:
                    w.writerow([t, name, float(torch.stack([feats[t] @ feats[x] for x in d]).mean())])
    print("wrote clip_sim.csv")


if __name__ == "__main__":
    main()