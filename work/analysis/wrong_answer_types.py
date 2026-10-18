#!/usr/bin/env python3
"""What a wrong MC answer looks like at N=5 (post-hoc; feeds Fig. 1c).

  .venv/bin/python work/analysis/wrong_answer_types.py   -> work/analysis/out/wrong_answer_types.csv

Each MC answer field names one shown image's attribute (item_slot / country_slot; at N=5 every option belongs
to a shown image, so "unshown option" cannot occur). For wrong answers (not both fields = target slot):
  half_right   one field from the target, the other from another image
  whole_swap   both fields from the same wrong image
  split        both wrong, from two different wrong images
  other        a field with no valid option (format error)
same_photo = whole_swap / (whole_swap + split): when both fields are wrong, how often they share one image
(chance 1/4: the second field picks one of the 4 non-target images at random).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
MODELS = ["gemma4", "qwen3vl", "qwen25vl", "gemma3_4b", "granite4v", "lfm25v"]
KINDS = ["half_right", "whole_swap", "split", "other"]


def classify(sl, i, c):
    sl, i, c = (np.asarray(x, float) for x in (sl, i, c))
    i, c = np.nan_to_num(i), np.nan_to_num(c)                 # NaN/0 = no valid option
    wrong = ~((i == sl) & (c == sl))
    valid = (i > 0) & (c > 0)
    k = {"whole_swap": wrong & valid & (i == c),
         "half_right": wrong & valid & (i != c) & ((i == sl) | (c == sl)),
         "split": wrong & valid & (i != c) & (i != sl) & (c != sl)}
    k["other"] = wrong & ~valid
    return wrong, k


def table():
    import os
    os.chdir(ROOT)
    import analyze as A
    core, _ = A.core_sets(A.load(MODELS))
    d = core[(core["mode"] == "mc") & (core.N == 5)]
    rows = []
    for m in MODELS + ["pooled"]:
        g = d if m == "pooled" else d[d.model == m]
        wrong, k = classify(g.slot.values, g.item_slot.values, g.country_slot.values)
        n = int(wrong.sum())
        r = {"model": m, "answers": len(g), "wrong": n, **{x: k[x].sum() / n for x in KINDS}}
        r["same_photo"] = k["whole_swap"].sum() / (k["whole_swap"].sum() + k["split"].sum())
        rows.append(r)
    return pd.DataFrame(rows)


def _selftest():
    w, k = classify([3, 3, 3, 3, 3], [3, 3, 4, 4, 0], [3, 4, 4, 1, 2])
    assert list(w) == [False, True, True, True, True]
    assert [int(k[x].sum()) for x in KINDS] == [1, 1, 1, 1]


if __name__ == "__main__":
    _selftest()
    t = table()
    out = ROOT / "work/analysis/out/wrong_answer_types.csv"
    t.round(4).to_csv(out, index=False)
    print(t.round(3).to_string(index=False))
