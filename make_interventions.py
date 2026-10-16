#!/usr/bin/env python3
"""Build post-freeze visual-operation controls without touching trials_main.jsonl.

E7: MC describe-first with answer options placed before the images (P5).
E8: MC tagged-image indexing, replacing ordinal labels with [IMG_k] addresses.
"""
import ch


def main():
    source = ch.read_jsonl("trials_main.jsonl")
    e7, e8 = [], []
    for t in source:
        if t["exp"] == "E6" and t["variant"] == "P4":
            x = dict(t, exp="E7", variant="P5", options_first=True)
            x["trial_id"] = t["trial_id"].replace("E6_", "E7_").removesuffix("_P4") + "_P5"
            e7.append(x)
        if t["exp"] == "E2" and t["variant"] == "P1":
            x = dict(t, exp="E8", indexing="tagged")
            x["trial_id"] = t["trial_id"].replace("E2_", "E8_")
            e8.append(x)
    out = e7 + e8
    assert len(e7) == 280 and len(e8) == 800 and len({t["trial_id"] for t in out}) == len(out)
    ch.write_jsonl("trials_interventions.jsonl", out)
    print(f"wrote {len(out)} trials: E7={len(e7)} E8={len(e8)}")


if __name__ == "__main__":
    main()
