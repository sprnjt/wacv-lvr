#!/usr/bin/env python3
"""E9 collage (post-freeze, exploratory, MC only): does binding fail across separate images or within one scene?

  python3 make_collage.py trials     -> trials_collage.jsonl (160 main targets x 4 conditions = 640 trials)
  python3 make_collage.py images     -> images/collage/E9_<target>_{grid,solo,p1..p4}.png (on the pod, after prep)

Per target: 3 companions from data/pool.csv, one from each other category, 4 distinct countries, never the
item of any (pilot or main) target; seed 2026. 2x2 collage, 768x768, white; each panel is the prepared image
letterboxed into a 376x376 cell; 8 px light-grey gutter between cells (4 px white outer margin). The target
slot is idx % 4 in reading order (top-left, top-right, bottom-left, bottom-right); C2 shows the same order.
  C1 solo          the canvas with only the target panel (other cells blank)   "the top-left panel"
  C2 separate      the 4 letterboxed panels as 4 images, same pixels            "Image k"
  C3 collage_pos   the collage                                                 "the top-left panel"
  C4 collage_kind  the collage                                                 "the panel showing the <kind>"
Options are fixed per target (same in C1-C4): the 4 shown items + 1 unshown decoy item of the target's
category, and the 4 shown countries + the decoy's (unshown) country. The gold item letter differs from the
gold country letter. h5 = [target, companions..., decoy], so ch.score_trial / ch.parse work unchanged.
Question 1 says "item" instead of the category in all conditions, so only C4 tells the model the kind.
slot = grid position of the target; C1 shows only the target, so its order is [target].
Images are PNG (lossless): a C2 panel has exactly the pixels of its collage cell.
"""
import random, sys
from collections import Counter
from pathlib import Path
import ch

CELL, GUT, SIZE = 376, 8, 768
MARGIN = (SIZE - 2 * CELL - GUT) // 2                                         # 4 px
XY = [(MARGIN + c * (CELL + GUT), MARGIN + r * (CELL + GUT)) for r in (0, 1) for c in (0, 1)]
GREY = (211, 211, 211)
POS = ["top-left", "top-right", "bottom-left", "bottom-right"]
CONDS = ("solo", "separate", "collage_pos", "collage_kind")
TRIALS, OUT = "trials_collage.jsonl", Path("images/collage")
HEAD = {"solo": "Below is 1 image: a 2x2 grid of panels (top-left, top-right, bottom-left, bottom-right). "
                "Only one panel contains a photograph; the others are blank.\n",
        "separate": "Below are 4 photographs from various countries.\nEach photograph is labeled with its number.\n",
        "collage": "Below is 1 image: a 2x2 grid of four photographs from various countries, "
                   "in its top-left, top-right, bottom-left and bottom-right panels.\n"}


def trials():
    pool, targets, _ = ch.load_data()
    banned = {ch.norm(pool[x["image_id"]]["item"]) for x in targets}
    out = []
    for x in (x for x in targets if x["split"] == "main"):
        t, cat, s = x["image_id"], x["category"], int(x["idx"]) % 4
        rng = random.Random(f"{ch.SEED}-e9-{t}")
        used, comp = {pool[t]["country"]}, []
        for c in [c for c in ch.CATS if c != cat] + [cat]:        # 3 companions, then the decoy
            cand = sorted(i for i, r in pool.items() if r["category"] == c and ch.norm(r["item"]) not in banned)
            rng.shuffle(cand)
            i = next(i for i in cand if pool[i]["country"] not in used)
            comp.append(i)
            used.add(pool[i]["country"])
        decoy, comp = comp[3], comp[:3]
        rng.shuffle(comp)
        panels = comp[:s] + [t] + comp[s:]
        h5 = [t] + comp + [decoy]
        items, countries = [pool[i]["item"] for i in h5], [pool[i]["country"] for i in h5]
        ro = random.Random(f"{ch.SEED}-e9-opt-{t}")
        while True:            # gemma3_4b copies its item letter into the country field: gold letters must differ
            ro.shuffle(items)
            ro.shuffle(countries)
            if items.index(pool[t]["item"]) != countries.index(pool[t]["country"]):
                break
        base = dict(exp="E9", kind="main", target=t, category=cat, tier=pool[t]["tier"], list="collage",
                    variant="P1", slot=s + 1, seed=ch.SEED, panels=panels, decoy=decoy,
                    panel_categories=[pool[i]["category"] for i in panels],
                    panel_regions=[pool[i]["region"] for i in panels],
                    h5=h5, item_options=items, country_options=countries, noun="item")
        grid, kind_ref, pos_ref = f"collage/E9_{t}_grid.png", f"the panel showing the {ch.CATS[cat][1]}", f"the {POS[s]} panel"
        spec = {"solo": ([f"collage/E9_{t}_solo.png"], [t], HEAD["solo"], pos_ref),
                "separate": ([f"collage/E9_{t}_p{j}.png" for j in range(1, 5)], panels, HEAD["separate"], f"Image {s + 1}"),
                "collage_pos": ([grid], panels, HEAD["collage"], pos_ref),
                "collage_kind": ([grid], panels, HEAD["collage"], kind_ref)}
        for cond in CONDS:
            imgs, order, head, ref = spec[cond]
            out.append(dict(trial_id=f"E9_{t}_{cond}", condition=cond, N=len(imgs), order=order, images=imgs,
                            head=head, ref=ref, **base))
    check(out, pool, banned)
    ch.write_jsonl(TRIALS, out)
    print(f"wrote {len(out)} trials to {TRIALS}:", dict(Counter(t["condition"] for t in out)))


def check(out, pool, banned):
    assert len(out) == 640 and len({t["trial_id"] for t in out}) == 640, len(out)
    by = {}
    for t in out:
        by.setdefault(t["target"], []).append(t)
    assert len(by) == 160 and all(len(v) == 4 for v in by.values())
    for tg, g in by.items():
        x = g[0]
        assert all(y[k] == x[k] for y in g for k in ("panels", "h5", "item_options", "country_options", "slot"))
        p, h5 = x["panels"], x["h5"]
        assert p[x["slot"] - 1] == tg and sorted(p) == sorted(h5[:4]) and h5[0] == tg and x["decoy"] == h5[4]
        assert sorted(pool[i]["category"] for i in p) == sorted(ch.CATS) and pool[h5[4]]["category"] == x["category"]
        assert len({pool[i]["country"] for i in h5}) == 5 and len({ch.norm(pool[i]["item"]) for i in h5}) == 5
        assert not any(ch.norm(pool[i]["item"]) in banned for i in h5[1:])
        for f in ("item", "country"):      # every option string scores as its own image
            assert all(ch.source_image(f, o, h5, pool) == next(i for i in h5 if pool[i][f] == o) for o in x[f"{f}_options"])
        assert x["item_options"].index(pool[tg]["item"]) != x["country_options"].index(pool[tg]["country"])
        for y in g:
            assert y["N"] == (4 if y["condition"] == "separate" else 1) and y["order"] == ([tg] if y["condition"] == "solo" else p)
    sl = Counter((t["category"], t["slot"]) for t in out if t["condition"] == "solo")
    assert set(sl.values()) == {10}, sl                        # 40 targets per category -> 10 per slot


def canvas(cells):
    from PIL import Image, ImageDraw
    im = Image.new("RGB", (SIZE, SIZE), "white")
    d, a = ImageDraw.Draw(im), MARGIN + CELL
    d.rectangle([a, 0, a + GUT - 1, SIZE - 1], fill=GREY)
    d.rectangle([0, a, SIZE - 1, a + GUT - 1], fill=GREY)
    for c, xy in zip(cells, XY):
        if c is not None:
            im.paste(c, xy)
    return im


def images(src="images/prepared"):
    from PIL import Image, ImageOps
    T = ch.read_jsonl(TRIALS)
    need = sorted({i for t in T for i in t["panels"]})
    missing = [i for i in need if not Path(f"{src}/{i}.jpg").exists()]
    assert not missing, f"{len(missing)} prepared images missing, e.g. {missing[:3]} (run ch.py prep first)"
    OUT.mkdir(parents=True, exist_ok=True)
    for t in {t["target"]: t for t in T}.values():
        cells = [ImageOps.pad(Image.open(f"{src}/{i}.jpg").convert("RGB"), (CELL, CELL),
                              method=Image.Resampling.LANCZOS, color="white") for i in t["panels"]]
        for j, c in enumerate(cells, 1):
            c.save(OUT / f"E9_{t['target']}_p{j}.png")
        canvas(cells).save(OUT / f"E9_{t['target']}_grid.png")
        canvas([c if j == t["slot"] - 1 else None for j, c in enumerate(cells)]).save(OUT / f"E9_{t['target']}_solo.png")
    refs = {p for t in T for p in t["images"]}
    missing = [p for p in refs if not (OUT.parent / p).exists()]
    assert not missing and len(refs) == 6 * 160, (len(refs), missing[:3])
    print(f"built {len(refs)} collage images in {OUT}")


if __name__ == "__main__":
    {"trials": trials, "images": images}[sys.argv[1]]()
