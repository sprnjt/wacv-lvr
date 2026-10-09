#!/usr/bin/env python3
"""CultureHaystack pipeline: data -> trials -> scoring.

  python ch.py build-data --meta metadata.csv     filter, pick targets, build distractor lists -> data/
                                                  (optional exclude.txt: metadata ids to drop; then rerun prep and trials)
  python ch.py prep --src <dataset_root>          resize + neutral names -> images/prepared/
  python ch.py trials pilot|main                  -> trials_pilot.jsonl | trials_main.jsonl
  python ch.py score NAME --trials FILE           raw_NAME.jsonl -> scored_NAME.csv + review_NAME.csv
  python ch.py simulate NAME --trials FILE        toy model, to test the analysis before real runs
  python ch.py selftest                           no GPU, no images needed
"""
import argparse, csv, json, random, re, sys, unicodedata
from collections import Counter
from pathlib import Path

SEED = 2026
MIN_SIDE = 384            # drop images whose short side is below this (pixels)
TARGETS_PER_CAT = 40
COUNTRY_CAP = 3           # at most this many targets per country per category (first pass)
PILOT_CATS = ["food", "instrument", "landmark"]
PILOT_PER_CAT = 2
DOMAIN_TO_CAT = {"food": "food", "dessert": "food", "drink": "food", "landmark": "landmark",
                 "instrument": "instrument", "attire": "attire"}      # food, dessert and drink form one category
CATS = {"food": ("foods and drinks", "food or drink"),
        "instrument": ("musical instruments", "musical instrument"),
        "landmark": ("landmarks", "landmark"),
        "attire": ("traditional garments", "traditional garment")}
LETTERS = "ABCDE"
ABSTAIN = {"", "unknown", "n a", "not sure", "unsure", "none"}
DATA = Path("data")

COUNTRY_ALIASES = {
    "Turkey": ["Turkiye"], "Czech Republic": ["Czechia"],
    "United Kingdom": ["UK", "U.K.", "Great Britain", "Britain", "England", "Scotland", "Wales", "Northern Ireland"],
    "United States": ["USA", "U.S.", "US", "U.S.A.", "United States of America", "America"],
    "Ivory Coast": ["Cote d'Ivoire", "Côte d'Ivoire"], "South Korea": ["Korea", "Republic of Korea"],
    "Russia": ["Russian Federation"], "Cape Verde": ["Cabo Verde"], "Myanmar": ["Burma"],
    "Palestine": ["State of Palestine", "Palestinian Territories"], "Taiwan": ["Republic of China"],
    "Iran": ["Islamic Republic of Iran", "Persia"], "Laos": ["Lao PDR"], "Syria": ["Syrian Arab Republic"],
    "Vietnam": ["Viet Nam"], "Netherlands": ["The Netherlands", "Holland"],
    "Bosnia and Herzegovina": ["Bosnia"], "Trinidad and Tobago": ["Trinidad"],
    "Tanzania": ["United Republic of Tanzania"], "Antigua and Barbuda": ["Antigua"],
}


# ---------- text matching ----------
def norm(s):
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"[^a-z0-9]+", " ", s).strip()
    return re.sub(r"^(the|a|an) ", "", s)


def match_len(ans, name, aliases):
    """Length of the best match between an answer and a gold name; 0 if none.
    Equality with the name or an alias counts; the canonical name also counts as whole
    words inside a longer answer. Aliases never match by containment (avoids 'cake' or 'America')."""
    a, n, best = norm(ans), norm(name), 0
    if not a:
        return 0
    if n and (a == n or re.search(rf"\b{re.escape(n)}\b", a)):
        best = len(n)
    for al in aliases:
        x = norm(al)
        if x and a == x:
            best = max(best, len(x))
    return best


def source_image(field, ans, h5, pool):
    """Which image of the 5-image set does this answer name? The target (h5[0]) wins ties."""
    best, who = 0, None
    for i in h5:
        L = match_len(ans, pool[i][field], pool[i][field + "_aliases"])
        if L > best:
            best, who = L, i
    return who


# ---------- data ----------
def clean_aliases(raw, name):
    out = []
    for a in (raw or "").split(";"):
        a = a.strip()
        if ":" in a:
            a = a.split(":", 1)[1].strip()
        if a and norm(a) != norm(name) and a not in out:
            out.append(a)
    return out


def country_aliases(r):
    """Accepted spellings: the dataset's `country_answers` column (lowercase) plus a few extras of ours."""
    canon = norm(r["countries"])
    out = []
    for a in (r.get("country_answers") or "").split(";") + COUNTRY_ALIASES.get(r["countries"], []):
        a = a.strip()
        if a and norm(a) != canon and a not in out:
            out.append(a)
    return out


def load_pool(path=None):
    path = path or DATA / "pool.csv"
    rows = list(csv.DictReader(open(path, encoding="utf-8", newline="")))
    for r in rows:
        r["item_aliases"] = [x for x in r["item_aliases"].split(";") if x]
        r["country_aliases"] = [x for x in r["country_aliases"].split(";") if x]
    return {r["image_id"]: r for r in rows}


def load_data():
    pool = load_pool()
    targets = list(csv.DictReader(open(DATA / "targets.csv", encoding="utf-8", newline="")))
    lists = json.load(open(DATA / "lists.json"))
    return pool, targets, lists


def pick_distractors(pool, t, kind):
    r = pool[t]
    cand = [i for i, x in pool.items()
            if x["category"] == r["category"] and x["country"] != r["country"]
            and (kind == "mixed" or (kind == "near" and x["region"] == r["region"])
                 or (kind == "far" and x["region"] != r["region"]))]
    cand.sort()
    random.Random(f"{SEED}-{t}-{kind}").shuffle(cand)
    out, used = [], set()
    for i in cand:
        c = pool[i]["country"]
        if c not in used:
            out.append(i)
            used.add(c)
        if len(out) == 4:
            return out
    return None


def build_data(meta, out=DATA):
    out.mkdir(exist_ok=True)
    rows = list(csv.DictReader(open(meta, encoding="utf-8-sig", newline="")))
    log = [("rows in metadata", len(rows))]
    rows = [r for r in rows if ";" not in r["countries"]]
    log.append(("single-country rows (shared-heritage rows removed)", len(rows)))
    rows = [r for r in rows if r["domain"] in DOMAIN_TO_CAT]
    log.append(("after keeping the 4 categories", len(rows)))
    if rows and "wikidata_countries" in rows[0]:      # keep only rows where curation and Wikidata agree
        agree = lambda r: {x.strip() for x in r["countries"].split(";")} == {x.strip() for x in r["wikidata_countries"].split(";")}
        disputed = [r for r in rows if not agree(r)]
        rows = [r for r in rows if agree(r)]
        log.append((f"after dropping rows where `countries` differs from `wikidata_countries` ({len(disputed)})", len(rows)))
        (out / "disputed_labels.txt").write_text("\n".join(f"{r['id']}\t{r['name']}\tcurated={r['countries']}\twikidata={r['wikidata_countries']}" for r in disputed) + "\n") if out.exists() else None
    dup = Counter(r["commons_file"] for r in rows if r["commons_file"])
    rows = [r for r in rows if not (r["commons_file"] and dup[r["commons_file"]] > 1)]
    log.append(("after dropping rows that share one Commons file", len(rows)))
    rows = [r for r in rows if min(int(r["width"]), int(r["height"])) >= MIN_SIDE]
    log.append((f"after requiring short side >= {MIN_SIDE}px", len(rows)))
    if Path("exclude.txt").exists():          # one metadata `id` per line: images you rejected by eye
        excl = set(Path("exclude.txt").read_text().split())
        rows = [r for r in rows if r["id"] not in excl]
        log.append((f"after exclude.txt ({len(excl)} ids listed)", len(rows)))
    rows.sort(key=lambda r: r["id"])
    pool = {}
    for n, r in enumerate(rows, 1):
        iid = f"img_{n:04d}"
        pool[iid] = dict(image_id=iid, category=DOMAIN_TO_CAT[r["domain"]], item=r["name"],
                         item_aliases=clean_aliases(r["aliases"], r["name"]), country=r["countries"],
                         country_aliases=country_aliases(r), region=r["regions"],
                         tier=r["tier"], sitelinks=r["sitelinks"], license=r["license"], author=r["author"],
                         source_url=r["source_url"], orig_id=r["id"], file=r["file_name"])
    # near-list feasibility
    def near_ok(i):
        r = pool[i]
        return len({x["country"] for x in pool.values() if x["category"] == r["category"]
                    and x["region"] == r["region"] and x["country"] != r["country"]}) >= 4
    nok = {i: near_ok(i) for i in pool}
    rng = random.Random(SEED)
    # pilot targets first (easy = high popularity tier), kept out of the main targets
    pilot = []
    for cat in PILOT_CATS:
        # the 20 most widely documented (most Wikipedia language editions) items, in random order
        cand = sorted((i for i in pool if pool[i]["category"] == cat and pool[i]["tier"] == "high" and nok[i]),
                      key=lambda i: -int(pool[i]["sitelinks"]))[:20]
        rng.shuffle(cand)
        used, got = set(), []
        for i in cand:
            if pool[i]["country"] not in used:
                got.append(i)
                used.add(pool[i]["country"])
            if len(got) == PILOT_PER_CAT:
                break
        pilot += got
    main = []
    for cat in CATS:
        by = {t: sorted(i for i in pool if pool[i]["category"] == cat and pool[i]["tier"] == t and i not in pilot)
              for t in ("high", "mid", "low")}
        for t in by:
            rng.shuffle(by[t])
        order = [x for grp in zip(*[by[t] for t in ("high", "mid", "low")]) for x in grp]
        order += [x for t in ("high", "mid", "low") for x in by[t][min(len(by[k]) for k in by):]]
        chosen, cnt = [], Counter()
        for cap in (COUNTRY_CAP, 10 ** 6):
            for i in order:
                if len(chosen) == TARGETS_PER_CAT:
                    break
                if i not in chosen and cnt[pool[i]["country"]] < cap:
                    chosen.append(i)
                    cnt[pool[i]["country"]] += 1
        main += chosen
    lists = {t: {k: pick_distractors(pool, t, k) for k in ("mixed", "near", "far")} for t in pilot + main}
    out.mkdir(exist_ok=True)
    cols = ["image_id", "category", "item", "item_aliases", "country", "country_aliases", "region", "tier",
            "sitelinks", "license", "author", "source_url", "orig_id", "file"]
    with open(out / "pool.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in pool.values():
            w.writerow([";".join(r[c]) if isinstance(r[c], list) else r[c] for c in cols])
    with open(out / "targets.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_id", "split", "category", "tier", "region", "country", "near_ok", "idx"])
        for split, ids in (("pilot", pilot), ("main", main)):
            seen = Counter()
            for i in ids:
                r = pool[i]
                w.writerow([i, split, r["category"], r["tier"], r["region"], r["country"],
                            int(lists[i]["near"] is not None), seen[r["category"]]])
                seen[r["category"]] += 1
    json.dump(lists, open(out / "lists.json", "w"))
    # audit
    L = [f"{k}: {v}" for k, v in log] + [""]
    for cat in CATS:
        g = [r for r in pool.values() if r["category"] == cat]
        tg = [i for i in main if pool[i]["category"] == cat]
        L.append(f"{cat}: {len(g)} images, {len({r['country'] for r in g})} countries, "
                 f"tiers {dict(Counter(r['tier'] for r in g))}; {len(tg)} main targets "
                 f"(tiers {dict(Counter(pool[i]['tier'] for i in tg))}, "
                 f"{len({pool[i]['country'] for i in tg})} countries), "
                 f"near list possible for {sum(lists[i]['near'] is not None for i in tg)}/{len(tg)}")
    L += ["", "pilot targets: " + ", ".join(f"{i} ({pool[i]['item']}, {pool[i]['country']})" for i in pilot),
          "licenses in pool: " + str(dict(Counter(r["license"] for r in pool.values()).most_common(6))),
          f"public domain or CC0: {sum(r['license'] in ('Public domain', 'CC0') for r in pool.values())}",
          f"missing author: {sum(not r['author'] for r in pool.values())} (use source_url for attribution)"]
    (out / "audit.txt").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def prep(src, out="images/prepared"):
    from PIL import Image
    pool = load_pool()
    Path(out).mkdir(parents=True, exist_ok=True)
    missing = 0
    for r in pool.values():
        p = Path(src) / r["file"]
        if not p.exists():
            missing += 1
            print("missing:", p)
            continue
        im = Image.open(p)
        if im.mode in ("RGBA", "LA", "P"):
            im = im.convert("RGBA")
            bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
            bg.alpha_composite(im)
            im = bg
        im = im.convert("RGB")
        im.thumbnail((768, 768))
        im.save(f"{out}/{r['image_id']}.jpg", quality=90)      # re-saving strips EXIF
    print(f"prepared {len(pool) - missing} images, {missing} missing")


# ---------- trials ----------
def mk(pool, lists, exp, t, lst, N, slot, variant="P1", kind="main"):
    d = lists[t][lst]
    h5 = [t] + d
    if kind == "leak":
        order, N, slot = h5, 5, 1
    else:
        seq = d[:N - 1]
        order = seq[:slot - 1] + [t] + seq[slot - 1:]
    rng = random.Random(f"{SEED}-opt-{t}-{lst}")
    items = [pool[i]["item"] for i in h5]
    rng.shuffle(items)
    countries = [pool[i]["country"] for i in h5]
    rng.shuffle(countries)
    tid = f"{exp}_{t}_{lst}_leak" if kind == "leak" else f"{exp}_{t}_{lst}_N{N}_k{slot}_{variant}"
    r = pool[t]
    return dict(trial_id=tid, exp=exp, kind=kind, target=t, category=r["category"], tier=r["tier"],
                list=lst, N=N, slot=slot, variant=variant, order=order, h5=h5,
                item_options=items, country_options=countries)


def pilot_trials(pool, targets, lists):
    out = []
    for t in (x["image_id"] for x in targets if x["split"] == "pilot"):
        for N in range(1, 6):
            for k in range(1, N + 1):
                out.append(mk(pool, lists, "P0", t, "mixed", N, k))
        out.append(mk(pool, lists, "LEAK", t, "mixed", 5, 1, kind="leak"))
    return out


def main_trials(pool, targets, lists):
    out = []
    for x in (x for x in targets if x["split"] == "main"):
        t, idx = x["image_id"], int(x["idx"])
        for N in range(1, 5):                                   # E1: N = 1..4, balanced slot
            out.append(mk(pool, lists, "E1", t, "mixed", N, idx % N + 1))
        for k in range(1, 6):                                   # E2: N = 5, every slot (also E1's N = 5 cell)
            out.append(mk(pool, lists, "E2", t, "mixed", 5, k))
        out.append(mk(pool, lists, "LEAK", t, "mixed", 5, 1, kind="leak"))
        if x["near_ok"] == "1":                                 # E3: near vs far
            for lst in ("near", "far"):
                for N in (2, 5):
                    out.append(mk(pool, lists, "E3", t, lst, N, idx % N + 1))
        if idx % 3 == 0:                                        # E5: paraphrases on every third target
            for v in ("P2", "P3"):
                for N in range(1, 6):
                    out.append(mk(pool, lists, "E5", t, "mixed", N, idx % N + 1, variant=v))
        if idx % 3 == 0:                                        # E6: describe-then-answer (LVR text-mediation probe)
            for N in range(1, 6):
                out.append(mk(pool, lists, "E6", t, "mixed", N, idx % N + 1, variant="P4"))
    return out


def write_jsonl(path, rows):
    Path(path).write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def read_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


# ---------- prompts ----------
def make_parts(tr, mode, images=True):
    """List of ('text', str) and ('image', image_id) parts, in the order they are sent."""
    plural, singular = CATS[tr["category"]]
    n, k, v = tr["N"], tr["slot"], tr.get("variant", "P1")
    if n == 1:
        head = f"Below is 1 photograph of a {singular}.\nEach photograph is labeled with its number.\n"
    else:
        head = f"Below are {n} photographs of {plural} from various countries.\nEach photograph is labeled with its number.\n"
    parts = [("text", head)]
    for i, img in enumerate(tr["order"], 1):
        parts.append(("text", f"\nImage {i}:"))
        if images:
            parts.append(("image", img))
    if mode == "open":
        ask = {"P1": f"Look only at Image {k} and answer two questions.\n"
                     f"1. What is the name of the {singular} shown in Image {k}?\n2. Which country is it from?",
               "P2": f"Focus on Image {k}. Tell me the name of the {singular} and the country it comes from.",
               "P3": f"Image {k} is the one I care about. What is it called, and where is it from?",
               "P4": f"First describe each photograph in one short line, in order (Image 1: ..., Image 2: ...). "
                     f"Then look only at Image {k} and answer two questions.\n"
                     f"1. What is the name of the {singular} shown in Image {k}?\n2. Which country is it from?"}[v]
        tail = ("Answer in exactly this format and nothing else:\nItem: <name>\nCountry: <country>\n"
                "If you cannot tell, write Unknown for that field.") if v != "P4" else (
                "End your reply with exactly these two lines and nothing after them:\nItem: <name>\nCountry: <country>\n"
                "If you cannot tell, write Unknown for that field.")
        q = "\n\n" + ask + "\n\n" + tail
    else:
        intro = {"P1": f"Look only at Image {k} and answer two multiple-choice questions.",
                 "P2": f"Focus on Image {k} and answer the two multiple-choice questions about it.",
                 "P3": f"Image {k} is the one I care about. Answer the two multiple-choice questions about it.",
                 "P4": f"First describe each photograph in one short line, in order. "
                       f"Then look only at Image {k} and answer two multiple-choice questions about it."}[v]
        io = "\n".join(f"{a}. {x}" for a, x in zip(LETTERS, tr["item_options"]))
        co = "\n".join(f"{a}. {x}" for a, x in zip(LETTERS, tr["country_options"]))
        tail = ("Answer in exactly this format and nothing else:\nItem: <letter>\nCountry: <letter>") if v != "P4" else (
                "End your reply with exactly these two lines and nothing after them:\nItem: <letter>\nCountry: <letter>")
        q = (f"\n\n{intro}\n\nQuestion 1. What is the name of the {singular} shown in Image {k}?\n{io}\n\n"
             f"Question 2. Which country is it from?\n{co}\n\n{tail}")
    parts.append(("text", q))
    return parts


# ---------- scoring ----------
def parse(raw, mode, tr):
    out = {}
    for field, key, opts in (("item", "Item", tr["item_options"]), ("country", "Country", tr["country_options"])):
        m = re.search(rf"{key}\s*:\s*(.+)", raw or "", re.I)
        if not m:
            out[field] = None
            continue
        val = m.group(1).strip().strip("*_`<> ")
        if mode == "mc":
            L = re.match(r"\(?([A-Ea-e])\b", val)
            val = opts[LETTERS.index(L.group(1).upper())] if L else None
        out[field] = val
    return out


def score_trial(tr, mode, raw, pool, overrides=None):
    overrides = overrides or {}
    t, order, h5 = tr["target"], tr["order"], tr["h5"]
    shown = set(order) if tr["kind"] == "main" else set()
    ans, res = parse(raw, mode, tr), {}
    for f in ("item", "country"):
        a = ans[f]
        if a is None:
            src, err = None, "format_error"
        elif norm(a) in ABSTAIN:
            src, err = None, "abstain"
        else:
            src = source_image(f, a, h5, pool)
            err = None if src == t else ("mis_binding" if src in shown else "prior_drift")
        ov = overrides.get((tr["trial_id"], mode, f))
        if ov == "correct":
            src, err = t, None
        elif ov == "incorrect" and err is None:
            src, err = None, "prior_drift"
        res[f"{f}_ans"], res[f"{f}_src"], res[f"{f}_err"] = a, src, err
        res[f"{f}_ok"] = int(err is None)
        res[f"{f}_slot"] = order.index(src) + 1 if src in shown else 0
    res["joint_ok"] = int(res["item_ok"] and res["country_ok"])
    i, c = res["item_src"], res["country_src"]
    res["illusory"] = int(not res["joint_ok"] and i in shown and c in shown and i != c)
    return res


def read_overrides(path="judge_overrides.csv", model=None):
    if not Path(path).exists():
        return {}
    return {(r["trial_id"], r["mode"], r["field"]): r["label"] for r in csv.DictReader(open(path, encoding="utf-8"))
            if r.get("label") in ("correct", "incorrect") and r.get("model") == model}


def score_file(name, trials_path):
    pool = load_pool()
    trials = {t["trial_id"]: t for t in read_jsonl(trials_path)}
    ov = read_overrides(model=name)
    rows, review = [], []
    for r in read_jsonl(f"raw_{name}.jsonl"):
        tr = trials[r["trial_id"]]
        s = score_trial(tr, r["mode"], r["raw"], pool, ov)
        rows.append(dict(model=name, mode=r["mode"], trial_id=tr["trial_id"], exp=tr["exp"], kind=tr["kind"],
                         target=tr["target"], category=tr["category"], tier=tr["tier"], list=tr["list"],
                         N=tr["N"], slot=tr["slot"], variant=tr["variant"], **s, raw=r["raw"]))
        if r["mode"] == "open":
            for f in ("item", "country"):
                if s[f"{f}_err"] == "prior_drift" and s[f"{f}_src"] is None:
                    review.append(dict(model=name, trial_id=tr["trial_id"], mode="open", field=f,
                                       answer=s[f"{f}_ans"], gold=pool[tr["target"]][f], label=""))
    with open(f"scored_{name}.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(f"review_{name}.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "trial_id", "mode", "field", "answer", "gold", "label"])
        w.writeheader()
        w.writerows(review)
    return rows, review


def console_report(name, rows, repeat_path=None):
    print(f"\n=== {name} ===")
    for mode in ("open", "mc"):
        main = [r for r in rows if r["mode"] == mode and r["kind"] == "main"]
        if not main:
            continue
        print(f"\n[{mode}] joint accuracy by N (and item / country)")
        for n in range(1, 6):
            g = [r for r in main if r["N"] == n]
            if g:
                print(f"  N={n}: joint {sum(r['joint_ok'] for r in g)}/{len(g)}   "
                      f"item {sum(r['item_ok'] for r in g)}/{len(g)}   country {sum(r['country_ok'] for r in g)}/{len(g)}")
        g5 = [r for r in main if r["N"] == 5]
        print(f"[{mode}] joint accuracy by slot at N=5:",
              {k: f"{sum(r['joint_ok'] for r in g5 if r['slot'] == k)}/{sum(1 for r in g5 if r['slot'] == k)}"
               for k in range(1, 6) if any(r['slot'] == k for r in g5)})
        multi = [r for r in main if r["N"] >= 2]
        errs = [r[f"{f}_err"] for r in multi for f in ("item", "country") if r[f"{f}_err"]]
        print(f"[{mode}] wrong fields at N>=2:", {e: errs.count(e) for e in ("mis_binding", "prior_drift", "abstain", "format_error")})
        wj = [r for r in multi if not r["joint_ok"]]
        print(f"[{mode}] illusory conjunctions: {sum(r['illusory'] for r in wj)} of {len(wj)} wrong joint answers")
        print(f"[{mode}] format errors:", sum(1 for r in main for f in ('item', 'country') if r[f'{f}_err'] == 'format_error'))
        leak = [r for r in rows if r["mode"] == mode and r["kind"] == "leak"]
        if leak:
            print(f"[{mode}] text-only leak check: item {sum(r['item_ok'] for r in leak)}/{len(leak)}, "
                  f"country {sum(r['country_ok'] for r in leak)}/{len(leak)}")
    if repeat_path and Path(repeat_path).exists():
        a = {(r["trial_id"], r["mode"]): r["raw"] for r in read_jsonl(f"raw_{name}.jsonl")}
        b = read_jsonl(repeat_path)
        print(f"repeat check: {sum(a.get((r['trial_id'], r['mode'])) == r['raw'] for r in b)}/{len(b)} identical")


# ---------- fake / simulated models ----------
def fake_raw(tr, mode, pool, kind, rng=None):
    t = pool[tr["target"]]
    first = pool[tr["order"][0]]
    absent = next((pool[i] for i in tr["h5"] if i not in tr["order"]), t)
    it, co = {"oracle": (t, t), "first": (first, first), "mixed": (first, t), "absent": (absent, absent)}[kind]
    if mode == "open":
        return f"Item: {it['item']}\nCountry: {co['country']}"
    return (f"Item: {LETTERS[tr['item_options'].index(it['item'])]}\n"
            f"Country: {LETTERS[tr['country_options'].index(co['country'])]}")


def simulate(name, trials_path, pool):
    """Toy model: accuracy falls with N, is worse in middle slots and with near distractors."""
    out = []
    base = {1: 0.95, 2: 0.72, 3: 0.66, 4: 0.62, 5: 0.58}
    for tr in read_jsonl(trials_path):
        for mode in ("open", "mc"):
            rng = random.Random(f"sim-{name}-{tr['trial_id']}-{mode}")
            N = tr["N"]
            p = base[N] - (0.06 if tr["list"] == "near" else 0) - (0.05 if 1 < tr["slot"] < N else 0)
            shown = [i for i in tr["order"] if i != tr["target"]]
            absent = [i for i in tr["h5"] if i not in tr["order"]]
            ans = {}
            for f in ("item", "country"):
                if tr["kind"] == "leak" or rng.random() > p:
                    if shown and rng.random() < 0.7 and tr["kind"] != "leak":
                        ans[f] = rng.choice(shown)
                    else:
                        ans[f] = rng.choice(absent or tr["h5"][1:])
                else:
                    ans[f] = tr["target"]
            ii, cc = pool[ans["item"]], pool[ans["country"]]
            if mode == "open":
                raw = f"Item: {ii['item']}\nCountry: {cc['country']}"
            else:
                raw = (f"Item: {LETTERS[tr['item_options'].index(ii['item'])]}\n"
                       f"Country: {LETTERS[tr['country_options'].index(cc['country'])]}")
            out.append(dict(trial_id=tr["trial_id"], mode=mode, raw=raw))
    write_jsonl(f"raw_{name}.jsonl", out)
    print("wrote", f"raw_{name}.jsonl", len(out), "rows")


# ---------- self test ----------
def selftest(meta="metadata.csv"):
    import tempfile, os
    cwd = os.getcwd()
    tmp = tempfile.mkdtemp()
    os.chdir(tmp)
    try:
        build_data(Path(cwd) / meta if not Path(meta).is_absolute() else meta)
        pool, targets, lists = load_data()
        P = pilot_trials(pool, targets, lists)
        M = main_trials(pool, targets, lists)
        print("\ntrials: pilot", len(P), " main", len(M), Counter(t["exp"] for t in M))
        by_key = {}
        for tr in P + M:
            h5, t = tr["h5"], tr["target"]
            assert h5[0] == t and len(set(h5)) == 5
            assert len({pool[i]["country"] for i in h5}) == 5 and len({pool[i]["item"] for i in h5}) == 5
            assert all(pool[i]["category"] == tr["category"] for i in h5)
            if tr["kind"] == "main":
                assert tr["order"][tr["slot"] - 1] == t and len(tr["order"]) == tr["N"]
                assert set(tr["order"]) <= set(h5)
            by_key.setdefault((t, tr["list"]), set()).add((tuple(tr["item_options"]), tuple(tr["country_options"])))
            if tr["list"] == "near":
                assert all(pool[i]["region"] == pool[t]["region"] for i in h5)
            if tr["list"] == "far":
                assert all(pool[i]["region"] != pool[t]["region"] for i in h5[1:])
        assert all(len(v) == 1 for v in by_key.values()), "options must be fixed per (target, list)"
        pm = Counter((t["category"], t["slot"]) for t in M if t["exp"] == "E1" and t["N"] == 4)
        assert max(pm.values()) - min(pm.values()) <= 4, pm
        ids_p = {x["image_id"] for x in targets if x["split"] == "pilot"}
        ids_m = {x["image_id"] for x in targets if x["split"] == "main"}
        assert not ids_p & ids_m
        for kind in ("oracle", "first", "mixed", "absent"):
            for trials in (P, M):
                rows = []
                for tr in trials:
                    for mode in ("open", "mc"):
                        s = score_trial(tr, mode, fake_raw(tr, mode, pool, kind), pool)
                        rows.append(dict(tr=tr, **s))
                main = [r for r in rows if r["tr"]["kind"] == "main"]
                if kind == "oracle":
                    assert all(r["joint_ok"] for r in main)
                if kind == "first":
                    assert all(r["joint_ok"] == int(r["tr"]["slot"] == 1) for r in main)
                    assert not any(r["illusory"] for r in main)
                if kind == "mixed":
                    assert all(r["illusory"] for r in main if r["tr"]["slot"] != 1)
                if kind == "absent":
                    assert all(r["item_err"] == "prior_drift" for r in main if r["tr"]["N"] < 5)
        txt = "".join(v if k == "text" else f" <{v}>" for k, v in make_parts(P[40], "mc"))
        assert "Look only at Image" in txt and "Question 2." in txt
        p4 = [t for t in M if t["exp"] == "E6"]
        assert all(t["variant"] == "P4" for t in p4)
        assert len(p4) == 5 * sum(1 for x in targets if x["split"] == "main" and int(x["idx"]) % 3 == 0)
        t4 = "".join(v if k == "text" else "" for k, v in make_parts(p4[0], "open"))
        assert "describe" in t4.lower() and "Item:" in t4
        ans = parse("Image 1: a stew\nImage 2: dumplings\nItem: Baklava\nCountry: Turkey", "open", p4[0])
        assert ans["item"] == "Baklava" and ans["country"] == "Turkey", ans
        print("SELFTEST OK")
    finally:
        os.chdir(cwd)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("arg", nargs="?")
    ap.add_argument("--meta", default="metadata.csv")
    ap.add_argument("--src", default=".")
    ap.add_argument("--trials", default="trials_main.jsonl")
    a = ap.parse_args()
    if a.cmd == "build-data":
        build_data(a.meta)
    elif a.cmd == "prep":
        prep(a.src)
    elif a.cmd == "trials":
        pool, targets, lists = load_data()
        tr = pilot_trials(pool, targets, lists) if a.arg == "pilot" else main_trials(pool, targets, lists)
        write_jsonl(f"trials_{a.arg}.jsonl", tr)
        print(len(tr), "trials;", dict(Counter(t["exp"] for t in tr)))
    elif a.cmd == "score":
        rows, review = score_file(a.arg, a.trials)
        console_report(a.arg, rows, f"raw_{a.arg}_repeat.jsonl")
        print(f"\nwrote scored_{a.arg}.csv; {len(review)} open answers in review_{a.arg}.csv need a hand/LLM-judge label")
    elif a.cmd == "simulate":
        simulate(a.arg, a.trials, load_pool())
    elif a.cmd == "selftest":
        selftest(a.meta)
    else:
        print(__doc__)