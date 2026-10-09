#!/usr/bin/env python3
"""Label audit: 6 pilot + 24 random main targets vs Wikidata (P495 country of origin / P17 country).

  python3 audit_sample.py            # prints PASS/FAIL per row + summary
Reproducible: sample seed 2026 over sorted target image_ids. Batched API calls (4 total), backoff on 429.
"""
import json, random, time, urllib.parse, urllib.request
import csv

NORM = {
    "Republic of Korea": "South Korea", "Korea": "South Korea", "State of Palestine": "Palestine",
    "Czech Republic": "Czechia", "Russian Empire": "Russia", "Soviet Union": "Russia",
    "Ottoman Empire": "Turkey", "Qing dynasty": "China", "Bohemia": "Czechia",
    "Austria-Hungary": "Austria", "Weimar Republic": "Germany", "Nazi Germany": "Germany",
    "West Germany": "Germany", "Dutch Republic": "Netherlands", "Kingdom of Great Britain": "United Kingdom",
    "England": "United Kingdom", "Great Britain": "United Kingdom", "Burma": "Myanmar",
    "Socialist Federal Republic of Yugoslavia": "Serbia", "Yugoslavia": "Serbia",
    "Persia": "Iran", "Qajar Iran": "Iran", "Pahlavi Iran": "Iran", "Ceylon": "Sri Lanka",
    "People's Republic of China": "China",
}
API = "https://www.wikidata.org/w/api.php"


def wd_get(params, tries=5):
    url = API + "?" + urllib.parse.urlencode({**params, "format": "json"})
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "CultureHaystack-audit/1.0 (benchmark label audit)"})
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code == 429 and i < tries - 1:
                time.sleep(5 * (i + 1))
                continue
            raise
    raise RuntimeError("unreachable")


def entities(qids):
    """Fetch several entities per call, up to 20 ids at a time."""
    out = {}
    for i in range(0, len(qids), 20):
        chunk = qids[i:i + 20]
        res = wd_get({"action": "wbgetentities", "ids": "|".join(chunk),
                      "props": "claims|labels", "languages": "en"})
        out.update(res["entities"])
        time.sleep(1)
    return out


def best_country_claim(ent):
    for prop in ("P495", "P17"):
        claims = ent.get("claims", {}).get(prop, [])
        if not claims:
            continue
        claims.sort(key=lambda c: {"preferred": 0, "normal": 1, "deprecated": 2}.get(
            c.get("rank", "normal"), 1))
        val = claims[0].get("mainsnak", {}).get("datavalue", {}).get("value", {})
        if val.get("id"):
            return val["id"]
    return None


def main():
    targets = list(csv.DictReader(open("data/targets.csv")))
    pool = {r["image_id"]: r for r in csv.DictReader(open("data/pool.csv"))}
    pilot = [t for t in targets if t["split"] == "pilot"]
    main_ids = sorted(t["image_id"] for t in targets if t["split"] == "main")
    sample_ids = set(random.Random(2026).sample(main_ids, 24))
    rows = sorted([t for t in targets if t["split"] == "pilot" or t["image_id"] in sample_ids],
                  key=lambda t: t["image_id"])
    qids = {t["image_id"]: pool[t["image_id"]]["orig_id"].rsplit("_", 1)[-1] for t in rows}
    ents = entities(sorted(set(qids.values())))
    country_qids = {img: best_country_claim(ents[q]) for img, q in qids.items()}
    labels = {q: e.get("labels", {}).get("en", {}).get("value", q)
              for q, e in entities(sorted({q for q in country_qids.values() if q})).items()}
    npass = 0
    out = []
    for t in rows:
        cq = country_qids[t["image_id"]]
        wl = labels.get(cq, "(no claim)") if cq else "(no claim)"
        ok = NORM.get(wl, wl) == t["country"]
        npass += ok
        out.append((t["image_id"], qids[t["image_id"]], pool[t["image_id"]]["item"], wl, t["country"],
                    "PASS" if ok else "FAIL"))
    for r in out:
        print(f"{r[0]}  {r[1]:<14} {r[2]:<38} wikidata={r[3]:<24} target={r[4]:<15} {r[5]}")
    print(f"\n{npass}/{len(rows)} consistent")
    with open("data/label_audit.txt", "w") as f:
        for r in out:
            f.write("  ".join(map(str, r)) + "\n")
        f.write(f"{npass}/{len(rows)} consistent\n")


if __name__ == "__main__":
    main()
