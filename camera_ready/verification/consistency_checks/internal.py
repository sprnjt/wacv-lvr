import json, glob, os, collections
import pandas as pd
H = "work/cc/hf"; os.chdir(H)
issues = []; ok = []
def rj(p): return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
def check(name, cond, detail=""):
    (ok if cond else issues).append(f"{name}{' — ' + detail if detail else ''}")

pool = pd.read_csv("items/main_study/pool.csv"); ids = set(pool.image_id)
check("pool: unique image ids", pool.image_id.is_unique, f"{len(pool)} rows")
T = {"main": rj("trials/main_study/trials_main.jsonl"), "pilot": rj("trials/main_study/trials_pilot.jsonl"),
     "collage": rj("trials/probes/trials_collage.jsonl"), "interv": rj("trials/probes/trials_interventions.jsonl"),
     "full": rj("trials/full_pool/trials_full.jsonl"), "g1": rj("trials/full_pool/trials_g1.jsonl"),
     "g2_160": rj("trials/full_pool/trials_g2_160.jsonl"), "g2_full": rj("trials/full_pool/trials_g2_full.jsonl")}
for k, ts in T.items():
    tid = [t["trial_id"] for t in ts]
    check(f"trials {k}: unique ids", len(tid) == len(set(tid)), f"{len(tid)} trials")
    used = {i for t in ts for i in t.get("order", []) + [t.get("target")] if isinstance(i, str) and i.startswith("img_")}
    check(f"trials {k}: every image id in pool.csv", used <= ids, f"{len(used)} images; unknown {sorted(used - ids)[:3]}")
    bad = [t["trial_id"] for t in ts if "order" in t and t.get("N") and t.get("exp") != "LEAK" and len(t["order"]) != t["N"]]
    check(f"trials {k}: N equals images shown", not bad, f"{len(bad)} bad")
    bad = [t["trial_id"] for t in ts if t.get("order") and t.get("slot") and t.get("exp") != "LEAK" and (len(t["order"]) < t["slot"] or t["order"][t["slot"] - 1] != t["target"])]
    check(f"trials {k}: target sits at its slot", not bad, f"{len(bad)} bad")
check("main study: 3,048 trials", len(T["main"]) == 3048)
main_t = {t["trial_id"]: t for t in T["main"]}
full_t = {t["trial_id"]: t for t in T["full"]}
same = all(full_t.get(k) == v for k, v in main_t.items() if v["kind"] == "main")
check("full pool keeps every main-study trial unchanged", same, f"{sum(k in full_t for k in main_t)} of {len(main_t)} present")
tg = pd.read_csv("items/full_pool/targets.csv"); check("full pool: 884 eligible targets in trials",
    len({t['target'] for t in T['full'] if t.get('kind') == 'main'}) == 884, f"{len({t['target'] for t in T['full'] if t.get('kind')=='main'})}")

def replies(p, trials, modes, name):
    R = rj(p); keys = [(r["trial_id"], r["mode"]) for r in R]
    dup = len(keys) - len(set(keys)); unknown = {k for k in keys if k[0] not in trials}
    missing = {(t, m) for t in trials for m in modes} - set(keys)
    check(f"{name}: replies match trials", not unknown and not missing,
          f"{len(R)} rows, {dup} duplicate ids (first one counts), {len(unknown)} unknown, {len(missing)} missing")
for m in ["gemma4", "qwen3vl", "qwen25vl", "gemma3_4b", "granite4v", "lfm25v"]:
    d = f"models/{m}/main_study"
    replies(f"{d}/raw_{m}.jsonl", main_t, ("open", "mc"), f"{m} main raw")
    rep = rj(f"{d}/raw_{m}_repeat.jsonl"); orig = {(r["trial_id"], r["mode"]): r["raw"] for r in rj(f"{d}/raw_{m}.jsonl")}
    check(f"{m} repeat identical to original", all(orig.get((r["trial_id"], r["mode"])) == r["raw"] for r in rep), f"{len(rep)} replies")
    s = pd.read_csv(f"{d}/scored_{m}.csv")
    check(f"{m} scored: one row per trial and mode", len(s) == 6096 and not s.duplicated(["trial_id", "mode"]).any(), f"{len(s)} rows")
    check(f"{m} scored: joint = item and country", (s.joint_ok == (s.item_ok & s.country_ok)).all())
    check(f"{m} scored: answer slot within 0..N", ((s.item_slot.between(0, s.N)) & (s.country_slot.between(0, s.N))).all())
    ok_slot = s[(s.item_ok == 1)]; check(f"{m} scored: correct item names the target slot", (ok_slot.item_slot == ok_slot.slot).all() | (ok_slot["exp"] == "LEAK").all() or ((ok_slot[ok_slot.exp != "LEAK"].item_slot == ok_slot[ok_slot.exp != "LEAK"].slot).all()))
    for f in glob.glob(f"models/{m}/probes/raw_{m}_collage.jsonl"):
        replies(f, {t["trial_id"]: t for t in T["collage"]}, ("mc",), f"{m} collage raw")
    for f in glob.glob(f"models/{m}/probes/raw_{m}_interventions.jsonl"):
        replies(f, {t["trial_id"]: t for t in T["interv"]}, ("mc",), f"{m} interventions raw")
for m in ["gemma4", "qwen3vl", "qwen25vl", "gemma3_4b", "lfm25v", "gemma3_12b"]:
    d = f"models/{m}/full_pool"
    replies(f"{d}/raw_{m}.jsonl", full_t, ("mc",), f"{m} full-pool raw (MC)")
    s = pd.read_csv(f"{d}/scored_{m}.csv"); mc = s[s["mode"] == "mc"]
    check(f"{m} full-pool scored MC rows", len(mc) == 16720 and not mc.duplicated(["trial_id"]).any(), f"{len(mc)}")
    replies(glob.glob(f"models/{m}/one_photo_control_g1/*.jsonl")[0], {t["trial_id"]: t for t in T["g1"]}, ("mc",), f"{m} G1 raw")
for m in ["gemma4", "qwen3vl", "qwen25vl", "gemma3_4b", "lfm25v"]:
    for k in ("g2_160", "g2_full"):
        ts = {t["trial_id"]: t for t in T[k]}
        R = rj(f"models/{m}/split_query_g2/raw_{m}_{k}.jsonl")
        check(f"{m} {k}: every reply belongs to a trial", all(r["trial_id"] in ts for r in R), f"{len(R)} replies for {len(ts)} trials")
d12 = "models/gemma3_12b/main_study_160"
replies(f"{d12}/raw_gemma3_12b.jsonl", main_t, ("open", "mc"), "gemma3_12b main-study raw")
jo = pd.read_csv("judge/judge_overrides.csv")
check("judge labels: unique per model/trial/mode/field", not jo.duplicated(["model", "trial_id", "mode", "field"]).any(), f"{len(jo)} labels")
print(f"PASS {len(ok)}"); [print("  ok  ", x) for x in ok]
print(f"ISSUES {len(issues)}"); [print("  !!  ", x) for x in issues]
