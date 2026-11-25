"""Task 7. Scorer edge cases in the frozen scored_<m>.csv (frozen ch.py logic re-applied to the stored raw text).
 (i)  MC: the parser's letter regex \(?([A-Ea-e])\b accepted a LOWERCASE letter; 'article' = lowercase letter
      followed by a space and a word (e.g. 'a traditional ...'), i.e. not 'a.', 'a)', or a bare 'a'.
 (ii) open: a field resolved to the TARGET only by whole-word containment (norm(answer) is neither the gold name
      nor one of its aliases), e.g. 'Unknown (Tarte Tatin)'.
 (iii) open country: answer names >=2 distinct countries (canonical pool country names, whole words), e.g.
      'Israel/Palestine', and was scored correct.
By default fields set by a judge override are excluded; arg incl_overrides counts them too (their label is the judge's, not the scorer's).
'Rescued' = joint_ok==1 rows that become wrong when the edge-case field is scored wrong.
Then Table 1 (analyze.scaling on core) is recomputed with each case, and all cases, scored wrong.
Run from cr_analyses/_frozen. Descriptive recomputation; the 'scored as wrong' variants are exploratory."""
import sys, re; sys.path.insert(0, ".")
import numpy as np, pandas as pd
import analyze as A, ch
MODELS = ["gemma4", "qwen3vl", "qwen25vl", "gemma3_4b", "granite4v", "lfm25v"]
pool = ch.load_pool()
INCL = len(sys.argv) > 1 and sys.argv[1] == "incl_overrides"; TAG = "_incl_overrides" if INCL else ""
ov = pd.read_csv("judge_overrides.csv"); ovk = set() if INCL else set(zip(ov.model, ov.trial_id, ov["mode"], ov.field))
countries = sorted({ch.norm(r["country"]) for r in pool.values()} - {""}, key=len, reverse=True)
df = A.load(MODELS)

def field_val(raw, key):
    m = re.search(rf"{key}\s*:\s*(.+)", raw if isinstance(raw, str) else "", re.I)
    return m.group(1).strip().strip("*_`<> ") if m else None

def lower_letter(raw, key):
    v = field_val(raw, key)
    if v is None: return None
    L = re.match(r"\(?([A-Ea-e])\b", v)
    if not L or not L.group(1).islower(): return None
    return "article" if re.match(r"\(?[a-e]\s+[A-Za-z]{2,}", v) else "letter"

def contain(ans, tgt, f):
    if not isinstance(ans, str): return False
    a = ch.norm(ans); r = pool[tgt]
    return a != ch.norm(r[f]) and a not in {ch.norm(x) for x in r[f + "_aliases"]}

def n_countries(ans):
    a = " " + ch.norm(ans) + " " if isinstance(ans, str) else ""
    found = set()
    for c in countries:
        if f" {c} " in a: found.add(c); a = a.replace(f" {c} ", " | ")
    return len(found)

flags = {k: pd.Series(False, index=df.index) for k in ("i_item", "i_country", "ii_item", "ii_country", "iii_country")}
kinds = pd.Series("", index=df.index)
for idx, r in df.iterrows():
    for f, key in (("item", "Item"), ("country", "Country")):
        if (r.model, r.trial_id, r["mode"], f) in ovk: continue
        if r["mode"] == "mc":
            k = lower_letter(r.raw, key)
            if k == "article" and r[f"{f}_ok"] == 1 or k == "article":
                flags[f"i_{f}"][idx] = True
        elif r[f"{f}_ok"] == 1 and contain(r[f"{f}_ans"], r.target, f):
            flags[f"ii_{f}"][idx] = True
    if r["mode"] == "open" and r.country_ok == 1 and n_countries(r.country_ans) >= 2 \
            and (r.model, r.trial_id, "open", "country") not in ovk:
        flags["iii_country"][idx] = True
for k, v in flags.items(): df[k] = v
case = {"i": ["i_item", "i_country"], "ii": ["ii_item", "ii_country"], "iii": ["iii_country"]}
rows = []
for c, cols in case.items():
    for (m, mode), g in df.groupby(["model", "mode"]):
        hit = g[cols].any(axis=1)
        okf = np.zeros(len(g), bool)
        for col in cols:
            f = col.split("_")[1]; okf |= g[col].values & (g[f"{f}_ok"].values == 1)
        if hit.sum() == 0: continue
        rows.append(dict(case=c, model=m, mode=mode, rows=int(hit.sum()), fields=int(g[cols].values.sum()),
                         field_scored_correct=int(okf.sum()), rescued_joint=int((okf & (g.joint_ok.values == 1)).sum()),
                         in_core=int((hit & g.exp.isin(["E1", "E2"]) & (g.variant == "P1") & (g.list == "mixed") & (g.kind == "main")).sum())))
cnt = pd.DataFrame(rows); cnt.to_csv(f"../07_scorer_edges/edge_counts{TAG}.csv", index=False); print(cnt.to_string(index=False))
ex = df[df[sum(case.values(), [])].any(axis=1)][["model", "mode", "trial_id", "item_ans", "country_ans", "joint_ok"] + sum(case.values(), [])]
ex.to_csv(f"../07_scorer_edges/edge_rows{TAG}.csv", index=False)

def table1(d):
    core, _ = A.core_sets(d); acc, summ = A.scaling(core)
    j = acc[acc.metric == "joint"].pivot_table(index=["model", "mode"], columns="N", values="acc")
    s = summ.set_index(["model", "mode"])
    return pd.DataFrame({"acc1": j[1], "acc5": j[5], "drop": s.initial_drop, "slope": s.slope_per_image}).round(3)
base = table1(df)
res = []
for c, cols in list(case.items()) + [("all", sum(case.values(), []))]:
    d = df.copy()
    for col in cols:
        f = col.split("_")[1]; d.loc[d[col], f"{f}_ok"] = 0
    d["joint_ok"] = d.item_ok * d.country_ok
    t = table1(d); diff = (t - base)
    ch_ = diff[(diff != 0).any(axis=1)]
    res.append(ch_.assign(case=c))
    print(f"\ncase {c}: Table-1 cells changed at 3rd decimal (new - frozen):\n", ch_.to_string() if len(ch_) else "  none")
pd.concat(res).to_csv(f"../07_scorer_edges/table1_changes{TAG}.csv")
