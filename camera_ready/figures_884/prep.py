"""Collect every series the 884-target figures need into figdata.json (all from the analysis outputs)."""
import json, sys, numpy as np, pandas as pd
from pathlib import Path
FP = Path("work/fp884"); A = FP / "results_with_gran/results"; O = FP / "work/analysis/out"
HF = Path("hf_release/culturehaystack-results")
GS = Path("work/gscore/home/CultureHaystack_handoff/fullpool/results")
sys.path.insert(0, "CultureHaystack_handoff/wacv-lvr")
M6 = ["gemma4", "qwen3vl", "qwen25vl", "gemma3_4b", "granite4v", "lfm25v"]
rng = np.random.default_rng(0); B = 2000
def bci(v):
    v = np.asarray(v, float); b = v[rng.integers(0, len(v), (B, len(v)))].mean(1); return [float(v.mean()), *map(float, np.percentile(b, [2.5, 97.5]))]
def scored(m): return pd.read_csv(FP / f"scored_{m}.csv")
def core(d): return d[((d.exp == "E1") & (d.N <= 4)) | ((d.exp == "E2") & (d.N == 5))]
out = {}
# accuracy by N (MC, open) with CIs; Gemma-3-12B computed the same way (per-target means, target bootstrap)
acc = {}
for m in M6 + ["gemma3_12b"]:
    d = core(scored(m)); acc[m] = {}
    for mode in ("mc", "open"):
        x = d[d["mode"] == mode]
        if x.empty: continue
        acc[m][mode] = {int(N): bci(x[x.N == N].groupby("target").joint_ok.mean().values) for N in range(1, 6)}
out["acc"] = acc
# known-alone curves (MC): targets correct at N=1
known = {}
for m in M6 + ["gemma3_12b"]:
    d = core(scored(m)); d = d[d["mode"] == "mc"]
    k = set(d[(d.N == 1) & (d.joint_ok == 1)].target)
    x = d[d.target.isin(k)]
    known[m] = {int(N): bci(x[x.N == N].groupby("target").joint_ok.mean().values) for N in range(1, 6)}
out["known"] = known
# error shares by N (MC)
e = pd.read_csv(A / "error_shares_by_N.csv"); e = e[e["mode"] == "mc"]
out["errshares"] = {m: {int(r.N): [r.mis_binding, r.prior_drift, r.format_error] for r in e[e.model == m].itertuples()} for m in M6}
# slot profile N=5
p = pd.read_csv(A / "position_profile.csv"); p = p[p["mode"] == "mc"]
out["slots"] = {m: {int(r.slot): [r.acc, r.lo, r.hi] for r in p[p.model == m].itertuples()} for m in M6}
# describe-first rescue by N (MC)
dd = pd.read_csv(A / "describe_then_answer.csv"); dd = dd[(dd["mode"] == "mc") & (dd.N != "all")]
out["describe"] = {m: {int(r.N): [r.rescue, r.lo, r.hi] for r in dd[dd.model == m].itertuples()} for m in M6 if (dd.model == m).any()}
# near vs far wrong-image rate N=5
nf = pd.read_csv(A / "near_vs_far.csv"); nf = nf[(nf["mode"] == "mc") & (nf.N == 5) & (nf.metric == "mis_binding_rate")]
out["nearfar"] = {r.model: [r.near, r.far, -r.far_minus_near, -r.hi, -r.lo] for r in nf.itertuples()}
# retention control (paraphrase): f12
f12 = pd.read_csv(O / "f12_rtm_control.csv")
out["rtm"] = {r.model: [r.retention_N1_paraphrase, r.r_lo, r.r_hi, r.retention_N5_paraphrase, r.r5_lo, r.r5_hi] for r in f12.itertuples()}
# per-image cost forest: all targets (f1), category (f4), tier (f11), gold letters differ (f13)
f1 = pd.read_csv(O / "f1_per_image_cost.csv").set_index("model")
forest = {"All targets": {m: [f1.loc[m, "joint_slope"], f1.loc[m, "joint_slope_lo"], f1.loc[m, "joint_slope_hi"]] for m in M6}}
for fn, col, rows in (("f4_category_slopes.csv", "category", ["food", "landmark", "instrument", "attire"]), ("f11_tier_slopes.csv", "tier", ["high", "mid", "low"])):
    t = pd.read_csv(O / fn)
    for r in rows:
        forest[r.capitalize()] = {m: list(t[(t.model == m) & (t[col] == r)][["slope", "lo", "hi"]].values[0]) for m in M6}
g = pd.read_csv(O / "f13_slope_gold_letters_differ.csv").set_index("model")
forest["Gold letters differ"] = {m: [g.loc[m, "slope_gold_diff"], g.loc[m, "lo"], g.loc[m, "hi"]] for m in M6}
out["forest"] = forest
# wrong-pick features + pooled same-region
f7 = pd.read_csv(O / "f7_wrong_pick_features.csv")
out["wrongpick"] = {f: {r.model: [r.excess, r.lo, r.hi, r.p_holm] for r in f7[f7.feature == f].itertuples()} for f in f7.feature.unique()}
f15 = pd.read_csv(O / "f15_region_attraction_pooled.csv"); r = f15[f15.models == "all6"].iloc[0]
out["wrongpick_pooled_region"] = [r.excess, r.lo, r.hi]
# one-photo option control (860) and five-photo gap
g1 = pd.read_csv(HF / "results/full_pool/g1_near_far.csv"); g1 = g1[(g1.subset == "all860") & (g1.metric == "joint_ok")]
gg = pd.read_csv(GS / "g1_near_far.csv"); gg = gg[(gg.subset == "all860") & (gg.metric == "joint_ok")]
g1 = pd.concat([g1, gg])
out["g1"] = {r.model: [r.near_minus_far, r.lo, r.hi] for r in g1.itertuples()}
nfj = pd.read_csv(A / "near_vs_far.csv"); nfj = nfj[(nfj["mode"] == "mc") & (nfj.N == 5) & (nfj.metric == "joint_acc")]
five = {r.model: [-r.far_minus_near, -r.hi, -r.lo] for r in nfj.itertuples()}
d12 = scored("gemma3_12b"); d12 = d12[(d12["mode"] == "mc") & (d12.exp == "E3") & (d12.N == 5)]
w = d12.pivot_table(index="target", columns="list", values="joint_ok", aggfunc="mean").dropna()
v = bci((w.near - w.far).values); five["gemma3_12b"] = v
out["g1_five"] = five
# split query on all targets: same image if both wrong, split vs joint, per-target CIs; joint accuracy split vs joint
import ch
pool = ch.load_pool(str(HF / "items/main_study/pool.csv"))
T = {json.loads(l)["trial_id"]: json.loads(l) for l in open(HF / "trials/full_pool/trials_g2_full.jsonl")}
F = {json.loads(l)["trial_id"]: json.loads(l) for l in open(HF / "trials/full_pool/trials_full.jsonl")}
def cls(isrc, csrc, t, shown):
    if isrc == t and csrc == t: return "right"
    if isrc is None or csrc is None or isrc not in shown or csrc not in shown: return "other"
    if isrc == t or csrc == t: return "half"
    return "whole" if isrc == csrc else "mix"
split = {}
for m in ["gemma4", "qwen3vl", "qwen25vl", "gemma3_4b", "lfm25v"]:
    R = {}
    for l in open(HF / f"models/{m}/split_query_g2/raw_{m}_g2_full.jsonl"):
        r = json.loads(l); t = T[r["trial_id"]]; s = ch.score_trial(t, "mc", r["raw"], pool)
        sid = r["trial_id"][3:].rsplit("_Q", 1)[0]; R.setdefault(sid, {})[t["ask"]] = s
    sc = scored(m); sc = sc[(sc["mode"] == "mc") & (sc.exp == "E2")].set_index("trial_id")
    rows = []
    for sid, v in R.items():
        if "item" not in v or "country" not in v or sid not in sc.index: continue
        t = F[sid]; shown = set(t["order"])
        a = cls(v["item"]["item_src"], v["country"]["country_src"], t["target"], shown)
        jr = sc.loc[sid]; b = cls(jr.item_src if isinstance(jr.item_src, str) else None, jr.country_src if isinstance(jr.country_src, str) else None, t["target"], shown)
        rows.append(dict(target=t["target"], s=a, j=b))
    D = pd.DataFrame(rows)
    def same_rate(col):
        x = D[D[col].isin(["whole", "mix"])]; per = x.groupby("target")[col].apply(lambda s: ((s == "whole").sum(), len(s)))
        num = np.array([p[0] for p in per]); den = np.array([p[1] for p in per]); ix = rng.integers(0, len(num), (B, len(num)))
        bs = num[ix].sum(1) / den[ix].sum(1); return [float(num.sum() / den.sum()), *map(float, np.percentile(bs, [2.5, 97.5]))]
    def acc_rate(col):
        return bci(D.groupby("target")[col].apply(lambda s: (s == "right").mean()).values)
    split[m] = dict(same_split=same_rate("s"), same_joint=same_rate("j"), acc_split=acc_rate("s"), acc_joint=acc_rate("j"))
    print(m, {k: [round(x, 3) for x in v] for k, v in split[m].items()}, flush=True)
out["split"] = split
# Fig. 1 overview numbers: pooled wrong-answer types (six models) and retention loss
w = pd.read_csv(O / "wrong_answer_types.csv").set_index("model")
out["types"] = w.reset_index().to_dict("records")
f2 = pd.read_csv(O / "f2_acc_given_known_at_N1.csv"); f2 = f2[f2.N == 5].set_index("model")
out["ret5"] = {m: float(f2.loc[m, "acc_given_known"]) for m in M6}
json.dump(out, open("work/figs884/figdata.json", "w"), indent=1, default=float)
print("saved")
