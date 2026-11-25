"""Task 3 (POST-REGISTRATION / EXPLORATORY). H3 near-region test with a CLIP control computed from the distractor
images actually SHOWN in each trial (frozen control: mean over all 4 distractors of the list).
Images: rebuilt with frozen `ch.py prep` from the Kaggle dataset into work/images/prepared.
CLIP model and feature code: as frozen clip_sim.py (openai/clip-vit-base-patch32, L2-normalised image features, cosine).
Regression: as frozen analyze.near_far (E3 main trials; logistic GLM joint_ok ~ 1 + near + N5 + clip, SE clustered by
target; missing clip filled with column mean; Holm over the 12 model x mode cells). CPU.
Run from cr_analyses/03_clip_shown/work."""
import sys, json, csv
sys.path.insert(0, "."); sys.path.insert(0, "../../_frozen")
import numpy as np, pandas as pd, torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
import ch
import analyze as A   # frozen copy in _frozen (fit, holm); its load() reads scored files from cwd -> use explicit paths
FZ = "wacv/wacv-lvr/"
CLIP = "openai/clip-vit-base-patch32"
torch.set_num_threads(4)
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
        feats.update(dict(zip(chunk, torch.nn.functional.normalize(out, dim=-1))))
sim = lambda a, b: float(feats[a] @ feats[b])
reb = pd.DataFrame([dict(target=t, list=n, mean_sim=np.mean([sim(t, x) for x in d])) for t, ls in lists.items() for n, d in ls.items() if d])
reb.to_csv("../clip_sim_rebuilt.csv", index=False)
fz = pd.read_csv(FZ + "clip_sim.csv").merge(reb, on=["target", "list"], suffixes=("_frozen", "_rebuilt"))
print(f"rebuilt vs frozen clip_sim.csv: {len(fz)} rows matched of {len(reb)}; max |diff| = {(fz.mean_sim_frozen - fz.mean_sim_rebuilt).abs().max():.2e}")
tr = {t["trial_id"]: t for t in ch.read_jsonl(FZ + "trials_main.jsonl")}
shown = {k: np.mean([sim(t["target"], x) for x in t["order"] if x != t["target"]]) for k, t in tr.items() if t["exp"] == "E3"}
pd.Series(shown, name="shown_sim").rename_axis("trial_id").to_csv("../clip_shown_per_trial.csv")
MODELS = ["gemma3_4b", "gemma4", "granite4v", "lfm25v", "qwen25vl", "qwen3vl"]
regs = []
for m in MODELS:
    d = pd.read_csv(f"{FZ}scored_{m}.csv")
    d0 = d[(d.exp == "E3") & (d.kind == "main")]
    for mode, g in d0.groupby("mode"):
        g = g.reset_index(drop=True)
        base = pd.DataFrame({"const": 1.0, "near": (g.list == "near").astype(float), "N5": (g.N == 5).astype(float)})
        ctrls = {"frozen_mean4": g[["target", "list"]].merge(pd.read_csv(FZ + "clip_sim.csv"), on=["target", "list"], how="left")["mean_sim"],
                 "rebuilt_mean4": g[["target", "list"]].merge(reb, on=["target", "list"], how="left")["mean_sim"],
                 "shown": g.trial_id.map(shown)}
        for cname, cv in ctrls.items():
            X = base.assign(clip=cv.values); X = X.fillna(X.mean())
            r = A.fit(g.joint_ok.values.astype(float), X.values, pd.factorize(g.target)[0])
            lo, hi = r.conf_int()[1]
            regs.append(dict(model=m, mode=mode, control=cname, odds_ratio=np.exp(r.params[1]), or_lo=np.exp(lo), or_hi=np.exp(hi),
                             near_p=r.pvalues[1], clip_coef=r.params[3], clip_p=r.pvalues[3], n_trials=len(g)))
R = pd.DataFrame(regs)
for c in R.control.unique():
    k = R.control == c; R.loc[k, "near_p_holm"] = A.holm(R.loc[k, "near_p"])
R.to_csv("../h3_near_regression_shown.csv", index=False)
fh = pd.read_csv(FZ + "results/h3_near_regression.csv").set_index(["model", "mode"])
chk = R[R.control == "frozen_mean4"].set_index(["model", "mode"])
print("frozen-control reproduction: max |OR diff| =", float((chk.odds_ratio - fh.odds_ratio).abs().max()),
      " max |Holm p diff| =", float((chk.near_p_holm - fh.near_p_holm).abs().max()))
print(R.round(4).to_string(index=False))
