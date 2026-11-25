"""Phase 09: build REPORT.md from results/, RUN_LOG.md, logs/ (CPU). Everything is labelled post-registration / exploratory."""
import os, re, json
from pathlib import Path
import pandas as pd
F = Path.home()/"CultureHaystack_handoff"/"fullpool"; os.chdir(F)
def tbl(p, cols=None, fmt=3):
    if not Path(p).exists(): return "_not available yet_\n"
    d = pd.read_csv(p)
    if d.empty: return "_no rows yet_\n"
    if cols: d = d[[c for c in cols if c in d.columns]]
    d = d.round(fmt).astype(object).where(d.notna(), "")
    return "| " + " | ".join(map(str, d.columns)) + " |\n|" + "---|" * len(d.columns) + "\n" + "\n".join("| " + " | ".join(map(str, r)) + " |" for r in d.values) + "\n"
def rows(p):
    try: return sum(1 for _ in open(p, "rb"))
    except Exception: return 0
L = ["# CultureHaystack full-pool run: report", "", "**All results below are post-registration / exploratory. None is pre-registered.** The 160-target frozen study stays primary.", ""]
L += ["## 1. Environment", "", "Cluster: single-node on-demand GPU host without a scheduler (A100 40 GB, driver 550 / CUDA 12.4). Jobs run through a dependency queue (ext/jobqueue.py).", "", "~~~", Path("logs/env.txt").read_text()[:2500] if Path("logs/env.txt").exists() else "env.txt missing", "~~~", "", "Model revisions:", "", "~~~", Path("logs/model_revisions.txt").read_text() if Path("logs/model_revisions.txt").exists() else "see HF cache", "~~~", ""]
L += ["## 2. What ran", "", "| block | file | rows | expected |", "|---|---|---|---|"]
js = json.load(open("jobs/queue_jobs.json")); st = json.load(open("logs/queue_state.json")) if Path("logs/queue_state.json").exists() else {}
for j in js:
    if j.get("expect_file") and j["name"] != "clip_sim":
        L.append(f"| {j['name']} | {j['expect_file']} | {rows(j['expect_file'])} | {j['expect_rows']} | {st.get(j['name'], {}).get('status', 'pending')} |".replace("| |", "|"))
L += ["", "## 3. Reproduction", "", "P0 (20 calls per model vs frozen replies):", ""]
for p in sorted(Path("logs").glob("p0_*.out")):
    m = [l for l in p.read_text().splitlines() if l.startswith("REPRO")]
    L.append("- " + (m[-1] if m else p.name + ": no result"))
L += ["", "Granite-Vision-4B reproduced 15/20 (below 18) and was excluded by decision; no code was changed to fix it.", "", "Large-scale identical-reply rate on the 3,048 frozen trials inside F1:", "", tbl("results/repro_large.csv")]
L += ["## 4. G1 near vs far options at N=1", "", tbl("results/g1_near_far.csv")]
L += ["## 5. G3 Gemma-3-12B (piloted before the freeze and dropped from the roster)", "", tbl("results/g3_table2.csv")]
L += ["## 6. G2 split query", "", tbl("results/g2_split.csv")]
L += ["## 7. F1 full pool", "", "Per-model tables, slot profiles and near/far cells are in results/f1_subsets.csv. Pooled, frozen-160, new-724 and per category:", "", tbl("results/f1_subsets.csv"), "Original analysis outputs are in results/ (analyze.py)."]
fixes = [l for l in Path("RUN_LOG.md").read_text().splitlines() if re.search(r"\[fix\]|\[queue\].*(rc=|FAILED|STOPPED|blocked)|Decision|Kaggle|Judge|judge", l)]
L += ["", "## 8. Deviations and problems", ""] + ["- " + l.lstrip("- ")[:400] for l in fixes]
L += ["- No scheduler on the node: GPU jobs ran as background processes under a dependency queue instead of SLURM.", "- The Kaggle API returned 404 from the node; the dataset zip was uploaded by hand. The byte-identical version check passed.", "- The 31B judge (about 62 GB) does not fit a 40 GB GPU. Open-mode answers that the string matcher could not resolve are unjudged; open-mode numbers are lower bounds unless judge_overrides.csv exists.", "- Python bytecode caches were written into wacv-lvr/__pycache__ by imports before PYTHONDONTWRITEBYTECODE was set; no listed file changed (see frozen_check.txt).", "", "## 9. Labels", "", "Every result here is post-registration / exploratory."]
Path("REPORT.md").write_text("\n".join(L) + "\n"); print("REPORT.md written", len("\n".join(L)), "chars")
