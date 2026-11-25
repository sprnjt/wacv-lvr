"""Build the clean, results-only, model-wise Hugging Face dataset from the review package.
Source: hf_release/culturehaystack-review (verified: 75/75 reproduction checks). Output: hf_release/culturehaystack-results.
Every file is copied unchanged (sha256 kept in MANIFEST.csv with its source path); only the figure CSVs are new,
computed here from the copied files. Nothing in the source is modified."""
import csv, hashlib, json, os, shutil, sys
from pathlib import Path

import pandas as pd

SRC = Path(__file__).resolve().parent / "culturehaystack-review"
OUT = Path(__file__).resolve().parent / "culturehaystack-results"
MS, FP = SRC / "main_study", SRC / "post_registration/fullpool_package"
GPU = MS / "work/gpu_out"
G3 = SRC / "post_registration/gemma3_12b_table1"
CR = SRC / "post_registration/camera_ready_analyses"
EXTRA = SRC / "post_registration/cluster_bucket_extras/culturehaystack"
SIX = ["gemma4", "qwen3vl", "qwen25vl", "gemma3_4b", "granite4v", "lfm25v"]
FIVE = ["gemma4", "qwen3vl", "qwen25vl", "gemma3_4b", "lfm25v"]          # post-registration runs (Granite excluded)
NAME = {"gemma4": "Gemma-4-12B", "qwen3vl": "Qwen3-VL-8B", "qwen25vl": "Qwen2.5-VL-7B", "gemma3_4b": "Gemma-3-4B",
        "granite4v": "Granite-Vision-4B", "lfm25v": "LFM2.5-VL-3B", "gemma3_12b": "Gemma-3-12B"}
LOG = []


def put(src, dst):
    src, dst = Path(src), OUT / dst
    assert src.is_file(), src
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    LOG.append((str(dst.relative_to(OUT)), str(src.relative_to(SRC)), hashlib.sha256(src.read_bytes()).hexdigest()))


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir()

    # ---------------- per model
    for m in SIX:
        d = f"models/{m}/main_study"
        put(MS / f"raw_{m}.jsonl", f"{d}/raw_{m}.jsonl")
        put(MS / f"raw_{m}_repeat.jsonl", f"{d}/raw_{m}_repeat.jsonl")
        put(MS / f"scored_{m}.csv", f"{d}/scored_{m}.csv")
        put(MS / f"review_{m}.csv", f"{d}/review_{m}.csv")
        if (MS / f"attn_{m}.jsonl").exists():
            put(MS / f"attn_{m}.jsonl", f"{d}/attn_{m}.jsonl")
        p = f"models/{m}/probes"
        for f in sorted(GPU.glob(f"raw_{m}_*.jsonl")):
            put(f, f"{p}/{f.name}")
        for f in sorted(GPU.glob(f"*_{m}.jsonl")):                    # causal_, causal_cases_, attn_ (LFM extension)
            put(f, f"{p}/{f.name}")
        for f in sorted((GPU / "results").glob(f"*_{m}*.csv")):
            put(f, f"{p}/results/{f.name}")
    put(G3 / "noov/scored_gemma3_4b.csv", "models/gemma3_4b/main_study/scored_gemma3_4b_without_judge_labels.csv")

    for m in FIVE:
        d = f"models/{m}/full_pool"
        put(FP / f"raw_{m}.jsonl", f"{d}/raw_{m}.jsonl")
        put(FP / f"scored_{m}.csv", f"{d}/scored_{m}.csv")
        put(FP / f"review_{m}.csv", f"{d}/review_{m}.csv")
        put(FP / f"raw_{m}_g1.jsonl", f"models/{m}/one_photo_control_g1/raw_{m}_g1.jsonl")
        put(FP / f"raw_{m}_g2_160.jsonl", f"models/{m}/split_query_g2/raw_{m}_g2_160.jsonl")
        put(FP / f"raw_{m}_g2_full.jsonl", f"models/{m}/split_query_g2/raw_{m}_g2_full.jsonl")

    m = "gemma3_12b"                                                   # post-registration model
    put(FP / "raw_gemma3_12b_main.jsonl", f"models/{m}/main_study_160/raw_gemma3_12b.jsonl")
    put(G3 / "work/scored_gemma3_12b.csv", f"models/{m}/main_study_160/scored_gemma3_12b.csv")
    put(G3 / "work/review_gemma3_12b.csv", f"models/{m}/main_study_160/review_gemma3_12b.csv")
    put(FP / "raw_gemma3_12b.jsonl", f"models/{m}/full_pool/raw_gemma3_12b.jsonl")
    put(CR / "_fullpool/scored_gemma3_12b.csv", f"models/{m}/full_pool/scored_gemma3_12b.csv")
    put(CR / "_fullpool/review_gemma3_12b.csv", f"models/{m}/full_pool/review_gemma3_12b.csv")
    put(FP / "raw_gemma3_12b_g1.jsonl", f"models/{m}/one_photo_control_g1/raw_gemma3_12b_g1.jsonl")

    # ---------------- what was asked (trials) and the item pool (no images)
    for f in ("trials_main.jsonl", "trials_pilot.jsonl"):
        put(MS / f, f"trials/main_study/{f}")
    for f in ("trials_collage.jsonl", "trials_interventions.jsonl"):
        put(MS / f, f"trials/probes/{f}")
    for f in ("trials_full.jsonl", "trials_g1.jsonl", "trials_g2_160.jsonl", "trials_g2_full.jsonl"):
        put(FP / f, f"trials/full_pool/{f}")
    for f in ("pool.csv", "targets.csv", "lists.json", "audit.txt", "disputed_labels.txt", "label_audit.txt"):
        put(MS / "data" / f, f"items/main_study/{f}")
    put(MS / "clip_sim.csv", "items/main_study/clip_sim.csv")
    put(MS / "exclude.txt", "items/main_study/exclude.txt")
    put(FP / "data/targets.csv", "items/full_pool/targets.csv")
    put(FP / "data/lists.json", "items/full_pool/lists.json")
    put(FP / "clip_sim.csv", "items/full_pool/clip_sim.csv")
    put(CR / "03_clip_shown/clip_sim_rebuilt.csv", "items/full_pool/clip_sim_rebuilt_from_kaggle.csv")

    # ---------------- judge (open answers)
    put(MS / "judge_overrides.csv", "judge/judge_overrides.csv")
    for f in ("judge_log.jsonl", "judge_sample_100.jsonl"):
        put(MS / "data" / f, f"judge/{f}")
    put(EXTRA / "fullpool_progress_20261120_1052_ckpt/data/judge_q4_validation.csv", "judge/judge_q4_validation.csv")

    # ---------------- results tables
    for f in sorted((MS / "results").glob("*.csv")):
        put(f, f"results/main_study/{f.name}")
    put(MS / "results/locus_discrimination.md", "results/main_study/locus_discrimination.md")
    for f in sorted((FP / "results").glob("*.csv")):
        if f.stat().st_size > 5:                                       # attention_capture.csv is empty (no probe run)
            put(f, f"results/full_pool/{f.name}")
    put(EXTRA / "fullpool_progress_20261119_1047_g3_full_scored/results/g3_full_by_N.csv", "results/full_pool/g3_full_by_N.csv")
    for f in ("acc_by_N_gemma3_12b.csv", "scaling_summary_gemma3_12b.csv"):
        put(G3 / f, f"results/gemma3_12b_main_study/{f}")
    for f in ("acc_by_N_gemma3_4b.csv", "scaling_summary_gemma3_4b.csv"):
        put(G3 / "noov" / f, f"results/main_study/without_judge_labels/{f}")
    for sub in ("01_fullpool_ci", "02_g2_same_image", "03_clip_shown", "04_holm_mc", "05_slot_tier", "06_format_errors",
                "07_scorer_edges"):
        for f in sorted((CR / sub).glob("*.csv")):
            if f.name != "clip_sim_rebuilt.csv":
                put(f, f"results/camera_ready/{sub}/{f.name}")

    for f in sorted((MS / "work/analysis/out").glob("*.csv")):        # exploratory analyses (supplement)
        put(f, f"results/main_study/exploratory/{f.name}")
    put(G3 / "noov/review_gemma3_4b.csv", "models/gemma3_4b/main_study/review_gemma3_4b_without_judge_labels.csv")
    put(FP / "MANIFEST.sha256", "checks/full_pool_cluster_MANIFEST.sha256")

    # ---------------- run checks (replay / determinism)
    put(FP / "results/repro_large.csv", "checks/replay_main_study_on_new_gpu.csv")
    put(FP / "frozen_check.txt", "checks/frozen_data_rebuild_check.txt")
    put(EXTRA / "fullpool_progress_20261113_2303_g1_partial_gemma3_4b_lfm25v/logs/repro_granite4v.jsonl",
        "checks/replay_granite4v_failed.jsonl")

    # ---------------- pre-registration
    put(SRC / "prereg/PREREG_MANIFEST.sha256", "preregistration/PREREG_MANIFEST.sha256")

    figures()
    with open(OUT / "MANIFEST.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["file", "source_in_review_package", "sha256"])
        w.writerows(sorted(LOG))
    print(len(LOG), "files")


def write(df, name):
    p = OUT / "figures" / name
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)
    LOG.append((f"figures/{name}", "computed (see figures/README in the dataset card)", hashlib.sha256(p.read_bytes()).hexdigest()))


def figures():
    D = json.loads((CR / "10_figures_redesign/figdata.json").read_text())   # numbers recomputed from the files above
    eb = pd.read_csv(MS / "results/joint_error_breakdown.csv")
    rows = [dict(model=r.model, wrong_answers=r.wrong, whole_swap=r.whole, half_right=r.half, two_image_mix=r.two,
                 format_error=r.other, same_image_if_both_wrong=r.same) for r in eb.itertuples()]
    w = eb.wrong
    rows.append(dict(model="pooled", wrong_answers=int(w.sum()), **{k: float((eb[c] * w).sum() / w.sum()) for k, c in
                (("whole_swap", "whole"), ("half_right", "half"), ("two_image_mix", "two"), ("format_error", "other"))},
                same_image_if_both_wrong=float((eb.same * eb.both).sum() / eb.both.sum())))
    write(pd.DataFrame(rows), "fig1c_wrong_answer_types.csv")

    rows = []
    for m, v in D["accN"].items():
        rows += [dict(model=m, panel="all_targets", N=r[0], acc=r[1], lo=r[2], hi=r[3], n_targets=D["accN_targets"]) for r in v]
    for m, v in D["known"].items():
        rows += [dict(model=m, panel="known_alone", N=r[0], acc=r[1], lo=r[2], hi=r[3], n_targets=v["n"]) for r in v["rows"]]
    write(pd.DataFrame(rows), "fig1d_accuracy_by_N.csv")

    rows = []
    for m, v in D["causal"].items():
        for k, lab in (("rescue", "clean_target_patch"), ("induct", "wrong_image_patch"), ("self", "self_patch")):
            rows += [dict(model=m, layer=r[0], patch=lab, delta_logodds=r[1], lo=r[2], hi=r[3], n_cases=v["n"],
                          n_targets=v["ntg"]) for r in v["rows"][k]]
    write(pd.DataFrame(rows), "fig3a_patching.csv")

    rows = []
    for m, v in D["e9"].items():
        for c, lab in (("solo", "target_alone"), ("separate", "four_separate_images"), ("collage_pos", "collage_by_position"),
                       ("collage_kind", "collage_by_kind")):
            rows.append(dict(model=m, condition=lab, joint_acc=v["acc"][c][0], lo=v["acc"][c][1], hi=v["acc"][c][2], n=v["n"]))
        rows.append(dict(model=m, condition="collage_minus_alone", joint_acc=v["d_c3c1"][0], lo=v["d_c3c1"][1],
                         hi=v["d_c3c1"][2], n=v["n"]))
    write(pd.DataFrame(rows), "fig3b_collage.csv")

    rows = []
    for m, v in D["near_five"].items():
        rows.append(dict(model=m, setting="five_photos", acc_lost_near_vs_far=v[0], lo=v[1], hi=v[2], n_targets=v[3]))
    for m, v in D["near_one"].items():
        rows.append(dict(model=m, setting="target_alone", acc_lost_near_vs_far=v[0], lo=v[1], hi=v[2], n_targets=v[3]))
    write(pd.DataFrame(rows), "fig4a_near_options.csv")
    rows = [dict(model=k.split("|")[0], prompt={"frozen": "joint", "split": "split"}[k.split("|")[1]], same_image_pct=v[0],
                 lo=v[1], hi=v[2], n_trials=800) for k, v in D["same"].items()]
    write(pd.DataFrame(rows), "fig4b_same_image_if_both_wrong.csv")
    rows = [dict(model=m, N=r[0], joint_acc=r[1], lo=r[2], hi=r[3], n_targets=884) for m, v in D["fullpool"].items() for r in v]
    write(pd.DataFrame(rows), "fig4c_full_pool_accuracy.csv")

    # Fig 2: the three trials, from the scored replies (photos are on Kaggle; ids in items/main_study/pool.csv)
    sys.path.insert(0, str(MS)); cwd = os.getcwd(); os.chdir(MS)
    import ch
    P = ch.load_pool()
    TC = {t["trial_id"]: t for t in ch.read_jsonl("trials_collage.jsonl")}
    os.chdir(cwd)
    rows = []
    lfm = pd.read_csv(MS / "scored_lfm25v.csv"); g4 = pd.read_csv(MS / "scored_gemma4.csv")
    for panel, tid, frames in (("a_half_right", "E2_img_0260_mixed_N5_k3_P1", (("lfm25v", lfm), ("gemma4", g4))),
                               ("b_whole_swap", "E2_img_0805_mixed_N5_k3_P1", (("lfm25v", lfm),))):
        for m, df in frames:
            r = df[(df.trial_id == tid) & (df["mode"] == "mc")].iloc[0]
            rows.append(dict(panel=panel, model=m, trial_id=tid, asked_slot=r.slot, item_answer=r.item_ans, item_ok=r.item_ok,
                             item_from_slot=r.item_slot, country_answer=r.country_ans, country_ok=r.country_ok,
                             country_from_slot=r.country_slot))
    raw = {}
    for l in open(GPU / "raw_qwen3vl_collage.jsonl"):
        r = json.loads(l)
        if r.get("mode") == "mc" and r["trial_id"].startswith("E9_img_0618_") and r["trial_id"] not in raw:
            raw[r["trial_id"]] = r["raw"]
    for c in ("solo", "separate", "collage_pos"):
        tid = f"E9_img_0618_{c}"; s = ch.score_trial(TC[tid], "mc", raw[tid], P)
        rows.append(dict(panel="c_collage", model="qwen3vl", trial_id=tid, asked_slot=TC[tid]["slot"], item_answer=s["item_ans"],
                         item_ok=s["item_ok"], item_from_slot=s["item_slot"], country_answer=s["country_ans"],
                         country_ok=s["country_ok"], country_from_slot=s["country_slot"]))
    write(pd.DataFrame(rows), "fig2_example_trials.csv")


if __name__ == "__main__":
    main()
