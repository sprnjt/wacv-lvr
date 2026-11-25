# Camera-ready code (WACV 2027 LVR, paper 36)

This branch (`camera-ready-884`) adds the code used for the camera-ready version of CultureHaystack to the code of
the pre-registered study and probes on `post-freeze-analysis`. In the camera-ready, every main result is reported on
**all 884 eligible targets**. The 160-target subset is used only for the probes.

**Where the other parts live:**
- **Results** (model replies, scored answers, analysis tables, figure data): the Hugging Face dataset
  [shinjinii/culturehaystack-results](https://huggingface.co/datasets/shinjinii/culturehaystack-results).
- **Images:** the Kaggle dataset
  [World Cultural Items 1K](https://www.kaggle.com/datasets/shinjinichakraborty/world-cultural-items-1k), version 2.

The frozen study code in the repository root (`ch.py`, `run_models.py`, `analyze.py`, `work/analysis/`) is unchanged.
Everything below reuses it.

| Folder | What it is | Used for |
|---|---|---|
| `cluster_fullpool/` | Scripts and job files of the run on all 884 targets (one NVIDIA A100); `REPORT.md` records what ran, the environment, the reproduction checks and the problems during the run | Gemma-4, both Qwens, Gemma-3-4B, LFM2.5 and Gemma-3-12B on the full pool; the one-photo control (`build_g1.py`); the split query (`build_g2.py`, `ch_g2.py`); the per-run reproduction checks (`check_repro.py`, `check_det.py`) |
| `granite_kaggle/` | Runner for Granite-Vision-4B on Kaggle T4 GPUs | Granite's MC haystack-size and slot tests, the one-photo control and the 160-target split query. `gen.py` embeds the frozen `ch.py`, `run_models.py` and `ch_g2.py` byte for byte, checked by SHA-256. It writes one Kaggle script per chunk (one model copy per GPU, fp16). `merge_and_score.sh` merges the chunks and scores them with the frozen scorer |
| `analysis_884/` | Glue that runs the frozen analysis on the 884 targets with all seven models | `analyze.py` and `work/analysis/*.py` (`MODELS` list extended to Granite), the reworded-prompt control, moderators, wrong-pick features; `table2_884.py` recomputes Table 2 |
| `camera_ready_checks/` | Earlier camera-ready checks | full-pool CIs, split-query same-image, CLIP control on shown distractors, Holm corrections, slot × tier, format errors, scorer edge cases |
| `figures_884/` | `prep.py` collects every plotted series from the analysis outputs; `plot.py` draws the paper and supplement figures | All redrawn figures |
| `verification/` | `reproduce_all.py` reruns every analysis script and checks that each number printed in the paper matches the outputs. It expects the review-package layout (`main_study/`, `post_registration/`, `paper/`) next to it; `setup_links.py` and `links.json` set that layout up. `consistency_checks/` holds the line-by-line checks of the paper's numbers against the released data. `build_results_dataset.py` assembled the Hugging Face dataset from the review package | Checking that the paper and the released data agree |
| `gemma3_12b_160/` | `table1_g3.py` computes the Gemma-3-12B row on the 160-target subset next to Gemma-3-4B | The same-family size control on the original sample |

## Reproducing the 884-target numbers

1. **Prepare the data.** Download the Kaggle images. Run `python ch.py prep <kaggle_dir>`, then build the full pool
   with `cluster_fullpool/ext/build_full.py`. Trial files are also in the HF dataset under `trials/full_pool/`.
2. **Run the models.** Run each model with `run_models.py` on `trials_full.jsonl`; see `cluster_fullpool/jobs/`. For Granite on T4 GPUs, use `granite_kaggle/gen.py` and `merge_and_score.sh`. The replies are
   already in the HF dataset under `models/<model>/full_pool/`, so this step can be skipped.
3. **Score.** Score with `python ch.py score <model> --trials trials_full.jsonl`.
4. **Analyse.** Run `python analyze.py gemma4 qwen3vl qwen25vl gemma3_4b granite4v lfm25v`, then the scripts in
   `analysis_884/`.
5. **Draw the figures.** Run `figures_884/prep.py`, then `figures_884/plot.py`.

**Paths in these scripts** are relative to the project folder they were run from. That folder holds this repository
at `wacv/wacv-lvr/`, a copy of the Hugging Face dataset at `hf_release/culturehaystack-results/`, and a `work/` folder
for intermediate files.

**Credentials** (Hugging Face, Kaggle) are read from the environment or local config files. None are stored here.

During the cluster run, progress checkpoints were copied to a temporary Hugging Face bucket, which was deleted
afterwards. All final data is in the Hugging Face dataset repository
[shinjinii/culturehaystack-results](https://huggingface.co/datasets/shinjinii/culturehaystack-results).
