# Full-pool cluster run: report

This is the record of the run on all 884 eligible targets (14–18 Nov 2026). Everything here is post-registration and
exploratory; the pre-registered plan specified the 160-target sample. The final numbers used in the paper are in the
Hugging Face dataset [shinjinii/culturehaystack-results](https://huggingface.co/datasets/shinjinii/culturehaystack-results)
under `results/paper_884/`. The replies listed below are there under `models/<model>/`.

`ext/make_report.py` wrote the first version of this report at the end of the run. It was then edited by hand for
release: the tables were laid out again and the environment filled in. No number was changed.

## 1. Environment

- **Hardware:** one node with one NVIDIA A100 40 GB (driver 550, CUDA 12.4), shared with other users. The node had no
  job scheduler, so GPU jobs ran as background processes under a dependency queue (`ext/jobqueue.py`).
- **Software:** Python 3.11, torch 2.9.1 and torchvision 0.24.1 (CUDA 12.6 wheels), transformers 5.20.1,
  accelerate 1.15.0. The install script is `ext/setup02.sh`; `env.sh` sets offline model use and the paths.
- **Code:** the frozen study code (`ch.py`, `run_models.py`) unchanged; its 258 files were checked against the frozen
  checksums before the run (all OK).
- **Models** (Hugging Face Hub IDs):

| Key | Model |
|---|---|
| `gemma4` | `google/gemma-4-12B-it` |
| `qwen3vl` | `Qwen/Qwen3-VL-8B-Instruct` |
| `qwen25vl` | `Qwen/Qwen2.5-VL-7B-Instruct` |
| `gemma3_4b` | `google/gemma-3-4b-it` |
| `lfm25v` | `LiquidAI/LFM2.5-VL-3B` |
| `gemma3_12b` | `google/gemma-3-12b-it` |

  The weights were the Hub versions current at the time of the run. Revision hashes were not recorded (as stated in
  the supplement).

Granite-Vision-4B was not run here (see section 3). It was run separately on Kaggle; see `../granite_kaggle/`.

## 2. What ran

Each row is one block of the run. **Replies in file** counts every line of the output file; the full-pool files hold
both MC and open replies, so they have 2 × 16,720 = 33,440 lines. All blocks finished.

| Block | Output file | Targets | Mode | Replies expected | Replies in file |
|---|---|---|---|---|---|
| Full pool, MC (5 models) | `raw_<model>.jsonl` | 884 | MC | 16,720 | 33,440 (MC + open) |
| Full pool, open (5 models) | `raw_<model>.jsonl` | 884 | open | 16,720 | 33,440 (MC + open) |
| Full pool, Gemma-3-12B | `raw_gemma3_12b.jsonl` | 884 | MC | 16,720 | 16,720 |
| One-photo control (G1, 6 models) | `raw_<model>_g1.jsonl` | 860 | MC | 1,720 | 1,720 |
| Split query, all targets (G2, 5 models) | `raw_<model>_g2_full.jsonl` | 884 | MC | 8,840 | 8,840 |
| Split query, 160-target subset (G2, 5 models) | `raw_<model>_g2_160.jsonl` | 160 | MC | 1,600 | 1,600 |
| Gemma-3-12B on the original trials | `raw_gemma3_12b_main.jsonl` | 160 | MC + open | 6,096 | 6,096 |

The five models are Gemma-4, Qwen3-VL, Qwen2.5-VL, Gemma-3-4B and LFM2.5. G1 also includes Gemma-3-12B.

## 3. Reproduction checks

**Before the run** (`check_repro.py`): the first ten main-study trials of each model (MC and open, 20 replies) were
replayed and compared with the original replies.

| Model | Identical replies |
|---|---|
| Gemma-3-4B | 20 / 20 |
| Gemma-4 | 19 / 20 |
| LFM2.5 | 20 / 20 |
| Qwen2.5-VL | 20 / 20 |
| Qwen3-VL | 20 / 20 |
| Granite-Vision-4B | 15 / 20 (below the threshold of 18, so not run on this GPU) |

Gemma-3-12B has no original replies to compare with, so `check_det.py` ran its 20 replies twice: 20 / 20 identical.

**After the run:** the full pool contains the 3,048 original 160-target trials. The share of replies identical to the
original run (in the HF dataset as `checks/replay_main_study_on_new_gpu.csv`):

| Model | MC | Open |
|---|---|---|
| Gemma-3-4B | 0.868 | 0.742 |
| Gemma-4 | 0.907 | 0.850 |
| LFM2.5 | 0.962 | 0.948 |
| Qwen2.5-VL | 0.936 | 0.895 |
| Qwen3-VL | 0.923 | 0.849 |

## 4. Problems during the run

No code, prompt, seed or decoding setting was changed to fix any of these.

- **Lost run folder (14 Nov).** The run folder on the node was found empty; the cause is unknown. It was rebuilt: the
  frozen code was restored from the original package (258 of 258 checksums OK). The trial files were rebuilt, and are
  byte-identical to the copies made before the loss. Finished G1 and Gemma-3-12B outputs were recovered from a
  byte-identical copy made for upload. The parts that were lost were rerun: the pre-run checks, the Gemma-3-12B
  determinism check and the 160-target split query.
- **Ended sessions (15, 16 Nov).** The interactive session on the node ended three times, which stopped the running
  jobs. Files were intact each time. The queue was restarted, and each interrupted job resumed from the replies it had
  already saved.
- **GPU memory shared with other users.** Other users held 17–35 GB of the 40 GB GPU, so some jobs ran out of memory.
  After a fix to `ext/jobqueue.py`, a job that runs out of memory waits 5 minutes and retries with its normal memory
  estimate.
- **Upload helper.** `ext/upload_chunk.py` failed on relative paths and was fixed; its upload and secret-scan logic
  did not change. During the run, progress checkpoints were copied to a temporary Hugging Face bucket, which was
  deleted afterwards.
- **Kaggle download.** The Kaggle API returned 404 from the node, so the image archive was uploaded by hand. The
  check that it is byte-identical to version 2 of the Kaggle dataset passed.
- **No LLM judge.** The 31B judge (about 62 GB) does not fit a 40 GB GPU. Open answers that the string matcher could
  not resolve are left unjudged, so open accuracies on the 884 targets are lower bounds.
- **Bytecode caches.** Python wrote `__pycache__` folders into the frozen code folder before
  `PYTHONDONTWRITEBYTECODE` was set. No frozen file changed (checked against the checksums).
