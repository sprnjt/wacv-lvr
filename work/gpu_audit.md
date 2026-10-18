# GPU-path audit — 2026-10-17

Scope: every GPU path in the one RunPod session: `causal_probe.py select/run/summarize` (lfm25v,
qwen3vl), `run_models.py --trials trials_interventions.jsonl --modes mc --out …` (E7/E8 on lfm25v,
gemma3_4b, granite4v), `run_models.py lfm25v --exps E2 --attn`, `analyze_interventions.py`.
Method: I read the code, read transformers 5.20.1 source (`lfm2`, `lfm2_vl`, `qwen3_vl`), and ran the
**real processors/chat templates** plus **tiny random-weight models built from the real configs** on CPU
(`work/cpu_check.py`, no weights downloaded).

Frozen files untouched: `raw_*.jsonl`, `scored_*.csv`, `trials_main.jsonl`, `attn_*.jsonl` (git diff clean).

## Findings

| # | Where (file:line, current) | Severity | Problem | Fix / status |
|---|---|---|---|---|
| C1 | `run_models.py:69-82` (`image_token_ids`/`image_spans`) | **critical** | LFM2-VL tiles any image with h·w > 524,288 px (768×768, 640×768, …): `<|img_row_r_col_c|> <image>×256 … <|img_thumbnail|> <image>×256`. One image became several spans; on the real processor an N=5 trial gave **9 spans**. The attention job would die at the first near-square image (`expected 5 spans`). `causal_probe` had no count check, so `rec_spans[target_pos]` indexed **the wrong tile** and `target_spans[0]` was the first tile only, i.e. silently invalid patches. | **Fixed.** The tile-marker tokens now count as image tokens, so one image gives one contiguous span (`run_models.py:79-81`). `run_cases` now checks that span counts are N/1/1. Verified: 15 E2 trials give 5 spans each, with N=1 length = N=5 length. |
| C2 | `causal_probe.py:196-217` (`logprob`) | **critical** (qwen3vl) | The Qwen3-VL processor returns `mm_token_type_ids` (1, L), which drives M-RoPE. Teacher-forcing appended answer tokens to `input_ids`/`attention_mask` only, so lengths mismatched and the run would crash on case 1 (or get wrong RoPE). | **Fixed** (`:203-207`). It now also extends `mm_token_type_ids`/`token_type_ids` with 0, i.e. text. |
| C3 | `causal_probe.py` old `:247-261` (`wrong_clean_to_target`) | **critical** | The patch needs donor span length = recipient span length, but different images give different token counts (qwen3vl 336/432/576 for 768×432/576/768; LFM 234–256, or 1,280 when tiled). It raised `incompatible image spans` on most cases, killing the job, and resume hit the same case again. | **Fixed.** The wrong donor is now rendered at the target image's pixel size (`wrong_donor_messages`, `:223-233`), so its token grid matches the target slot. |
| C4 | `causal_probe.py:53-110` (`select_cases`) | **critical for qwen3vl n** | Why only 7 qwen3vl cases: the selection needed the *named wrong image* to have its own E1/N=1 trial, i.e. to be a main target. Most wrong images are distractors. qwen3vl: 221 E2/N=5 MC mis-binding fields → 74 with target correct alone → 9 also with wrong∈targets → 7 with wrong correct alone. lfm25v: 783 → 352 → 86 → 60 (so the old `--limit 128` never bound). | **Changed selection (same design).** The wrong donor is now the target's own N=1 prompt with the wrong image swapped in (identical text; only pixels differ). Its "recognized alone" check now runs on GPU as a gate: one greedy MC answer, which must name the wrong image's label for that field. Failures are logged as `wrong_donor_gate_fail` rows and excluded by `summarize`. New candidate sets: **lfm25v 128** (limit binds; 95 targets; 34 of the old 60 kept) and **qwen3vl 74** (28 targets; ≥40 expected after the gate given its N=1 MC accuracy). If the qwen gate leaves <40, add E3/N=5 recipients (+39 fields; same mixed-list N=1 donors; options come after the images, so the image-span states are unaffected). Not implemented. `causal_cases_*.jsonl` regenerated. |
| H1 | `run_models.py:157` | **high** | E7 (variant P5) is describe-first like P4 but got 64 new tokens. Five one-line descriptions plus the answer don't fit, so it would truncate before `Item:/Country:` and E7 would be mass format errors for all 3 models. | **Fixed:** P5 now gets 256, same as P4/E6. |
| H2 | `causal_probe.py:247-266` (auto layer grid) | **high** | The old grid `{0,n/4,n/2,3n/4,n-1}` included the last block. A patch there cannot reach the answer positions, so Δ≡0 by construction (confirmed on CPU). For LFM, layers 28-29 are conv after the last attention layer (27), with a 3-token window. The old LFM grid `0,7,15,22,29` therefore had one null condition, 20% of compute wasted. | **Fixed:** the grid is now evenly spaced up to (last attention layer − 1): **LFM 0,6,13,20,26; qwen3vl 3,11,19,26,34**. An explicit `--layers` beyond that raises. |
| H3 | same; `transformers/models/qwen3_vl/modeling_qwen3_vl.py:833` | **high** (qwen3vl) | Deepstack adds the recipient image's own visual features to the residual after decoder layers 0-2. A patch at layer 0 was partly overwritten, so the wrong donor got contaminated with target features. | **Fixed:** the auto grid starts at `len(deepstack_visual_indexes)` = 3 (0 for LFM). |
| H4 | `run_models.py:143-145` | **high** (data safety) | `run_models.py X --trials trials_interventions.jsonl` without `--out` would **append E7/E8 rows to frozen `raw_X.jsonl`**: new trial ids aren't in the resume set, so nothing stops it. | **Fixed:** `SystemExit` unless `--out` is given for any trials file other than main/pilot. Tested. |
| H5 | `analyze.py:418-430` | high (process) | `analyze.py` `main()` rewrites `results/summary.md`, which has hand-added content (judge hand-check, ~line 729). Running it on the pod to refresh the LFM attention table would destroy that. | **Not run.** `run_all.sh` writes `results/attention_capture_lfm25v.csv` through `analyze.attention()` only. `verify_results.py` (read-only) checks `attn_lfm25v.jsonl` once it exists and expects 1,600 rows, so a partial file reports FAIL by design. |
| M1 | `run_models.py:99-134` (`generate_attn`) | medium (VRAM) | `output_attentions` keeps the step-0 L×L matrix for LFM's 8 attention layers × 32 heads (bf16). That is ≈3 GB at L≈2.4k (typical) but **≈23 GB if all 5 images tile (L≈6.7k)**: OOM on 24 GB cards, and resume can't get past that trial. | Not changed: use a **48 GB** GPU (A40). |
| M2 | `causal_probe.py:236-310` | medium (cost) | Per case: 3 donor captures per layer, plus the 2 unpatched baselines recomputed 3× per layer = **75 forwards**/case. Resume reloaded and recomputed finished cases. | **Fixed:** one capture per prompt for all layers (`capture_layers`), baselines once, and finished or gate-failed cases are skipped: **35 forwards + 1 short generation**/case. The self-control Δ is exactly 0 on CPU (fp32). |
| M3 | design note | medium (interpretation) | LFM layers 0-1 are short-conv (3-token window), so the target span after layer 0 is (almost) context-free. `target_clean_to_target` at layer 0 is ≈0 by construction (exactly 0 in the tiny model). | Report layer 0 as an input-level image swap (the induction contrast), not as a rescue test. |
| M4 | `analyze_interventions.py:37-42` | medium | `ZeroDivisionError` when an experiment has no rows (smoke/partial raw file). There was no format-error count. | **Fixed:** skips empty experiments, NaN when unpaired, and adds a `format_errors` column. Tested on fake raws. |
| M5 | environment | medium | The exact transformers minor version of the main study isn't recorded. Images are re-prepared on the pod (Pillow re-encode). The E7/E8 paired deltas vs frozen E6/E2 assume identical inputs. | `setup.sh` rebuilds `data/` from the Kaggle metadata and requires byte-identical `pool.csv`/`lists.json`. `run_all.sh` runs a **repro step** per model (first 10 main trials × 2 modes, compared with frozen raw; prints `REPRO m: k/20`). Report k next to the deltas. |
| L1 | `ch.py:336-353` | low (design) | The E8 header still says "labeled with its number", and the `[IMG_k]` tags equal the ordinal position. So E8 tests label *format*, not position-independent addressing. | Unchanged (no post-hoc prompt edits). Worth a sentence in the paper. |
| L2 | `ch.py:355-362` | low | P5 in open mode would `KeyError`. Only `--modes mc` is run. | none needed |
| L3 | `ch.py:345-349` | info | E7 lists the options both before the images and in the questions. This is intended ("keep visible"). | — |
| L4 | `ch.py selftest` | info | Needs `dataset/metadata.csv`, which isn't local. | It runs in `setup.sh` on the pod. Locally: rebuilt main trials from `data/` (byte-identical to `trials_main.jsonl`, 3,048), and `make_interventions.py` regenerates `trials_interventions.jsonl` byte-identically (E7 280, E8 800). |
| L5 | LFM config | info | The config carries `lfm2_attention_backend: flash_varlen` etc. (training-side keys, ignored by HF). | Don't install `kernels`/`causal_conv1d`; keep the reference conv path the main study used. |

Checked and fine: `--modes`/`--out`/`--exps` parsing, and resume keys `(trial_id, mode)` and
`(case_id, layer, intervention)`. Hook path `model.language_model.layers` is right for both models, and
decoder layers return a tensor (handled). LFM hybrid layers are fine for residual patching. LFM's
`output_attentions` only records `Lfm2Attention` modules, so the conv layers don't produce None entries.
Gold/wrong letters come from the recipient's options, which equal the N=1 donor's options (same
`(target, list)` seed). E7/E8 trial ids and the analysis back-mapping (E7↔E6 P5↔P4, E8↔E2) are correct.
granite4v and gemma3 resolve through `AutoModelForMultimodalLM` in 5.20.1. No output file name
collides with a frozen file.

## Edits (all minimal)

- `run_models.py`: `import re`. The LFM tile markers count as image tokens (`:79-81`). Optional
  `messages=` in `prep_inputs` (`:54`). `--out` guard against frozen raw files (`:143-145`). P5 gets
  256 tokens (`:157`).
- `causal_probe.py`: selection without the wrong-image-must-be-a-target requirement (`select_cases`).
  `capture_layers` (one pass for all layers). `logprob` extends `mm_token_type_ids`/`token_type_ids`.
  New `wrong_donor_messages` (swap + resize). `run_cases` rewritten: span-count checks, wrong-donor gate,
  baselines once, resume skips finished cases, auto grid (deepstack start, stops before the last
  attention layer), per-case timing. `summarize` skips gate rows and reports the gate-fail count.
  Docstring examples updated.
- `analyze_interventions.py`: empty-experiment guard, NaN when unpaired, `format_errors` column.
- `causal_cases_lfm25v.jsonl` (60→128), `causal_cases_qwen3vl.jsonl` (7→74): regenerated by `select`.
- `README.md`: causal-extension paragraph describes the swapped/resized donor, the gate and the layer grid.
- New: `runpod/{setup.sh,run_all.sh,collect.sh,check.py}`, `work/cpu_check.py`, `work/runbook.md`, this file.
- Log these post-freeze changes in `data/pilot_gate_report.md` (post-freeze log). It ships inside `data.zip`, so I left it alone.

## CPU verification run (all pass)

Run: `/Users/shinjini/Desktop/wacv/.venv-cpu-check` (Python 3.11, torch 2.14.1 CPU, transformers
5.20.1), `cd <scratch with data/ symlink> && python work/cpu_check.py <repo>` → `CPU CHECK OK`:
- Real LFM2.5-VL-3B and Qwen3-VL-8B processors on fake images of mixed sizes: 5 spans per N=5 trial,
  and N=1 span length = N=5 span length.
- Tiny-model `generate_attn` through `run_models.main(..., attn=True)`: finite, positive per-image mass.
- Tiny-model `causal_probe.run_cases` end to end: gate pass and fail paths, resume is a no-op,
  self-control |Δ| = 0, rescue Δ ≠ 0 at every grid layer > 0, qwen grid starts after deepstack,
  `summarize` OK.
- `runpod/check.py` (repro/causal/attn/mc) on real and fake files. `run_all.sh` dry-run with a stubbed
  GPU command (flow, skip markers, failed-check → re-smoke).
- `~/.kaggle/kaggle.json`: **does not exist** on this Mac, so Kaggle credentials must be supplied for the pod.

## E9 collage (post-freeze, exploratory, MC only) — added 2026-10-17

Question: does binding fail across separate images or within one scene? 160 main targets × 4 conditions =
640 MC trials (`trials_collage.jsonl`, `make_collage.py trials`, seed 2026, deterministic). Each target gets 3
companions (one per other category, 4 distinct countries, never a pilot/main target item) in a 2×2 768×768
collage; target slot = idx % 4. C1 `solo` (only the target panel, query by position), C2 `separate` (the 4
letterboxed 376×376 panels as 4 images, "Image k"), C3 `collage_pos` ("the top-left panel"), C4
`collage_kind` ("the panel showing the <kind>"). Options: 4 shown items + 1 unshown same-category decoy, 4
shown countries + the decoy's country, fixed per target, gold item letter ≠ gold country letter.

Code: `ch.make_parts` uses E9's own `head`/`ref`/`noun` and `images` fields when present (all 16,336 existing
prompts are byte-identical, checked). `run_models.build_messages` resolves `collage/<id>.png` under `images/`.
`analyze_collage.py` → `results/collage_<m>.csv`. `run_all.sh collage MODEL`: smoke (4 trials) → `check.py mc`
→ `raw_<m>_collage.jsonl` → analysis. `collect.sh` packs the new files.

Design compromises: (1) Question 1 says "item" instead of the category in **all** conditions, because the
main-study wording names the kind and would make C1–C3 as kind-cued as C4. (2) The images are **PNG**, not
JPEG, so a C2 panel has exactly the pixels of its collage cell. (3) The unshown country is the decoy's country
(as in the main study's absent image). (4) LFM tiles the 768×768 collage (1,285 image tokens) but not the
376×376 panels (4 × 144 = 576 in total), so for LFM C2 vs C3 also differs in visual token budget. Qwen3-VL
uses 576 tokens in every condition. (5) C4's item field is solvable by kind: read country and ICR there.

CPU check (`work/cpu_check_e9.py`, all in a temp dir outside the repo, ends `E9 CPU CHECK OK`): trials
reproduce byte-identically. 960 images are built deterministically from fake prepared images, and C2 panels
equal the collage cells pixel for pixel. Prompts and paths are checked for all 640 trials. Scoring: oracle all
correct, a copied item letter is always a wrong country, other-panel and decoy answers are classified
correctly. The real LFM and Qwen3-VL processors give 1 span for C1/C3/C4 and 4 for C2. Tiny-model generation
goes through `run_models.main`, with resume and the `--out` guard tested. `analyze_collage` gives the known
values on a constructed raw file and also runs on a partial one. A stubbed `run_all.sh` passes (failed smoke
→ re-smoke). `collect.sh` passes with a failing HF upload and with a token-in-file guard, and never prints the
token. The repo's git status and frozen-file hashes are unchanged.

GPU time (per-call costs above, ×2 for A40): lfm25v 640 × 0.52 s ≈ 6 min, gemma3_4b × 0.76 ≈ 8, granite4v
× 0.70 ≈ 8, qwen3vl × 1.54 ≈ 16. That's ≈ 38 min, plus loads and smoke ≈ **0.75 h planning, ≈ 1.5 h
pessimistic**.
