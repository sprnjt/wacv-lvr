# CultureHaystack

A diagnostic benchmark for **image–attribute binding in multimodal LLMs**: show a model N photographs of
cultural items (foods and drinks, landmarks, instruments, attire) from different countries, ask it to name
the item and country of image k, and measure how identity and origin come unbound as N grows from 1 to 5.
Design and analysis plan: `plan.md`. Built for the LVR @ WACV 2027 workshop (Latent Visual Reasoning).

**Status (2026-10-09): main study complete.** 6 models × 6,096 calls, all judged and analyzed;
tables in `results/` (17 tables + `summary.md`), figures in `results/figures/`, full write-up in
`paper.md` §8. `verify_results.py` recomputes every headline number from the raw logs without
importing the analysis code — all checks pass (max |diff| 0.0000).

## Headline results

- **Clean split by size class.** MC joint accuracy is flat N=1→5 for the ≥7B models (gemma4
  0.856→0.801, qwen3vl 0.831→0.807, qwen25vl 0.694→0.753) but drops 0.13–0.22 for the ≤4B models
  (gemma3_4b 0.562→0.434, granite4v 0.544→0.320, lfm25v 0.556→0.315; slopes −0.04…−0.06/image,
  CIs exclude 0).
- **Gradual, not a step** (H1 rejected everywhere); **U-shaped slot profile** at N=5 in exactly the
  three degrading models (H2, Holm p ≤ 0.008); **near distractors hurt more than far** beyond CLIP
  similarity (H3b supported for gemma3_4b/lfm25v MC; direction unanimous in MC).
- **Errors are whole answers lifted from one wrong shown image**: illusory-conjunction rate at or
  below the independence baseline in all 48 cells; ≥92% of wrong MC fields at N=5 are mis-binding
  (read against option availability — H3a above chance only for lfm25v).
- **Not a readout-chain failure**: describe-then-answer never rescues (hurts gemma3_4b and
  granite4v MC), and answer-token attention shows no capture of the named wrong image in 5 of 6
  probed cells (probe run on qwen3vl, gemma4, granite4v).
- **Open-ended naming is at floor even at N=1** (joint ≤ 0.30) — binding claims rest on MC.
- Validity: text-only leak ≈ chance, paraphrases indistinguishable, repeats 20/20 × 6, judge
  hand-check 95/100. Details: `results/summary.md`, `paper.md` §8.

## Data

- Source: [World Cultural Items 1k](https://www.kaggle.com/datasets/shinjinichakraborty/world-cultural-items-1k),
  accessed 2026-10-08 (1,107 rows, 22 columns). Cite version and access date.
- After filtering (single country, curated–Wikidata agreement, dedup, ≥384 px): **890 images, 123 countries,
  166 targets** (6 pilot + 160 main, 40 per category). Counts and label gate: `data/audit.txt`,
  `data/disputed_labels.txt`.
- Manual audits before any run: label audit of 6 pilot + 24 random main targets against Wikidata
  (**30/30 consistent**, script `audit_sample.py`, output `data/label_audit.txt`); visual review of all
  images via contact sheets → 17 rejected (readable item names/captions/diagrams) in `exclude.txt`,
  applied as the last build filter.
- Attribution for every image is in `data/pool.csv` (license, author, source URL). Release the CSV, scripts
  and seeds — not the images.

## Setup

```bash
pip install -U transformers accelerate pillow torch pandas numpy scipy statsmodels kaggle
kaggle datasets download -d shinjinichakraborty/world-cultural-items-1k -p dataset
unzip -o dataset/world-cultural-items-1k.zip -d dataset
```

This host: the system cuDNN in `LD_LIBRARY_PATH` shadows the wheel-bundled one and crashes torch.
Prefix every GPU command with `env -u LD_LIBRARY_PATH`.

Models (`MODELS` in `run_models.py`; all 6 smoke-tested end to end on 2026-10-08: load, generate in the
required format, and score joint-correct on pilot N=1,2):

| Key | HF repo | Notes |
| --- | --- | --- |
| `qwen3vl` | Qwen/Qwen3-VL-8B-Instruct | |
| `qwen25vl` | Qwen/Qwen2.5-VL-7B-Instruct | |
| `gemma4` | google/gemma-4-12B-it | `enable_thinking` defaults off — `TEMPLATE_KWARGS` stays `{}` |
| `gemma3_4b` | google/gemma-3-4b-it | gated; access granted via HF token (2026-10-08) |
| `granite4v` | ibm-granite/granite-vision-4.1-4b | tends to add parentheticals in open mode; scorer handles it, watch the pilot format check |
| `lfm25v` | LiquidAI/LFM2.5-VL-3B | `Lfm2VlForConditionalGeneration` (transformers 5.x); runs on the reference `causal_conv1d` path (slower, correct) |

Dropped after smoke: `microsoft/Phi-3.5-vision-instruct` (remote code incompatible with transformers 5.x
attention/cache APIs beyond a reasonable shim) and `OpenGVLab/InternVL3-8B-Instruct` (old config format;
locally converted native copy produced degenerate output — not trusted for the study).
Dropped after pilot, pre-freeze: `HuggingFaceTB/SmolVLM2-2.2B-Instruct` — open mode was clean, but MC
format errors were 29/96 → 69/96 → 91/96 across three instruction variants (plan-literal, +clause+example,
+clause), including example parroting and bare-letter collapse. MC is half the design, so the model was
removed and replaced with LFM2.5-VL-3B (see `data/pilot_gate_report.md`).
Excluded from the main runs after the first freeze, before its main run started: `google/gemma-3-12b-it`
(pilot data retained in `pilot_out/`; roster decision, not data-driven — see `data/pilot_gate_report.md`).

Attention capture requires eager attention: `--attn` switches the model with
`set_attn_implementation("eager")` (SDPA returns no attention weights), which is slower per call.
Sliding-window layers (gemma4) return only the trailing attention window; the probe left-pads those
rows to each generation step's source length (post-freeze fix, logged in `data/pilot_gate_report.md`).
6 models × 6,096 calls = 36,576 main-study calls (plus pilots, repeats and the attention pass).

## Pipeline

```bash
python3 ch.py selftest --meta dataset/metadata.csv   # must end with SELFTEST OK
python3 ch.py build-data --meta dataset/metadata.csv # -> data/ (pool, targets, lists, audit)
python3 ch.py prep --src dataset                     # -> images/prepared/ (890 neutral, 768 px, EXIF-stripped)
python3 ch.py trials pilot                           # 96 trials
python3 ch.py trials main                            # 3,048 trials (E1 640, E2 800, E3 608, E5 560, E6 280, leak 160)

env -u LD_LIBRARY_PATH python3 run_models.py qwen3vl --trials trials_pilot.jsonl [--repeat]
python3 ch.py score qwen3vl --trials trials_pilot.jsonl
env -u LD_LIBRARY_PATH python3 judge.py      # LLM judge fills judge_overrides.csv from review_*.csv; rescore after
# ... pilot pass/fail gate (plan.md §5), freeze (prereg_snapshot/ + PREREG_MANIFEST.sha256), then:
env -u LD_LIBRARY_PATH python3 run_models.py qwen3vl --trials trials_main.jsonl [--repeat]
env -u LD_LIBRARY_PATH python3 run_models.py qwen3vl --trials trials_main.jsonl --exps E2 --attn   # attention probe
env -u LD_LIBRARY_PATH python3 clip_sim.py
python3 ch.py score qwen3vl --trials trials_main.jsonl
python3 analyze.py qwen3vl qwen25vl gemma4 gemma3_4b granite4v lfm25v   # -> results/ incl. locus_discrimination.md
python3 make_figures.py                              # -> results/figures/ (fig1-fig5, PDF + PNG)
python3 verify_results.py                            # independent audit of results/ vs raw data; exits 1 on mismatch
```

## LVR additions (beyond plan.md)

Decided pre-registration; `plan.md` is unchanged and remains the primary pre-registration document.

1. **E6 — describe-then-answer (text-mediation probe).** Same 56 targets as E5, N = 1–5, variant P4:
   the model first describes each image in one line, then answers. 280 extra trials (main total 3,048;
   6,096 calls per model across both answer modes). P4 trials get 256 new tokens. If verbalizing first
   rescues accuracy, the binding survived in the latent state and the failure is in latent→text readout;
   if not, binding fails earlier. Analysis: `results/describe_then_answer.csv`.
2. **Attention-capture probe.** `run_models.py --attn --exps E2` writes `attn_<model>.jsonl`: for every
   generated answer token, the attention mass (averaged over layers, heads and answer tokens) landing on
   each shown image's token span. Test: among mis-binding errors, does the *named* wrong image carry the
   highest attention mass? Analysis: `results/attention_capture.csv`.
3. **Locus-discrimination table.** `results/locus_discrimination.md` maps each observed signature onto
   three candidate binding loci. The predictions below are **pre-registered** (fixed in
   `analyze.py:LOCUS_ROWS`); only the Observed column is data:

| Signature | Encoder-bound | Latent scene state | Text-chain readout |
| --- | --- | --- | --- |
| Step at N=1→2 (H1) | flat | sharp step | gradual decline |
| Middle slots worse at N=5 (H2) | no slot effect | U-shape | possible — weak test |
| Near > far, CLIP controlled (H3b) | no | yes | yes — weak test |
| Illusory conjunctions > baseline | no | yes | no (whole answer copied) |
| Describe-then-answer rescues (E6) | no | partial | yes |
| Attention peaks on named wrong image | no (stays on target) | yes | no commitment |

## Scoring with the LLM judge

`ch.py score` resolves answers mechanically (aliases, no containment matching); open answers it cannot
resolve land in `review_<model>.csv`. `judge.py` labels those with **google/gemma-4-31B-it** (text-only,
greedy): the judge sees only the gold answer and the model's answer string — never the images and never
which model produced the answer. Verdicts go to `judge_overrides.csv` (per-model; rerun `score` to apply);
every prompt and verdict is logged in `data/judge_log.jsonl`. Judge model and prompt are pinned here for
the pre-registration. Because gemma-4-31B-it shares the gemma family with 2 of the 6 evaluated models, we
hand-checked a random 100-row subsample of main-study judged rows (`data/judge_sample_100.jsonl`):
**95/100 agreement** (4 false accepts, 1 false reject — reported in `results/summary.md`).

## Limitations and deferred work

- **Open mode is at floor.** Open-ended naming tops out at 0.30 joint even at N=1 with no N-decline,
  so every binding conclusion rests on the multiple-choice mode; open mode is a knowledge/naming
  ceiling diagnostic only.
- **MC error anatomy is availability-driven for 5 of 6 models.** H3a shows wrong MC choices are
  ~uniform over the wrong options (only lfm25v is attracted to shown images above chance), so the
  rising mis-binding share with N — and the ~100% share at N=5, where all options are shown — must
  be read against the shown-option rate.
- **Attention probe covers 3 of 6 models** (qwen3vl, gemma4, granite4v); the single above-chance
  capture cell (granite4v open) rests on 39 mis-binding fields.
- **No causal intervention (deferred).** Attention capture is correlational. The decisive LVR experiment —
  activation patching between target and distractor image-token spans to flip the answer — is deferred to
  camera-ready or follow-up work.
- **Model coverage (deferred).** Results are per-model behavioral signatures; adding a third model family
  for cross-architecture convergence is deferred. Add entries to `MODELS` in `run_models.py` to extend.
- **Mixed-effects models (deferred).** `analyze.py` uses logistic GLMs with target-clustered SEs, not
  crossed random effects; refit final models with `glmer` (R) for the camera-ready.
- Pool size (890 images / 160 targets) widens confidence intervals; bootstrap CIs reported throughout.
- Images come from Wikimedia Commons and have likely been seen in training; N = 1 accuracy is a
  knowledge baseline, and all claims concern degradation at N ≥ 2, not memorization.
- Country of origin is a coarse proxy for culture; curated labels were cross-checked against Wikidata
  (8 disputed rows dropped) and hand-audited on a sample.
- No generic-object control (H4 of the original design): whether the effect is culture-specific is untested.
