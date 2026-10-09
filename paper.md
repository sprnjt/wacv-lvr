# CultureHaystack: A Diagnostic Benchmark for Image–Attribute Binding in Multimodal LLMs

**Status (2026-10-09):** complete. All 6 models ran the full main study (6,096 calls each, plus
repeats and the attention pass); all analysis tables are in `results/` (regenerable with
`analyze.py`), all figures in `results/figures/` (`make_figures.py`), and `verify_results.py`
independently recomputes every headline number from the raw logs (all checks pass, max |diff|
0.0000). The primary pre-registration document is `plan.md`; the freeze of record is
`prereg_snapshot/` + `PREREG_MANIFEST.sha256`.

Target venue: **LVR @ WACV 2027** (Latent Visual Reasoning workshop).

---

## Abstract

We introduce **CultureHaystack**, a diagnostic benchmark that isolates *image–attribute binding* in
multimodal large language models: given N photographs of culturally specific items (foods and
drinks, landmarks, musical instruments, traditional attire) from different countries, the model must
name the item and the country of origin of the k-th image. Because every image is individually
recognizable (N = 1 accuracy provides a knowledge baseline), any degradation as N grows from 1 to 5
is attributable to binding — keeping *which* item belongs to *which* image straight — rather than to
recognition. The design uses nested haystacks (each N is a subset of N + 1), fixed multiple-choice
option sets across N, a four-way error taxonomy (mis-binding, prior drift, abstain, format error),
and an illusory-conjunction analysis that separates whole-answer swaps from attribute-level binding
failures. Three pre-registered probes target the *locus* of binding: a scaling/step analysis, a
describe-then-answer text-mediation experiment, and answer-token attention capture. Six open-weight
VLMs from five architecture families are evaluated under a frozen protocol with an LLM-judge
adjudication layer. Results: multiple-choice joint accuracy is flat from N = 1 to N = 5 for the
three ≥ 7B models (0.69–0.86 throughout) but drops 0.13–0.22 absolute for the three ≤ 4B models.
The decline is gradual (no N = 1→2 step anywhere), U-shaped across slots at N = 5 in exactly the
degrading models, amplified by culturally near distractors beyond visual similarity, and consists
of whole answers lifted from one wrong shown image — the illusory-conjunction rate is at or below
the independence baseline in every model × mode × N cell. Two probes fail to localize the failure
in the text readout chain: describe-then-answer prompting never rescues accuracy (it significantly
hurts two models), and answer-token attention shows no capture of the named wrong image in 5 of 6
probed model × mode cells. Open-ended naming of culturally specific items is at floor even at
N = 1 (joint ≤ 0.30), so all binding claims rest on the multiple-choice mode.

---

## 1. Motivation and research question

Multimodal LLMs fail in a characteristic way when a prompt contains several images: they answer
about the wrong one. Text-side "needle in a haystack" studies documented an analogous failure for
long context; CultureHaystack is its visual counterpart, but with two tightening twists:

1. **Binding, not retrieval.** Each image requires *two* attributes (identity and country of
   origin). A model can retrieve the right item yet bind the wrong country to it, or retrieve the
   wrong image entirely. The joint answer decomposes the failure.
2. **Recognition is controlled.** All items are real, named cultural objects (baklava, bansuri,
   Chartres cathedral) that models demonstrably know at N = 1. Degradation at N ≥ 2 therefore
   measures interference among shown images, not ignorance.

**Central question.** As the number of shown images grows from 1 to 5, how do identity and origin
come unbound, what do the errors look like, and *where* in the model does the binding live — in the
per-image encoding, in a latent scene state, or in the textual readout chain?

The workshop-level contribution is the design and analysis methodology: nested haystacks, fixed
option sets, a scored error decomposition, illusory conjunctions against an independence baseline,
and a pre-registered locus-discrimination table. The image pool itself is an existing public
dataset.

---

## 2. Hypotheses and decision rules (pre-registered)

Open-ended answering is the headline mode for H1, H2 and H3b; multiple choice is the replication.
H3a is tested in multiple choice only (all five options are on the page at every N, so
context-attraction is measurable there by construction). Holm correction is applied across
model-by-mode rows *within* each hypothesis.

| ID | Hypothesis | Supported if | Rejected if |
| --- | --- | --- | --- |
| **H1** | Most of the accuracy loss occurs between N = 1 and N = 2 | Step model beats linear on AIC **and** the step term passes a likelihood-ratio test (Holm-adjusted) | Linear wins (gradual dilution), or no decline |
| **H2** | Middle slots are worse than end slots at N = 5 | Positive quadratic slot term, Holm *p* < 0.05 | Flat, or monotone in slot |
| **H3a** | Wrong choices are attracted to *shown* images | Excess over the (N−1)/4 chance rate has a bootstrap CI above zero at N = 2, 3 and 4 | Excess at or below zero at some N |
| **H3b** | Culturally *near* distractors hurt more than *far* ones, beyond visual similarity | Negative near coefficient, Holm *p* < 0.05, with CLIP image similarity in the regression | No effect once CLIP similarity is controlled |

**Why H3 is split.** In multiple choice, all five option names are visible at every N, so wrong
choices naming a shown distractor rise with N almost by construction. H3a therefore asks whether
wrong choices pick shown distractors *more often than the chance rate* of (N−1)/4 among the four
non-target options. H3b asks the substantive question — whether cultural (regional) proximity
drives confusion after controlling for visual look-alike-ness.

**Illusory conjunctions (descriptive, two-sided).** The Illusory Conjunction Rate (ICR) is the
share of wrong joint answers whose item and country come from *different* shown images. It is read
against an independence baseline that shuffles country answers among trials with the same target
slot (preserving each field's own error pattern while breaking the binding):

- ICR **above** baseline → item and country are bound to images *separately*; attributes detach and
  recombine across images (attribute-level binding failure).
- ICR **below** baseline → the model lifts a *whole* answer from one wrong image (object-level
  selection failure).

**Popularity tiers (descriptive).** Accuracy and prior-drift share are reported by the dataset's
popularity tier (high/mid/low, roughly equal thirds, from Wikipedia sitelink counts). Higher prior
drift on low-tier items would indicate models leaning on parametric priors for rarer items.

### Interpretation matrix (pre-registered conclusions)

| Result | Conclusion licensed |
| --- | --- |
| H1 supported | The failure begins when a second image appears |
| Linear wins | Binding erodes gradually with every added image |
| H2 supported | End slots handled better than middle ones (primacy/recency) |
| H3a supported | Errors are mostly misattribution among shown images |
| H3a not supported | Wrong choices ≈ option-level guessing, not context-driven |
| H3b supported | Cultural closeness, not just look-alike photos, drives confusion |
| ICR above baseline | Item and country bound separately |
| ICR below baseline | Whole answers taken from one wrong image |
| Low-tier prior drift higher | Models lean on prior knowledge for rarer items |

---

## 3. Data

### 3.1 Source

[World Cultural Items 1k](https://www.kaggle.com/datasets/shinjinichakraborty/world-cultural-items-1k)
(Kaggle), accessed **2026-10-08**: 1,107 images of culturally specific items from Wikimedia Commons,
one representative image per item, each linked to a Wikidata entity and English Wikipedia article.
Domains: landmark 256, dessert 200, food 200, instrument 174, drink 148, attire 129. The 22-column
metadata includes `name`, `aliases`, `countries`, `regions`, `tier` (popularity), `sitelinks`,
`country_answers` (accepted country spellings), `wikidata_countries`, and full attribution
(`license`, `author`, `source_url`). The dataset version and access date are cited because the row
count changed between releases (the page text still said 1,047).

### 3.2 Filtering funnel (label and quality gates)

| Step | Rows left |
| --- | --- |
| Rows in metadata | 1,107 |
| Single-country rows (125 shared-heritage rows removed) | 982 |
| After keeping the 4 benchmark categories (food+dessert+drink merged into "foods and drinks") | 982 |
| **Label gate:** rows where curated `countries` disagrees with `wikidata_countries` dropped (8 rows: Espresso, Cheongsam, Aso Oke, Shehnai, Angklung, Red Fort, Summer Palace, Burana Tower — listed in `data/disputed_labels.txt`) | 974 |
| After dropping rows sharing one Commons file (3 duplicate-photo pairs) | 968 |
| After requiring short side ≥ 384 px | 907 |
| **Visual audit:** 17 images rejected by eye (readable item names, captions, diagrams) via contact sheets, listed in `exclude.txt` | **890** |

The label gate is deliberately conservative: several dropped rows are probably fine (e.g. Angklung →
Indonesia), but the rule is objective and free. Final pool: **890 images, 123 countries** — foods
and drinks 448 (76 countries), landmark 246 (79), instrument 110 (53), attire 86 (39). Region skew
of the 160 main targets (reported, not hidden): Asia 59, Europe 52, North America 19, Africa 17,
South America 11, Oceania 2; Asia and Europe dominate the pool itself.

### 3.3 Targets

**166 targets = 6 pilot + 160 main** (40 per category). Pilot targets are drawn from the 20 most
widely documented high-tier items per pilot category (by sitelinks; country-distinct; seed 2026) and
are **kept out of the main targets**, so main-study claims never depend on pilot-target properties:

| Target | Item | Country | Category |
| --- | --- | --- | --- |
| img_0462 | Lasagna | Italy | Foods and drinks |
| img_0094 | Baklava | Turkey | Foods and drinks |
| img_0544 | Bansuri | India | Instrument |
| img_0619 | Sheng | China | Instrument |
| img_0789 | Notre-Dame de Chartres | France | Landmark |
| img_0875 | Tower of London | United Kingdom | Landmark |

Main targets are tier-stratified (roughly 13 high / 13 mid / 14 low per category) with a per-country
cap (≤ 3 targets per country per category on the first pass) for diversity. A **near-list
feasibility** flag marks targets whose category × region cell offers ≥ 4 other countries (152 of 160
qualify: 39/40 food, 39/40 instrument, 36/40 landmark, 38/40 attire); only these enter E3.

### 3.4 Manual audits (both completed before any model run)

- **Label audit** (`audit_sample.py`, output `data/label_audit.txt`): the 6 pilot targets plus 24
  randomly chosen main targets (15%) checked by hand against their Wikidata/Wikipedia entries —
  **30/30 consistent**.
- **Visual audit**: every prepared image reviewed in contact sheets of ~40 (`contact_sheets/`); the
  17 rejections in `exclude.txt` were applied as the last build filter, after which targets and
  image IDs were regenerated.

### 3.5 Image preparation and licensing

Every image receives a neutral ID (`img_0001`…), is resized so the long side ≤ 768 px, flattened to
RGB on white, and re-saved as JPEG — which strips EXIF. The original `file_name` encodes the answer
(`des_baklava_Q187495.png`) and must never reach a model; prompts reference only neutral IDs, and
text-only leak trials verify the channel is closed (§4.5).

Images are **not redistributed**: `data/pool.csv` carries license, author and source URL for every
image (pool license mix: CC BY-SA 3.0 ×248, CC BY-SA 4.0 ×246, CC BY 2.0 ×95, public domain ×81,
CC BY-SA 2.0 ×80, CC0 ×37; 31 rows lack an author field — `source_url` is the attribution
reference). The release is the CSV, scripts, and seeds — not the images.

---

## 4. Benchmark design

### 4.1 Haystack construction

For each target, a fixed 5-image set is built once: the target plus 4 distractors, always **5
different countries and 5 different items within one category**. Distractor lists come in three
types:

- **mixed** — random other countries of the same category (the default);
- **near** — same world region (cultural proximity);
- **far** — different regions.

The haystack at size N is the target plus the first N−1 distractors, so **every N is a subset of
N+1** (nested design): adding one image is the only thing that changes between adjacent cells.
The target occupies a specified slot k; slot assignment is balanced per target across N by a fixed
rotation (`idx % N + 1`).

### 4.2 Fixed option sets

The multiple-choice options are the 5 item names and 5 country names of the target's 5-image set,
shuffled independently with a fixed seed (`2026-opt-<target>-<list>`) and **identical at every N**.
Consequences:

- At N < 5, some options name images *not shown*; choosing one counts as **prior drift**, which
  makes prior drift measurable in MC mode.
- Option-position artifacts are constant across N, so Acc@N curves are not confounded by option
  composition.

### 4.3 Answer modes and prompt

Every trial is asked twice, as a fresh request each time: **open-ended** and **multiple choice**.
Decoding is greedy throughout. Canonical prompt (open mode, N = 3, target in slot 2, landmark):

```text
Below are 3 photographs of landmarks from various countries.
Each photograph is labeled with its number.

Image 1: <photo>
Image 2: <photo>
Image 3: <photo>

Look only at Image 2 and answer two questions.
1. What is the name of the landmark shown in Image 2?
2. Which country is it from?

Answer in exactly this format and nothing else:
Item: <name>
Country: <country>
If you cannot tell, write Unknown for that field.
```

The MC variant lists the two questions with lettered options A–E and demands `Item: <letter>` /
`Country: <letter>`. Category nouns are templated ("foods and drinks", "musical instruments",
"landmarks", "traditional garments"). Paraphrase variants P2/P3 (E5) reword the question stem; P4
(E6) prepends "First describe each photograph in one short line, in order" and moves the format
demand to the end of the reply.

### 4.4 Experiments

Per model, per answer mode (trial counts as built; `trials_main.jsonl` = 3,048 trials):

| ID | Setup | Trials |
| --- | --- | --- |
| **E1** | 160 targets × N = 1–4, mixed list, balanced slot | 640 |
| **E2** | 160 targets × N = 5, target in every slot (also the N = 5 cell of E1) | 800 |
| **E3** | 152 near-feasible targets × {near, far} × N ∈ {2, 5}, balanced slot | 608 |
| **E5** | 56 targets (every third) × paraphrases P2, P3 × N = 1–5, mixed | 560 |
| **E6** | Same 56 targets × describe-then-answer (P4) × N = 1–5, mixed | 280 |
| **Leak** | 160 targets, MC/open questions with **no images** | 160 |
| **Total** | | **3,048** |

Both answer modes make **6,096 calls per model** (36,576 across the 6-model roster), plus repeat
checks and the attention pass. The **pilot** is 96 trials (6 targets × N = 1–5 × every slot = 90,
plus 6 leak trials), i.e. 192 calls per model plus 20 repeat calls.

**E6 — describe-then-answer (LVR text-mediation probe, pre-registered addition).** The model first
describes each photograph in one line, then answers (P4 trials get 256 new tokens vs 64 elsewhere).
If verbalizing first *rescues* accuracy, the binding survived in the latent state and the failure is
in latent→text readout; if not, binding fails earlier. Analysis: `results/describe_then_answer.csv`
(rescue = P4 accuracy − P1 accuracy on the same targets, bootstrap CI over targets).

### 4.5 Leak control

Leak trials ask the full question with no images. Pre-registered ceilings: MC ≤ 3/6 correct per
question, open ≤ 1/6 (pilot). With neutral IDs, stripped EXIF, and option order fixed by seed, no
leak channel exists; residual open-mode country hits are famous-item stereotypes (Pho→Vietnam),
which the ceiling absorbs.

---

## 5. Models

Final roster — 6 open-weight VLMs spanning 5 architecture families and 3–12B parameters
(`MODELS` in `run_models.py`); all six smoke-tested end-to-end on 2026-10-08 (load, generate in the
required two-line format in both modes, score joint-correct on pilot N = 1, 2):

| Key | HF repo | Notes |
| --- | --- | --- |
| `qwen3vl` | Qwen/Qwen3-VL-8B-Instruct | |
| `qwen25vl` | Qwen/Qwen2.5-VL-7B-Instruct | |
| `gemma4` | google/gemma-4-12B-it | `enable_thinking` defaults off — template kwargs stay empty |
| `gemma3_4b` | google/gemma-3-4b-it | gated repo; access via HF token |
| `granite4v` | ibm-granite/granite-vision-4.1-4b | parenthetical-prone in open mode; scorer handles it; 16/96 pilot MC format errors stand as a documented limitation |
| `lfm25v` | LiquidAI/LFM2.5-VL-3B | hybrid conv+attention (LFM2); `Lfm2VlForConditionalGeneration` on transformers 5.x, reference `causal_conv1d` path (slower, correct) |

**Dropped, with reasons (all pre-results):**

- `microsoft/Phi-3.5-vision-instruct` — remote code incompatible with transformers 5.x
  attention/cache APIs beyond a reasonable shim (dropped after smoke).
- `OpenGVLab/InternVL3-8B-Instruct` — old config format; a locally converted native copy produced
  degenerate output and was not trusted for the study (dropped after smoke).
- `HuggingFaceTB/SmolVLM2-2.2B-Instruct` — open mode clean, but MC format errors were
  29/96 → 69/96 → 91/96 across three instruction variants (example parroting, then bare-letter
  collapse). MC is half the design, so it was removed pre-freeze and replaced by `lfm25v`, keeping
  the small-model slot and adding a fifth family.
- `google/gemma-3-12b-it` — excluded from the main runs after the first freeze but before its main
  run started and before any main results existed: redundancy with `gemma4` at the same size class,
  and cutting it reduces the judge's gemma-family overlap to 2 of 6 evaluated models. Pilot data
  retained in `pilot_out/`.

**Decoding.** Greedy (`do_sample=False`), one fresh request per trial and mode, 64 max new tokens
(256 for P4). The runner resumes after interruption (skips `(trial_id, mode)` pairs already in the
output file). Per-model records (GPU, library versions, revision, time per call, failures) are kept
with the run outputs.

---

## 6. Scoring and adjudication

### 6.1 Mechanical scorer

`ch.py score` parses the two answer lines and resolves each field against the gold name and its
aliases. Normalization is Unicode NFKD → ASCII-fold → lowercase → strip articles. Matching is
*exact* for the canonical name and every alias, plus the canonical name as **whole words** inside a
longer answer; **aliases never match by containment** (avoids false hits like "America" or "cake").
Country aliases come from the dataset's `country_answers` column plus a small hand table (UK/Britain,
Turkiye, Czechia, …). `source_image` resolves *which* of the 5 set images an answer names (the
target wins ties), which drives the error taxonomy:

| Error type | Meaning |
| --- | --- |
| `mis_binding` | Names another image **shown** in this trial |
| `prior_drift` | Names something **not shown** (including an unshown set member in MC) |
| `abstain` | "Unknown" / empty / equivalent |
| `format_error` | No parseable answer line |

A joint answer is correct only if both fields are correct. **Illusory conjunction**: wrong joint
answer whose item and country resolve to *different shown* images.

### 6.2 LLM judge

Open answers the mechanical scorer cannot resolve land in `review_<model>.csv` and are labeled by a
pinned judge — **google/gemma-4-31B-it**, text-only, greedy (`judge.py`). The judge sees only the
gold answer and the model's answer string — **never the images, never which model produced the
answer**. It accepts synonyms, translations, alternate spellings and well-known equivalents, and
rejects broader/narrower categories ("pasta" for "lasagna"), different specific items, and different
countries including historical predecessors. One-word verdicts (`CORRECT`/`INCORRECT`) go to
`judge_overrides.csv` (applied per model on rescore; an `incorrect` override on a mechanically
correct match flips it to prior drift); every prompt and verdict is logged in
`data/judge_log.jsonl`. Pilot scale: 371 rows judged for the first 7 models (71 correct / 300
incorrect) + 53 rows for the lfm25v re-pilot (17 rescued as correct).

**Judge-bias mitigation.** Because the judge shares the gemma family with 2 of the 6 evaluated
models, a random 100-row subsample of main-study judged rows was hand-checked: **95/100
agreement** (4 false accepts, 1 false reject — errors split both directions), reported in
`results/summary.md`. The judge is blind to model identity by construction.

---

## 7. Analysis plan

`analyze.py` reads `scored_<model>.csv` and writes all tables to `results/` plus
`results/summary.md`. Uncertainty is target-resampled bootstrap (B = 2000; targets are the unit of
sampling); ICR uses a permutation baseline (1000 shuffles). Regressions are logistic GLMs with
standard errors clustered by target — **not** crossed random effects; refitting finals with `glmer`
(R) is deferred to camera-ready. Holm correction is applied within each hypothesis across
model × mode rows.

| Metric | Definition | Output |
| --- | --- | --- |
| Acc@N | Fraction correct at size N (item, country, joint), bootstrap CIs over targets | `acc_by_N.csv` |
| Initial Drop | Acc@1 − Acc@2 | `scaling_summary.csv` |
| Scaling slope | Joint-accuracy change per added image, N = 2–5 | `scaling_summary.csv` |
| Position Gap | Best − worst slot accuracy at N = 5, plus full profile and quadratic slot test | `h2_position.csv`, `position_profile.csv` |
| Context attraction (H3a) | Share of wrong choices naming a shown distractor, vs (N−1)/4 | `h3a_context_attraction.csv` |
| Near–far effect (H3b) | Joint accuracy and mis-binding, near minus far; logistic regression with `mean_sim` control | `near_vs_far.csv`, `h3_near_regression.csv` |
| Illusory Conjunction Rate | Wrong joint answers mixing two shown images, vs slot-conditioned independence baseline (permutation *p*) | `illusory_conjunctions.csv` |
| Popularity tier | Accuracy and prior-drift share by tier | `by_popularity_tier.csv` |
| Paraphrase check | P2/P3 vs P1 on the same targets | `paraphrase_check.csv` |
| Leak check | Text-only accuracy | `text_only_leak.csv` |
| Describe-then-answer (E6) | P4 − P1 rescue with CI | `describe_then_answer.csv` |
| Attention capture | Among mis-bindings, share where the named wrong image has the highest answer-token attention mass (vs chance 1/N) | `attention_capture.csv` |

**CLIP control** (`clip_sim.py`, already computed → `clip_sim.csv`, 491 target × list rows):
`openai/clip-vit-base-patch32` cosine similarity between each target and its distractors, averaged
per (target, list); enters the H3b regression so "near hurts" cannot be confounded by look-alikes.

**Attention probe** (`run_models.py --attn --exps E2`): the model is switched to eager attention
(SDPA/flash return no weights) and, for every generated answer token, the attention mass landing on
each shown image's token span is recorded (averaged over layers, heads, and answer tokens;
`attn_<model>.jsonl`). Test: among mis-binding errors, does the *named* wrong image carry the
highest mass? This is correlational; the causal version (activation patching) is deferred.

**Locus-discrimination table** (`results/locus_discrimination.md`): six observed signatures mapped
onto three candidate binding loci. Predictions are **pre-registered** (fixed in
`analyze.py:LOCUS_ROWS`, mirrored in the README); only the Observed column is data:

| Signature | Encoder-bound | Latent scene state | Text-chain readout |
| --- | --- | --- | --- |
| Step at N=1→2 (H1) | flat | sharp step | gradual decline |
| Middle slots worse at N=5 (H2) | no slot effect | U-shape | possible — weak test |
| Near > far, CLIP controlled (H3b) | no | yes | yes — weak test |
| Illusory conjunctions > baseline | no | yes | no (whole answer copied) |
| Describe-then-answer rescues (E6) | no | partial | yes |
| Attention peaks on named wrong image | no (stays on target) | yes | no commitment |

**Figures** (`make_figures.py` → `results/figures/`, PDF + PNG): fig1 Acc@N with CIs per model,
both modes; fig2 error-type shares vs N for all 6 models; fig3 slot profile at N = 5; fig4 E6
describe-then-answer rescue by N; fig5 near vs far mis-binding at N = 5. Two standing cautions
before reading any result: check `paraphrase_check.csv` and `text_only_leak.csv` first.

---

## 8. Results (main study, completed 2026-10-09)

All numbers below come from `results/` (17 tables + `summary.md`), regenerated from the frozen
inputs; `verify_results.py` recomputes every headline figure independently from the raw logs and
passes with max |diff| 0.0000. Raw completeness: 6,096/6,096 calls and 20/20 repeats per model;
repeats are identical 20/20 for all 6 models.

### 8.1 Validity checks (read before any verdict)

- **Leak:** text-only MC accuracy ≤ 0.26 per field against a 5-option chance rate of 0.20
  (gemma4 0.00; granite4v ≤ 0.09); open ≤ 0.05. No leak channel; residual MC hits are option-level
  guessing plus famous-item priors.
- **Paraphrase:** P1/P2/P3 joint accuracy differences are within noise on every model × mode
  (`paraphrase_check.csv`).
- **Repeat determinism:** 20/20 identical replies for all 6 models.
- **Judge:** hand-check of a random 100-row subsample (seed 2026, main-study rows only,
  `data/judge_sample_100.jsonl`): **95/100 agreement** — 4 false accepts, 1 false reject, errors
  split both directions, so the judge does not systematically inflate accuracy. 281/19,364 verdicts
  (1.4%) failed to parse and were not applied; affected rows keep their heuristic labels
  (predominantly format errors), a conservative direction.

### 8.2 Headline: a clean split by size class

MC joint accuracy, N = 1 → N = 5 (`acc_by_N.csv`, `scaling_summary.csv`):

| Class | Model | N = 1 | N = 5 | Slope per image (N = 2–5) [95% CI] |
| --- | --- | --- | --- | --- |
| ≥ 7B | gemma4 | 0.856 | 0.801 | −0.006 [−0.020, 0.007] |
| ≥ 7B | qwen3vl | 0.831 | 0.807 | −0.001 [−0.015, 0.013] |
| ≥ 7B | qwen25vl | 0.694 | 0.753 | +0.004 [−0.006, 0.013] |
| ≤ 4B | gemma3_4b | 0.562 | 0.434 | −0.042 [−0.062, −0.024] |
| ≤ 4B | granite4v | 0.544 | 0.320 | −0.058 [−0.078, −0.040] |
| ≤ 4B | lfm25v | 0.556 | 0.315 | −0.057 [−0.078, −0.038] |

The three ≥ 7B models are robust to haystack growth; the three ≤ 4B models lose 0.13–0.22
absolute, with slopes significantly below zero. The benchmark stratifies binding capability by
size class — the phenomenon is real but not universal.

**Open mode is at floor and carries no binding signal.** Open joint accuracy at N = 1 is already
0.044–0.300 (item naming 0.087–0.369) and flat in N: models can *recognize* culturally specific
items (MC) far better than they can *name* them — the pilot's documented N = 1 gate failure is
confirmed at scale. Per the pre-registration, open is the headline mode for H1/H2/H3b; since open
accuracy shows no N-decline to decompose, **all binding conclusions below rest on the MC mode**,
with open rows reported for completeness.

### 8.3 Hypothesis verdicts (MC; Holm-corrected within hypothesis)

| ID | Verdict | Detail |
| --- | --- | --- |
| **H1** | **Rejected everywhere; no step in any cell** | "Gradual" in the three ≤ 4B models (+ lfm25v open); "inconclusive or no decline" in the ≥ 7B rows. Binding erodes gradually per added image, not at the appearance of a second image |
| **H2** | **Supported in exactly the three degrading models (MC)** | U-shaped slot profile at N = 5: gemma3_4b gap 0.175 (Holm p < 0.001), granite4v 0.131 (p = 0.008), lfm25v 0.150 (p < 0.001); middle slots worst. Flat profiles in the robust models and everywhere in open mode |
| **H3a** | **Supported only for lfm25v** | lfm25v's wrong choices pick shown images above the (N−1)/4 chance rate at N = 2, 3, 4 (excess 0.18/0.14/0.09, CIs above 0). All other models sit at chance: their wrong MC choices are ~uniform over the four wrong options |
| **H3b** | **Supported for gemma3_4b and lfm25v (MC)** | CLIP-controlled near coefficient: OR 0.63 (Holm p = 0.015) and 0.60 (p = 0.010); granite4v raw p = 0.007 misses after Holm (p = 0.070). The *direction* (near hurts) is unanimous across all six models in MC at both N = 2 and N = 5; only the two weakest models survive correction |

**Reading of H3a (important caveat).** At N = 5 all five options name shown images, so the
near-100% mis-binding share there (§8.4) is partly mechanical. H3a was pre-registered exactly to
test attraction *beyond* option availability: it shows that for 5 of 6 models the rising
mis-binding share tracks the rising number of shown options, i.e. option-level guessing — only
lfm25v is genuinely attracted to shown distractors above chance.

### 8.4 What the errors are

- **Error anatomy vs N** (`error_shares_by_N.csv`, fig2): the mis-binding share of wrong MC fields
  climbs with N in every model (e.g. granite4v 0.28 → 0.56 → 0.72 at N = 2, 3, 4; lfm25v 0.43 →
  0.64 → 0.84) and reaches 0.92–1.00 at N = 5 (granite4v 0.918 with the remainder being its known
  MC format errors; all others ≥ 0.998). Prior drift dominates only where unshown options exist
  (N = 2) and in open mode throughout.
- **Illusory conjunctions** (`illusory_conjunctions.csv`): ICR is **at or below** the
  slot-conditioned independence baseline in all 48 model × mode × N cells (N = 5 MC: observed
  0.50–0.96 vs baseline 0.82–0.98). Per the pre-registered interpretation: wrong joint answers are
  **whole answers lifted from one wrong shown image** (object-level selection failure), not
  item-from-one-image + country-from-another recombination. The single Holm-significant excess
  (qwen25vl open N = 3, 0.9% vs 0.2%) is ~1 trial and descriptively negligible.
- **Popularity tiers** (`by_popularity_tier.csv`): in MC, the robust models' prior-drift share is
  ~2× higher on low-tier than high-tier items (qwen3vl 0.284 vs 0.129; gemma4 0.250 vs 0.113;
  qwen25vl 0.241 vs 0.175) — models lean on parametric priors for rarer items, as pre-registered.

### 8.5 Probes: describe-then-answer (E6) and attention capture

- **E6 never rescues** (`describe_then_answer.csv`, fig4). All-N rescue (P4 − P1) is significantly
  *negative* for gemma3_4b (MC −0.118 [−0.186, −0.054]; open −0.100 [−0.164, −0.039]) and
  granite4v MC (−0.082 [−0.143, −0.025]); every other model × mode CI spans 0. Verbalizing the
  scene first does not recover the binding — for the weakest models it actively interferes.
- **Attention: no capture in 5 of 6 cells** (`attention_capture.csv`; probe run for qwen3vl,
  gemma4, granite4v). Among mis-binding fields at N = 5, the share where the *named* wrong image
  carries the highest answer-token attention mass is at chance (0.20): qwen3vl 0.172 (MC) / 0.174
  (open), gemma4 0.225 / 0.214, granite4v MC 0.182. Only granite4v open exceeds chance (0.333
  [0.253, 0.663]) — on 39 mis-binding fields, the weakest evidentiary cell.

### 8.6 Locus reading (`results/locus_discrimination.md`)

No single pre-registered locus matches all six signatures. The observed combination — gradual
decline (against a fixed-capacity step), U-shape and near > far in the degrading models
(scene-state-like), ICR **below** baseline (against attribute detachment in a crowded state;
consistent with whole-answer copying), no E6 rescue (against text-chain readout), and no
attentional capture (against selection being visible in answer-token attention) — is most
compatible with **object-level selection failing early**, during cross-image integration, before
any verbalizable scene state exists: the model commits to a wrong image wholesale, and neither
text mediation nor answer-time attention reflects the mis-selection. The causal version of this
test (activation patching) is deferred (§12).

### 8.7 Figures

`results/figures/`: fig1 accuracy vs N (both modes, all 6 models, CI bands); fig2 error anatomy
vs N (2×3 grid, all models); fig3 slot profile at N = 5; fig4 E6 rescue by N; fig5 near vs far
mis-binding at N = 5. All regenerated by `make_figures.py` from `results/*.csv`.

---

## 9. Pilot study and gate (completed 2026-10-08)

The pilot ran 7 models × 192 calls on the final data build (890 images) against the pre-registered
gates of `plan.md` §5 (full record: `data/pilot_gate_report.md`):

| Gate | Threshold | Outcome | Verdict |
| --- | --- | --- | --- |
| Format | ≤ 5% of replies | 5 models 0/192; granite4v 8.3%, smolvlm2 15.1% (MC only) | remedied — see prompt history |
| N = 1 | ≥ 1 model ≥ 5/6 joint, open | best 4/6 open; MC 5–6/6 for 5 models | **documented fail, proceed** |
| Numbering | hand-read 20 replies | 14/20 named target, **0/20 other-shown**, 6/20 wrong-name attempts at the right image | **pass** |
| Text-only leak | MC ≤ 3/6; open ≤ 1/6 | MC all ≤ 3/6; gemma3_4b open country 2/6 (stereotype priors) | **pass with note** |
| Repeat | ≥ 18/20 identical | 20/20 for all 7 | **pass** |

**N = 1 documented fail — rationale.** Failures concentrate on 4 of 6 pilot targets and are item
properties, not pipeline faults: Sheng (37 sitelinks) was misidentified by all 7 models
(shakuhachi/dizi/erhu/…); Chartres was named a *different* cathedral by all 7; one Baklava→Greece
attribution reflects a genuinely contested origin; "flute" for Bansuri resolves via alias. The same
models reach 5–6/6 joint on the same targets in MC, the label audit is 30/30, and repeats are
20/20. Redesigning the sanity instrument after seeing pilot data would compromise the
pre-registration, so the decision (user-approved) was to **document and proceed**; the open-mode
naming gap (naming culturally specific items is harder than recognizing them) is itself reported as
a pilot observation. Pilot targets never enter the main pool, so no main claim depends on their
easiness.

**Prompt version history (declared per plan §6).** v1 = plan-literal text. v2 added an explicit
"option letters, not the full names" clause plus a two-line worked example: granite4v 16→2 MC format
errors but smolvlm2 parroted the example verbatim (69/96, with silent C/E bias). v3 kept the clause,
dropped the example: smolvlm2 collapsed to bare letters (91/96). No variant fixed both small models,
so the main runs **reverted to v1 with zero deviation for all models**; granite4v's 16/96 on v1
stands as a documented per-model limitation. smolvlm2 was replaced by lfm25v (which then passed the
full pilot: 0/192 format errors, 20/20 repeats; its open-mode profile is abstain-heavy — 41 abstains
vs 26 prior drifts at N ≥ 2 — while its MC errors already show the binding phenomenon: 16
mis-bindings and 12 illusory conjunctions among 18 wrong joint answers).

---

## 10. Pre-registration and reproducibility

- **Freeze of record (no-git variant):** `prereg_snapshot/` + `PREREG_MANIFEST.sha256` — a SHA-256
  manifest of all prompts, data, trials, scripts, judge state and pilot outputs, frozen
  **2026-10-08T15:27:50Z** (v2: regenerated after the gemma3_12b exclusion; supersedes the
  14:58:52Z v1 snapshot, itself superseding a v3-era snapshot — all changes strictly pre-results).
  For the record: automation created a local-only git commit (`a679183`, tag `prereg-v1`) earlier
  the same day capturing the superseded v2 prompt text; it was never pushed and is **not** the
  freeze of record.
- After the freeze, changing a prompt, alias list, seed or decision rule requires disclosure in the
  paper. The v1→v2→v3→v1 prompt history and both roster changes are disclosed here and in
  `data/pilot_gate_report.md`.
- **Seeds:** 2026 for pool/target/distractor construction; `2026-opt-<target>-<list>` for option
  shuffles; per-trial deterministic seeds in simulation; bootstrap/permutation RNGs pinned in
  `analyze.py`.
- **Self-test:** `python3 ch.py selftest` verifies the full trial algebra (nestedness, 5
  countries/items per set, fixed options per (target, list), near/far region constraints, slot
  balance, pilot/main disjointness), the scorer against four synthetic answerer strategies
  (oracle / first-image / mixed-attribute / absent-image), and the P4 prompt shape. It must end with
  `SELFTEST OK` before any build is trusted.
- **Host note:** on this machine the system cuDNN in `LD_LIBRARY_PATH` shadows the wheel-bundled one
  and crashes torch; every GPU command is prefixed with `env -u LD_LIBRARY_PATH`.

---

## 11. Final status (2026-10-09)

| Component | State |
| --- | --- |
| Dataset build (890 images, 166 targets) | done, audited (30/30 labels, 17 visual rejections applied) |
| Image preparation (`images/prepared/`, 768 px, EXIF-stripped) | done |
| Trials (`trials_pilot.jsonl` 96; `trials_main.jsonl` 3,048) | done, self-test green |
| Pilot (7 models, gate report, judge labels) | done |
| Pre-registration freeze | done (v2 manifest) |
| CLIP similarities (`clip_sim.csv`, 491 rows) | done |
| Main runs | **done** — 6 models × 6,096 calls = 36,576 (+ 120 repeat calls), all complete, ~6 h wall |
| Attention pass (`attn_<model>.jsonl`, E2) | done for qwen3vl, gemma4, granite4v (1,600 rows each; gemma4 rerun after the sliding-window fix — see post-freeze changes log in `data/pilot_gate_report.md`) |
| Scoring / judge / analysis (`results/`) | done — 19,364 judge verdicts, 19,083 overrides applied; all 17 result tables + `summary.md` |
| Figures, locus table Observed column, summary | done — `results/figures/` (fig1–fig5), Observed column filled |
| Independent verification | done — `verify_results.py` recomputes every headline table from raw/scored/attn data; **all checks pass** (max |diff| 0.0000) |
| Judge hand-check | done — 95/100 agreement on `data/judge_sample_100.jsonl` (reported in `results/summary.md`) |

---

## 12. Limitations and deferred work

- **Open mode is at floor.** Open-ended naming of culturally specific items tops out at 0.30 joint
  even at N = 1 and shows no N-decline, so every binding conclusion in §8 rests on the
  multiple-choice mode; open mode functions as a knowledge/naming ceiling diagnostic only.
- **MC error anatomy is availability-driven for 5 of 6 models.** H3a shows wrong MC choices are
  ~uniform over wrong options except for lfm25v, so the rising mis-binding share with N (and the
  ~100% share at N = 5, where all options are shown) must be read against the shown-option rate,
  not as direct evidence of attraction to shown images.
- **Attention probe covers 3 of 6 models** (qwen3vl, gemma4, granite4v), and the single
  above-chance capture cell (granite4v open) rests on 39 mis-binding fields with a wide CI.
- **E6 descriptions were not semantically scored** — the probe measures the net effect of the
  describe-first instruction on the answer, not the quality of the intermediate descriptions.
- **No causal intervention (deferred).** Attention capture is correlational. The decisive LVR
  experiment — activation patching between target and distractor image-token spans to flip the
  answer — is deferred to camera-ready or follow-up work.
- **Model coverage (deferred).** Results are per-model behavioral signatures; cross-architecture
  convergence with a third family beyond the current five is deferred (extend `MODELS`).
- **Mixed-effects models (deferred).** Logistic GLMs with target-clustered SEs approximate the
  crossed random effects; final models to be refit with `glmer` (R).
- **Pool size.** 890 images / 160 targets widens confidence intervals; bootstrap CIs are reported
  throughout. Attire (86 images) caps realistic target growth at ~60 per category.
- **Contamination.** Wikimedia Commons images have likely been seen in training. N = 1 accuracy is a
  knowledge baseline; all claims concern degradation at N ≥ 2, not memorization.
- **Culture proxy.** Country of origin is a coarse proxy for culture. Curated labels were gated
  against Wikidata (8 disagreements dropped) and hand-audited on a 30-target sample (30/30).
- **No generic-object control (H4 of the original design).** Whether the effect is culture-specific
  is untested; needs a second, clearly licensed image source and is future work.

## 13. Ethics and release

- Country of origin is used as an experimental proxy, not a ranking of cultures; prior-drift rates
  by tier are model behavior, not cultural claims.
- Images are not redistributed; `data/pool.csv` (license, author, source URL per image), the trial
  files, scripts and seeds are the data release. 118 of 890 pool images are public domain or CC0;
  the rest carry their Commons licenses.
- The LLM judge's family overlap with 2 of 6 evaluated models is disclosed; hand-checked agreement
  on a random 100-row subsample of main-study judged rows is 95/100 (`results/summary.md`).
- The pre-registration tag/manifest is cited in the paper; the submission checklist (page limit,
  anonymity, dataset documentation, regenerated figures, prompt-history disclosure) is in
  `plan.md` §13.

---

## Appendix A — File map

| Path | Role |
| --- | --- |
| `plan.md` | Primary pre-registration (design, hypotheses, decision rules) |
| `ch.py` | Data build, image prep, trial construction, prompts, scorer, simulator, self-test |
| `run_models.py` | Model runner (greedy, resumable) + attention-capture probe |
| `judge.py` | LLM judge (gemma-4-31B-it, blind, greedy, logged) |
| `clip_sim.py` | CLIP target–distractor similarity (H3b control) |
| `analyze.py` | All metrics, tests, tables, locus-discrimination report |
| `verify_results.py` | Independent audit: recomputes every headline table from raw/scored/attn data without importing `analyze.py` |
| `make_figures.py` | Paper figures → `results/figures/` (PDF + PNG) |
| `audit_sample.py` | Label-audit sampler |
| `data/` | `pool.csv` (attribution), `targets.csv`, `lists.json`, `audit.txt`, `disputed_labels.txt`, `label_audit.txt`, `pilot_gate_report.md`, `judge_log.jsonl`, `judge_sample_100.jsonl` |
| `dataset/` | Raw Kaggle download (`metadata.csv`, `images/`) |
| `images/prepared/` | 890 neutral-ID, 768 px, EXIF-stripped JPEGs |
| `trials_pilot.jsonl`, `trials_main.jsonl` | Frozen trial definitions (96 / 3,048) |
| `raw_<model>.jsonl` | Raw model outputs (one line per trial × mode) |
| `attn_<model>.jsonl` | Attention-probe rows (E2, per-trial attention mass per shown image) |
| `scored_<model>.csv`, `review_<model>.csv`, `judge_overrides.csv` | Scoring layer |
| `pilot_out/` | Frozen pilot artifacts for all piloted models (incl. dropped ones, and the v3-text re-pilot under `v3/`) |
| `prereg_snapshot/`, `PREREG_MANIFEST.sha256` | Freeze of record |
| `contact_sheets/` | Visual-audit contact sheets |
| `results/` | Analysis outputs: 17 tables + `summary.md` + `locus_discrimination.md`; `figures/` (fig1–fig5, PDF + PNG) |

## Appendix B — Pipeline commands

```bash
python3 ch.py selftest --meta dataset/metadata.csv    # must end with SELFTEST OK
python3 ch.py build-data --meta dataset/metadata.csv  # -> data/ (pool, targets, lists, audit)
python3 ch.py prep --src dataset                      # -> images/prepared/
python3 ch.py trials pilot                            # 96 trials
python3 ch.py trials main                             # 3,048 trials

env -u LD_LIBRARY_PATH python3 run_models.py qwen3vl --trials trials_pilot.jsonl [--repeat]
python3 ch.py score qwen3vl --trials trials_pilot.jsonl
env -u LD_LIBRARY_PATH python3 judge.py               # judge fills judge_overrides.csv; rescore after
# pilot gate (plan.md §5) -> freeze (prereg_snapshot/ + PREREG_MANIFEST.sha256) ->
env -u LD_LIBRARY_PATH python3 run_models.py qwen3vl --trials trials_main.jsonl [--repeat]
env -u LD_LIBRARY_PATH python3 run_models.py qwen3vl --trials trials_main.jsonl --exps E2 --attn
env -u LD_LIBRARY_PATH python3 clip_sim.py
python3 ch.py score qwen3vl --trials trials_main.jsonl
python3 analyze.py qwen3vl qwen25vl gemma4 gemma3_4b granite4v lfm25v   # -> results/
python3 make_figures.py                                               # -> results/figures/
python3 verify_results.py                                             # independent audit; exits 1 on any mismatch
```

*Document generated from the repository state of 2026-10-09: `README.md`, `plan.md`,
`data/pilot_gate_report.md`, `data/audit.txt`, `results/summary.md`, and the pipeline sources.
Where numbers differ between `plan.md` (written against the 907-image build) and the final build,
the final frozen numbers (890 images, 152 near-feasible targets, 3,048 trials) are used.*
