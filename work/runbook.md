# RunPod runbook: one GPU session, ≤ 12 GPU-hours, ~$2–5

You do not need to know the project. Follow the steps in order and copy-paste the commands.
`<IP>`/`<PORT>` come from the RunPod "Connect" dialog.

## 0. What you need before starting (the human must provide these)

| Item | Where to get it | Used for |
|---|---|---|
| RunPod account with ≥ $10 credit | runpod.io → Billing | the pod |
| SSH key pair on your laptop (`~/.ssh/id_ed25519.pub`) | `ssh-keygen -t ed25519` if missing | log in and copy files |
| **HF_TOKEN** (read token) from an account that has **accepted the google/gemma-3-4b-it license** | huggingface.co → Settings → Access Tokens; accept the license at huggingface.co/google/gemma-3-4b-it | model downloads (gemma-3 is gated; the others are public) |
| **Kaggle API credentials** (`KAGGLE_USERNAME` + `KAGGLE_KEY`) | kaggle.com → Settings → API → "Create New Token" (downloads kaggle.json; its two fields are these values) | dataset download. There is **no** `~/.kaggle/kaggle.json` on the project Mac. |

Never paste tokens into chat, git, or this file.

## 1. Pack the repo (on the Mac)

```bash
cd /Users/shinjini/Desktop/wacv
COPYFILE_DISABLE=1 tar czf wacv-lvr.tgz --exclude='wacv-lvr/.venv' --exclude='wacv-lvr/contact_sheets.zip' --exclude='wacv-lvr/images' --exclude='wacv-lvr/*smoke*' wacv-lvr
ls -lh wacv-lvr.tgz        # ~30-40 MB
```
(Alternative: commit and push the changes, then `git clone` on the pod. The repo `github.com/sprnjt/wacv-lvr`
may be private, which needs a GitHub token on the pod, so the tarball is simpler.)

## 2. Create the pod

1. RunPod → Settings → **SSH Public Keys**: paste the contents of `~/.ssh/id_ed25519.pub`. Save.
2. Pods → **Deploy**. GPU: **A40 (48 GB)**, Community Cloud, ~$0.35–0.44/h. If no A40 is free, use an
   **RTX A6000 48 GB** (~$0.49/h). Use a 24 GB card (4090/3090/A5000) only as a last resort; see §7.
3. Template: an official **RunPod PyTorch 2.x** template (Python ≥ 3.10, torch ≥ 2.6, CUDA 12.x, e.g.
   "RunPod Pytorch 2.8.0"). `setup.sh` checks the versions and installs torch cu128 if they're too old.
4. **Container disk 40 GB, Volume disk 100 GB** (mounted at `/workspace`). Models ≈ 41 GB in total
   (LFM2.5-VL-3B ≈ 6, Qwen3-VL-8B ≈ 17.5, gemma-3-4b ≈ 8.6, granite-vision-4.1-4b ≈ 8–9), plus dataset
   zip + extract, plus outputs.
5. Make sure TCP port 22 is exposed (default in these templates). Deploy. Billing starts now.

## 3. Upload and log in

Pod → **Connect** → "SSH over exposed TCP" shows `ssh root@<IP> -p <PORT>`. Use that one (the
`ssh.runpod.io` proxy can't scp).
```bash
scp -P <PORT> -i ~/.ssh/id_ed25519 /Users/shinjini/Desktop/wacv/wacv-lvr.tgz root@<IP>:/workspace/
ssh root@<IP> -p <PORT> -i ~/.ssh/id_ed25519
```
On the pod:
```bash
cd /workspace && tar xzf wacv-lvr.tgz && cd wacv-lvr
tmux new -s gpu || tmux attach -t gpu          # survives SSH drops (Ctrl-b d to detach)
cat > /root/secrets.env <<'EOF'
export HF_TOKEN=hf_xxx
export KAGGLE_USERNAME=xxx
export KAGGLE_KEY=xxx
export HF_HOME=/workspace/hf
EOF
chmod 600 /root/secrets.env && source /root/secrets.env
```
(`/root` is wiped when the pod stops. After a restart, recreate the file and `source` it again.)

## 4. Setup (~15–25 min; must end with `SETUP OK`)

```bash
mkdir -p logs && bash runpod/setup.sh 2>&1 | tee logs/setup.log
```
It installs transformers 5.20.1 (and torch only if needed). It picks the `env -u LD_LIBRARY_PATH`
prefix (the README host note) if CUDA works without it. It unzips `data.zip`/`results.zip` and downloads
the Kaggle dataset. It **checks that the dataset rebuilds the frozen `data/pool.csv` byte-for-byte**,
then runs `ch.py selftest` and prepares the 890 images. It checks gemma access and downloads LFM.
If it stops with FATAL, read the line above it. A `WARNING: no access to google/gemma-3-4b-it` only
affects job 4a, so you can still continue.

## 5. Run everything

```bash
nohup bash runpod/run_all.sh > logs/run_all.log 2>&1 &
```
Order (each job starts with a 2–3-item smoke test plus an automatic check; a failure skips to the
next job instead of stopping):

| # | Job | Size | Est. GPU time (planning) | Pessimistic |
|---|---|---|---|---|
| 0 | setup (§4) | pip, Kaggle, prep, LFM download | 0.4 h | 0.6 h |
| 1 | causal probe **lfm25v** | 128 cases × (35 fwd + gate) | 0.35 h | 0.8 h |
| 2 | **lfm25v** repro (20 calls) + E8 (800) + E7 (280, describe-first) | 1,100 calls | 0.4 h | 0.7 h |
| 3 | causal probe **qwen3vl** | 74 cases | 0.45 h | 0.9 h |
| — | **cut line A: the core is done (≈1.6 h elapsed)** | | | |
| 4 | **gemma3_4b** then **granite4v**: repro + E8 + E7 | 2 × 1,100 calls | 1.0 h | 1.8 h |
| — | **cut line B** | | | |
| 5 | **lfm25v** attention pass (E2, eager) | 1,600 calls | 0.55 h | 1.2 h |
| post | CPU summaries (`results/causal_*`, `results/interventions_*`, `results/attention_capture_lfm25v.csv`, `verify_results.py`) | | 0.05 h | 0.1 h |
| | **Total** | | **≈ 3.2 h ≈ $1.3–1.5** | **≈ 6.1 h ≈ $2.5–3** |

How the estimates were made: the main study ran 36,576 calls in ~6 h wall, i.e. 0.59 s/call on the
original (unrecorded) host. Splitting that by model size gives ≈0.09 s per B-params per call: lfm25v
≈0.26 s, gemma3_4b ≈0.38, granite4v ≈0.35, qwen3vl ≈0.77. I applied a ×2 slowdown for an A40. E7
describe-first calls count as ≈5 short calls. A causal case is one teacher-forced forward per
condition: 3 donor captures + 2 baselines + 5 layers × 3 conditions × 2 answers = 35 forwards (~0.25 s
for LFM, ~0.5 s for qwen3vl on A40), plus one short gate generation. Attention calls count as ×2
(eager + stored attentions). The first real progress lines in the logs calibrate this. Hard cap:
`MAX_HOURS=11` (no new step starts after 11 h), and each step has a 4 h `timeout`. 12 h on an A40 is at
most ≈ $5.3, well inside $10–15.

To stop at a cut line, either start with it (`STOP_AFTER=3 nohup bash runpod/run_all.sh > logs/run_all.log 2>&1 &`)
or, while it's running, `pkill -f run_all.sh; pkill -f run_models.py; pkill -f causal_probe.py` and
restart with `STOP_AFTER=…`. Everything resumes.

## 6. Monitor

```bash
tail -f logs/run_all.log                     # one line per step: start / done (min) / FAIL / skip
tail -f logs/causal_lfm25v.log               # "k/128 cases (12.3s)" per case
tail -f logs/E8_lfm25v.log                   # "100/800 trials" every 100
grep REPRO logs/repro_*.log                  # want 20/20 (≥18). Lower = environment drift: report it next to E7/E8 deltas
watch -n 30 nvidia-smi                       # GPU busy, memory headroom
wc -l causal_*.jsonl raw_*_interventions.jsonl attn_lfm25v.jsonl 2>/dev/null
cat logs/prefetch.log                        # later models downloading in the background
```
Expected final sizes: `raw_<m>_interventions.jsonl` 1,080 rows each. `attn_lfm25v.jsonl` 1,600.
`causal_lfm25v.jsonl` = 15 × (cases passing the gate) + 1 × (cases failing; `wrong_donor_gate_fail`),
over 128 cases. `causal_qwen3vl.jsonl` the same over 74 cases. `logs/summary_causal_*.log` gives the
gate-fail count. Usable n = cases − gate fails (want ≥ 40 for qwen3vl).

If a step FAILs: `less logs/<step>.log`, fix the cause, then run the same `nohup bash runpod/run_all.sh …`
again (finished steps are skipped, partial outputs resume). A failed smoke check automatically re-smokes
on the next start. To force a step again: `rm runpod/state/<step>.done`.

## 7. If only a 24 GB GPU is available

Jobs 1–4 fit (qwen3vl bf16 ≈ 17.5 GB + activations). Job 5 (LFM attention) stores L×L attention for 8
layers and can OOM (~23 GB) on trials where all five images are near-square. Run with `STOP_AFTER=4`
and skip job 5. It is the lowest priority.

## 8. Download the results

On the pod:
```bash
bash runpod/collect.sh       # -> /workspace/gpu_outputs_<date>.tar.gz (new outputs + logs only; no frozen files)
```
On the Mac:
```bash
scp -P <PORT> -i ~/.ssh/id_ed25519 'root@<IP>:/workspace/gpu_outputs_*.tar.gz' /Users/shinjini/Desktop/wacv/
cd /Users/shinjini/Desktop/wacv/wacv-lvr && tar xzf ../gpu_outputs_*.tar.gz     # adds causal_*, raw_*_interventions, attn_lfm25v, results/causal_*, results/interventions_*
```
Check that the archive opens and has the files listed by `collect.sh` **before** you stop the pod.

## 9. Stop billing (do not skip)

RunPod → Pods → your pod → **Stop**. GPU billing ends; the volume keeps a small storage charge.
Once the download is checked → **Terminate**, which ends all charges and deletes the volume.
(`AUTO_STOP=1` makes `run_all.sh` collect and stop the pod by itself. Downloading afterwards then needs
a restart, so only use it if nobody will be watching at the end.)

## 10. After the session (CPU, on the Mac)

- Record the post-freeze changes (see `work/gpu_audit.md` → Edits) and the REPRO counts in the post-freeze
  log of `data/pilot_gate_report.md`.
- Causal: read `results/causal_lfm25v.csv` and `results/causal_qwen3vl.csv` (rescue = target_clean_to_target
  Δ > 0; induction = wrong_clean_to_target Δ < 0; self_to_target_control ≈ 0). LFM layer 0 rescue is ≈0
  by construction (conv layers). Report it as exploratory, post-freeze.
- E7/E8: `results/interventions_<m>.csv` (paired Δ vs E6/E2, plus `format_errors`).

## 11. E9 collage + optional HF upload (added 2026-10-17)

Before the first `collage <model>` on the pod, the repo needs `make_collage.py`, `analyze_collage.py`,
`trials_collage.jsonl`, and the updated `ch.py`, `run_models.py` and `runpod/collect.sh`. Then:
```bash
python3 make_collage.py images          # needs the 890 images/prepared/*.jpg; -> 960 PNGs in images/collage/
sha1sum trials_collage.jsonl            # 9f2292e7b211de0c4b63661196ac18f046608b45 (or: python3 make_collage.py trials)
JOBS="collage:lfm25v collage:gemma3_4b collage:granite4v collage:qwen3vl" nohup bash runpod/run_all.sh > logs/e9_lane.log 2>&1 &
```
(`run_all.sh` writes `work/smoke/e9_trials.jsonl` at start, so start a fresh invocation after the files are in
place.) Outputs: `raw_<m>_collage.jsonl` (640 rows each), `results/collage_<m>.csv`, `logs/E9_<m>.log`.
Est. ≈ 0.75 h (pessimistic 1.5 h) for all 4 models. For C4, read country accuracy and ICR, not item/joint.

**HF upload (optional):** `collect.sh` also uploads the outputs to a private HF dataset if both are set:
```bash
export HF_REPO=HiLEx-DLA/culturehaystack-outputs     # HF_TOKEN must have WRITE access to this org/repo
bash runpod/collect.sh
```
It uploads only `raw_*_interventions/collage.jsonl`, `causal_*.jsonl`, `attn_lfm25v.jsonl`, `results/*.csv`
and `logs/`, into `gpu_outputs_<stamp>/`. It never uploads images/, dataset/ or credentials, and it skips the
upload if any of those files contains the token. A failure or a 30 min timeout prints a WARNING and never
fails the collection. The read-only token from §0 is not enough for the upload.
