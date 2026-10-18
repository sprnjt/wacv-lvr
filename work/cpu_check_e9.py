#!/usr/bin/env python3
"""CPU-only check of E9 (collage), same approach as work/cpu_check.py: real LFM2.5-VL / Qwen3-VL processors and
chat templates, TINY random-weight models built from the real configs, fake images.

  /Users/shinjini/Desktop/wacv/.venv-cpu-check/bin/python work/cpu_check_e9.py

Everything is written to a fresh temp dir OUTSIDE the repo (cwd; data/ symlinked); the repo is only read.
Ends with "E9 CPU CHECK OK" only if the repo's git status and the frozen files are unchanged.
"""
import hashlib, os, shutil, subprocess, sys, tempfile
from pathlib import Path
sys.dont_write_bytecode = True                  # no __pycache__ in the repo (subprocesses inherit the env var)
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

REPO = Path(__file__).resolve().parent.parent
TMP = Path(tempfile.mkdtemp(prefix="e9_cpu_check_")).resolve()
assert REPO not in TMP.parents and TMP != REPO
PY = sys.executable
git = lambda: subprocess.run(["git", "status", "--porcelain", "--untracked-files=all"], cwd=REPO,
                             capture_output=True, text=True, check=True).stdout
FROZEN = sorted([*REPO.glob("raw_*.jsonl"), *REPO.glob("scored_*.csv"), *REPO.glob("attn_*.jsonl"),
                 REPO / "trials_main.jsonl", REPO / "trials_pilot.jsonl", *(REPO / "results").rglob("*")])
sha = lambda ps: {str(p): hashlib.sha1(p.read_bytes()).hexdigest() for p in ps if p.is_file()}
git0, frozen0, had_images = git(), sha(FROZEN), (REPO / "images").exists()
os.chdir(TMP)
(TMP / "data").symlink_to(REPO / "data")
sys.path.insert(0, str(REPO))
import torch
from PIL import Image
import transformers
from transformers import AutoConfig, AutoProcessor
import ch, run_models, make_collage, analyze_collage

# 1. trials: deterministic and identical to the repo's file -----------------------------------------------
make_collage.trials()
assert Path("trials_collage.jsonl").read_bytes() == (REPO / "trials_collage.jsonl").read_bytes(), "trials differ from repo"
T = ch.read_jsonl("trials_collage.jsonl")
pool = ch.load_pool()
print("trials OK:", len(T), "(identical to repo trials_collage.jsonl)")

# 2. images from FAKE prepared images (temp dir only) -----------------------------------------------------
Path("images/prepared").mkdir(parents=True)
sizes = [(768, 576), (576, 768), (768, 768), (768, 512), (640, 768)]
for n, i in enumerate(sorted({i for t in T for i in t["panels"]})):
    Image.effect_noise(sizes[n % 5], 40 + n % 50).convert("RGB").save(f"images/prepared/{i}.jpg", quality=90)
make_collage.images()
h1 = sha(Path("images/collage").glob("*.png"))
make_collage.images()
assert sha(Path("images/collage").glob("*.png")) == h1 and len(h1) == 960, "image build not deterministic"
for t in [x for x in T if x["condition"] == "solo"]:
    tg, s = t["target"], t["slot"] - 1
    grid, solo = (Image.open(f"images/collage/E9_{tg}_{k}.png") for k in ("grid", "solo"))
    assert grid.size == solo.size == (768, 768) and grid.mode == "RGB"
    for j, xy in enumerate(make_collage.XY):
        p = Image.open(f"images/collage/E9_{tg}_p{j + 1}.png")
        box = (*xy, xy[0] + 376, xy[1] + 376)
        assert p.size == (376, 376) and grid.crop(box).tobytes() == p.tobytes(), (tg, j)     # same pixels as C2
        assert solo.crop(box).tobytes() == (p.tobytes() if j == s else Image.new("RGB", (376, 376), "white").tobytes())
    assert grid.getpixel((383, 100)) == grid.getpixel((100, 383)) == make_collage.GREY and grid.getpixel((1, 1)) == (255, 255, 255)
print("images OK: 960 PNGs, deterministic; C2 panels == collage cells pixel-for-pixel; solo blanks white")

# 3. prompts and image paths for every trial --------------------------------------------------------------
for t in T:
    parts = ch.make_parts(t, "mc")
    txt = "".join(v for k, v in parts if k == "text")
    assert [v for k, v in parts if k == "image"] == t["images"] and len(t["images"]) == t["N"]
    assert txt.count(t["ref"]) == (3 if t["condition"] == "separate" else 2) and txt.endswith("Item: <letter>\nCountry: <letter>")
    q1 = txt.split("Question 1.")[1].split("\n")[0]
    assert (ch.CATS[t["category"]][1] in q1) == (t["condition"] == "collage_kind"), q1
    msgs = run_models.build_messages(t, "mc")
    assert sum(c["type"] == "image" for c in msgs[0]["content"]) == t["N"]
print("prompts OK: ref phrase, MC format, kind only in C4, every image file resolves")

# 4. scoring with ch.py on constructed answers --------------------------------------------------------------
L = lambda opts, x: ch.LETTERS[opts.index(x)]
ans = lambda t, i, c: f"Item: {L(t['item_options'], pool[i]['item'])}\nCountry: {L(t['country_options'], pool[c]['country'])}"
for t in T:
    tg, other = t["target"], next(p for p in t["panels"] if p != t["target"])
    assert ch.score_trial(t, "mc", ans(t, tg, tg), pool)["joint_ok"] == 1
    li = L(t["item_options"], pool[tg]["item"])
    assert ch.score_trial(t, "mc", f"Item: {li}\nCountry: {li}", pool)["country_ok"] == 0     # letter copy is wrong
    s = ch.score_trial(t, "mc", ans(t, other, other), pool)
    assert s["item_err"] == ("prior_drift" if t["condition"] == "solo" else "mis_binding")
    assert ch.score_trial(t, "mc", ans(t, t["decoy"], t["decoy"]), pool)["country_err"] == "prior_drift"
print("scoring OK: oracle all correct; copied item letter always wrong country; other panel / decoy classified")

# 5. tiny models: image spans (1 for C1/C3/C4, 4 for C2), generation through run_models.main, resume, --out guard
torch.manual_seed(0)


def tiny(name):      # as in work/cpu_check.py
    hf = run_models.MODELS[name]["hf_id"]
    cfg = AutoConfig.from_pretrained(hf)
    t, v = cfg.text_config, cfg.vision_config
    if name == "lfm25v":
        t.num_hidden_layers, t.layer_types = 4, ["conv", "full_attention", "conv", "full_attention"]
        t.hidden_size, t.intermediate_size, t.block_ff_dim = 64, 128, 128
        t.num_attention_heads, t.num_key_value_heads, t.conv_dim = 4, 2, 64
        v.hidden_size, v.intermediate_size, v.num_hidden_layers, v.num_attention_heads = 32, 64, 2, 2
        cfg.projector_hidden_size = 64
    else:
        t.num_hidden_layers, t.hidden_size, t.intermediate_size = 6, 64, 128
        t.num_attention_heads, t.num_key_value_heads, t.head_dim = 4, 2, 16
        t.rope_parameters = dict(t.rope_parameters or {}, mrope_section=[2, 3, 3])
        v.depth, v.hidden_size, v.intermediate_size, v.num_heads, v.out_hidden_size = 4, 32, 64, 2, 64
        v.deepstack_visual_indexes = [1, 2, 3]
    return getattr(transformers, run_models.MODELS[name]["arch"])(cfg).eval().float(), AutoProcessor.from_pretrained(hf)


ch.write_jsonl("e9_smoke.jsonl", T[:4])
for name in ("lfm25v", "qwen3vl"):
    model, proc = tiny(name)
    run_models.load = lambda _n, m=model, p=proc: (m, p)
    fam = run_models.MODELS[name]["family"]
    lens = {}
    for t in T[:40]:                                 # 10 targets x 4 conditions
        x = run_models.prep_inputs(model, proc, fam, t, "mc")
        sp = run_models.image_spans(x["input_ids"][0].tolist(), run_models.image_token_ids(proc))
        assert len(sp) == t["N"] == (4 if t["condition"] == "separate" else 1), (name, t["trial_id"], len(sp))
        lens.setdefault(t["condition"], set()).add(sum(e - s for s, e in sp))
    print(name, "spans OK; image tokens per condition:", {k: sorted(v) for k, v in lens.items()})
    out = f"raw_{name}_collage_smoke.jsonl"
    run_models.main(name, "e9_smoke.jsonl", modes=("mc",), out_path=out)
    rr = ch.read_jsonl(out)
    run_models.main(name, "e9_smoke.jsonl", modes=("mc",), out_path=out)            # resume: no-op
    assert len(rr) == 4 == len(ch.read_jsonl(out)) and all(r["mode"] == "mc" for r in rr)
    tr = {t["trial_id"]: t for t in T}
    for r in rr:
        ch.score_trial(tr[r["trial_id"]], "mc", r["raw"], pool)
    try:
        run_models.main(name, "trials_collage.jsonl", modes=("mc",))
        raise AssertionError("--out guard did not fire")
    except SystemExit as e:
        assert "--out is required" in str(e)
    print(name, "generate OK (4 conditions, resume no-op, --out guard); first reply:", repr(rr[0]["raw"][:60]))

# 6. analyze_collage on a constructed raw file with known answers, plus a partial file -----------------------
#    C1, C2 oracle (5 C2 format errors); C3 answers the top-left panel; C4 item right, country of another panel
rows = []
for n, t in enumerate(T):
    tg, c = t["target"], t["condition"]
    other = next(p for p in t["panels"] if p != tg)
    raw = {"solo": ans(t, tg, tg), "separate": ans(t, tg, tg) if n >= 20 else "I cannot tell.",
           "collage_pos": ans(t, t["panels"][0], t["panels"][0]), "collage_kind": ans(t, tg, other)}[c]
    rows.append(dict(trial_id=t["trial_id"], mode="mc", raw=raw))
ch.write_jsonl("raw_fake_collage.jsonl", rows)
res = analyze_collage.main("fake", "raw_fake_collage.jsonl", "trials_collage.jsonl")
get = lambda tab, cond, m: next(r for r in res if r["table"] == tab and r["condition"].startswith(cond) and r["metric"] == m)
assert get("acc", "solo", "joint")["value"] == 1 and get("acc", "separate", "joint")["value"] == 155 / 160
assert get("acc", "separate", "format_errors")["value"] == 5 and get("acc", "collage_pos", "joint")["value"] == 0.25
assert get("acc", "collage_kind", "country")["value"] == 0 and "kind" in get("acc", "collage_kind", "item")["note"]
assert get("delta", "C3-C2", "joint")["value"] == (40 - 155) / 160 and get("delta", "C3-C2", "joint")["hi"] < 0
assert get("retention", "collage_pos", "joint")["value"] == 0.25 and get("errsrc", "collage_pos", "item:other_panel")["value"] == 120
assert get("icr", "collage_kind", "icr")["value"] == 1 and get("icr", "collage_pos", "icr")["value"] == 0
assert Path("results/collage_fake.csv").exists()
ch.write_jsonl("raw_part_collage.jsonl", rows[:37])
analyze_collage.main("part", "raw_part_collage.jsonl", "trials_collage.jsonl")
r = subprocess.run([PY, str(REPO / "runpod/check.py"), "mc", str(TMP / "raw_part_collage.jsonl"), str(TMP / "trials_collage.jsonl")],
                   capture_output=True, text=True)
assert r.returncode == 0 and "OK mc" in r.stdout, r.stdout[-500:] + r.stderr[-500:]
print("analyze_collage OK (known values, partial file); check.py mc OK")

# 7. shell scripts: syntax, then a stubbed run_all.sh (E9 only) and collect.sh incl. a failing HF upload -----
for s in ("setup.sh", "run_all.sh", "collect.sh"):
    subprocess.run(["bash", "-n", str(REPO / "runpod" / s)], check=True)
W = TMP / "pod"
(W / "runpod/state").mkdir(parents=True)
for f in ("ch.py", "run_models.py", "make_collage.py", "analyze_collage.py", "trials_main.jsonl",
          "trials_interventions.jsonl", "trials_collage.jsonl"):
    shutil.copy(REPO / f, W / f)
for f in ("run_all.sh", "collect.sh", "check.py"):
    shutil.copy(REPO / "runpod" / f, W / "runpod" / f)
(W / "data").symlink_to(REPO / "data")
stub = W / "stub_gpu.sh"      # GPU="$RUN python3": oracle MC replies (garbage for the smoke if FAIL_SMOKE=m)
stub.write_text(f"""#!/usr/bin/env bash
shift; {PY} - "$@" <<'EOF'
import sys, json; sys.path.insert(0, ".")
import ch
a = sys.argv[1:]; m, trials, out = a[1], a[a.index("--trials") + 1], a[a.index("--out") + 1]
import os; bad = os.environ.get("FAIL_SMOKE") == m and "smoke" in trials
pool = ch.load_pool()
with open(out, "a") as f:
    for t in ch.read_jsonl(trials):
        f.write(json.dumps(dict(trial_id=t["trial_id"], mode="mc", raw="??" if bad else ch.fake_raw(t, "mc", pool, "oracle"))) + "\\n")
EOF
""")
stub.chmod(0o755)
(W / "runpod/state/env.sh").write_text(f'RUN="{stub}"\n')
pre = ["prefetch"] + [f"{s}_{m}" for m in ("lfm25v", "qwen3vl") for s in ("select", "smoke_causal", "check_causal", "causal", "summary_causal")]
pre += [f"{s}_{m}" for m in ("lfm25v", "gemma3_4b", "granite4v") for s in ("repro", "smoke_iv", "check_iv", "E8", "E7")]
pre += ["smoke_attn_lfm25v", "check_attn_lfm25v", "attn_lfm25v"]
for s in pre:
    (W / f"runpod/state/{s}.done").touch()
env = dict(os.environ, PATH=f"{Path(PY).parent}:{os.environ['PATH']}", HF_HOME=str(TMP / "hf"))
env.pop("HF_TOKEN", None)
run = lambda extra: subprocess.run(["bash", "runpod/run_all.sh"], cwd=W, env=dict(env, **extra), capture_output=True, text=True).stdout
o1 = run({"FAIL_SMOKE": "lfm25v"})
assert "FAIL  check_E9_lfm25v" in o1 and not (W / "raw_lfm25v_collage.jsonl").exists(), o1
assert all(len(ch.read_jsonl(W / f"raw_{m}_collage.jsonl")) == 640 for m in ("gemma3_4b", "granite4v", "qwen3vl")), o1
o2 = run({})
assert "done  E9_lfm25v" in o2 and "[skip] E9_qwen3vl" in o2 and len(ch.read_jsonl(W / "raw_lfm25v_collage.jsonl")) == 640, o2
assert all((W / f"results/collage_{m}.csv").exists() for m in ("lfm25v", "gemma3_4b", "granite4v", "qwen3vl"))
assert o1.index("smoke_E9_lfm25v") < o1.index("smoke_E9_gemma3_4b") < o1.index("smoke_E9_granite4v") < o1.index("smoke_E9_qwen3vl")
print("run_all.sh stub OK: failed smoke check -> re-smoke on rerun; 640 rows x 4 models; analysis written")
fake = "hf_" + "FAKEtoken" * 4
(W / "logs/leak.log").write_text("nothing here\n")
cenv = dict(env, OUT_DIR=str(TMP), HF_TOKEN=fake, HF_REPO="nobody/none", HF_ENDPOINT="http://127.0.0.1:9", HF_HUB_DISABLE_TELEMETRY="1")
c = subprocess.run(["bash", "runpod/collect.sh"], cwd=W, env=cenv, capture_output=True, text=True)
both = c.stdout + c.stderr
assert c.returncode == 0 and "WARNING: HF upload failed" in both and fake not in both, both
listing = both.split("HF upload:")[0]
assert "raw_qwen3vl_collage.jsonl" in listing and "results/collage_lfm25v.csv" in listing and "images/" not in listing
(W / "logs/leak.log").write_text(f"token={fake}\n")
c = subprocess.run(["bash", "runpod/collect.sh"], cwd=W, env=cenv, capture_output=True, text=True)
both = c.stdout + c.stderr
assert c.returncode == 0 and "HF upload skipped" in both and fake not in both, both
c = subprocess.run(["bash", "runpod/collect.sh"], cwd=W, env=dict(env, OUT_DIR=str(TMP)), capture_output=True, text=True)
assert c.returncode == 0 and "HF upload" not in c.stdout + c.stderr
print("collect.sh OK: new files packed; HF upload failure/token-in-file never fails collection or prints the token")

# 8. the repo is untouched ---------------------------------------------------------------------------------
os.chdir("/")
shutil.rmtree(TMP)
assert git() == git0, "git status of the repo changed during the check"
assert sha(FROZEN) == frozen0, "a frozen file changed"
assert (REPO / "images").exists() == had_images and not list(REPO.glob("*smoke*")), "fake images or smoke files in repo"
print("repo untouched (git status, frozen-file hashes, no images/ or smoke files); temp dir removed")
print("E9 CPU CHECK OK")
