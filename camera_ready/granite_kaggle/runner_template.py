# CultureHaystack: Granite-Vision-4.1-4B runs for the camera-ready (Kaggle, 2x T4).
# Prompts come from the ORIGINAL ch.py / run_models.py (frozen SHA-256 abe1da12... / 90d48114...)
# and the cluster's split-query wrapper ch_g2.py (7d2b1834...), embedded below byte for byte.
import base64, csv, filecmp, glob, hashlib, json, os, subprocess, sys, tempfile, time, urllib.request
from pathlib import Path

ACCOUNT = __ACCOUNT__          # 1..3; this kernel runs shards 2*(ACCOUNT-1) and 2*(ACCOUNT-1)+1
NSHARDS = 6
JOB = "__JOB__"                # "gate" or "run"
DTYPE = "__DTYPE__"            # "bfloat16" or "float16" (run mode); gate tries both
HF = "https://huggingface.co/datasets/shinjinii/culturehaystack-results/resolve/main/"
W = Path(os.environ.get("CH_WORK", "/tmp/ch")); OUT = Path(os.environ.get("CH_OUT", "/kaggle/working"))
GPUQ = __GPUQ__              # per-GPU queue of (chunk, part, number_of_parts)
FILES = {"ch.py": "__CH__", "run_models.py": "__RM__", "ch_g2.py": "__G2__"}
SHA = {"ch.py": "abe1da122d7c25ef955a359c619f9db673ceefeb2ac8ca90857a2d0851c70059",
       "run_models.py": "90d481144c1c661e26bae1811b26c812551d11fd1ea8f970a8f0002a813119cb",
       "ch_g2.py": "7d2b18341c36fa5685b0fb6b0f705fb00fb8c125d44e851d4faadaf5b9d4a43e"}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def setup():
    W.mkdir(parents=True, exist_ok=True); (W / "data").mkdir(exist_ok=True)
    for n, b in FILES.items():
        data = base64.b64decode(b)
        assert hashlib.sha256(data).hexdigest() == SHA[n], n
        (W / n).write_bytes(data)
    log("code written, hashes OK")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "torch==2.9.1", "torchvision==0.24.1",
                    "--index-url", "https://download.pytorch.org/whl/cu126"], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers==5.20.1", "accelerate==1.15.0"], check=True)
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchaudio"], check=False)  # Kaggle's build mismatches torch 2.9.1; unused
    log("pip done")
    get = {"data/pool.csv": "items/main_study/pool.csv", "data/lists.json": "items/main_study/lists.json",
           "data/targets.csv": "items/main_study/targets.csv", "exclude.txt": "items/main_study/exclude.txt",
           "trials_main.jsonl": "trials/main_study/trials_main.jsonl", "trials_full.jsonl": "trials/full_pool/trials_full.jsonl",
           "trials_g1.jsonl": "trials/full_pool/trials_g1.jsonl", "trials_g2_160.jsonl": "trials/full_pool/trials_g2_160.jsonl",
           "raw_granite4v_orig.jsonl": "models/granite4v/main_study/raw_granite4v.jsonl"}
    for dst, src in get.items():
        urllib.request.urlretrieve(HF + src, W / dst)
    log("data downloaded")
    os.chdir(W); sys.path.insert(0, str(W))
    import ch
    meta = glob.glob(os.environ.get("CH_DATASET", "/kaggle/input") + "/**/metadata.csv", recursive=True)
    assert meta, "attach the world-cultural-items-1k dataset"
    d = Path(tempfile.mkdtemp())
    ch.build_data(meta[0], out=d)
    ok = filecmp.cmp(d / "pool.csv", "data/pool.csv", shallow=False) and filecmp.cmp(d / "lists.json", "data/lists.json", shallow=False)
    log("VERSION_CHECK", "PASS" if ok else "FAIL")
    assert ok, "Kaggle dataset version differs from the one used in the paper"
    ch.prep(str(Path(meta[0]).parent), out="images/prepared")
    n = len(list(Path("images/prepared").glob("img_*.jpg"))); log("PREPARED_IMAGES", n); assert n == 890


def tasks():
    import ch
    core = [t for t in ch.read_jsonl("trials_full.jsonl") if (t["exp"] == "E1" and t["N"] <= 4) or t["exp"] == "E2"]
    g1 = ch.read_jsonl("trials_g1.jsonl"); g2 = ch.read_jsonl("trials_g2_160.jsonl")
    assert (len(core), len(g1), len(g2)) == (7956, 1720, 1600), (len(core), len(g1), len(g2))
    return [("core", t) for t in core] + [("g1", t) for t in g1] + [("g2", t) for t in g2]


_LOAD_LOCK = __import__("threading").Lock()


def load(dtype, gpu=0):
  with _LOAD_LOCK:
    import torch, transformers
    from transformers import AutoProcessor
    hf_id = "ibm-granite/granite-vision-4.1-4b"
    proc = AutoProcessor.from_pretrained(hf_id)
    model = transformers.AutoModelForMultimodalLM.from_pretrained(
        hf_id, dtype=getattr(torch, dtype), device_map={"": gpu})
    bad = {str(p.dtype) for p in model.parameters()} - {str(getattr(torch, dtype))}
    log(f"loaded on gpu {gpu}: parameter dtypes ok" if not bad else f"WARNING mixed dtypes {bad}")
    return model, proc


def worker(shard, dtype, gpu, part=0, parts=1):
    os.chdir(W); sys.path.insert(0, str(W))
    import ch, ch_g2, run_models  # noqa: F401  (ch_g2 patches ch.make_parts for trials with "ask")
    mine = [x for i, x in enumerate(tasks()) if i % NSHARDS == shard][part::parts]
    out = OUT / f"raw_granite4v_shard{shard}_p{part}of{parts}.jsonl"
    done = {(r["task"], r["trial_id"]) for r in ch.read_jsonl(out)} if out.exists() else set()
    model, proc = load(dtype, gpu)
    t0 = time.time(); n = 0
    with open(out, "a", encoding="utf-8") as f:
        for task, t in mine:
            if (task, t["trial_id"]) in done:
                continue
            raw = run_models.generate(model, proc, "granite", t, "mc", 64)
            f.write(json.dumps(dict(task=task, trial_id=t["trial_id"], mode="mc", raw=raw, dtype=dtype)) + "\n"); f.flush()
            n += 1
            if n == 1:
                log(f"shard {shard} p{part}: first answer {raw!r}")
            if n % 20 == 0:
                log(f"shard {shard} p{part}: {n + len(done)}/{len(mine)}  {(time.time() - t0) / n:.2f} s/call")
    log(f"shard {shard} DONE {len(mine)}")


def gate():
    os.chdir(W); sys.path.insert(0, str(W))
    import torch, ch, run_models
    orig = {(r["trial_id"], r["mode"]): r["raw"] for r in ch.read_jsonl("raw_granite4v_orig.jsonl")}
    rep = [t for t in ch.read_jsonl("trials_main.jsonl") if t["kind"] == "main"][:10]
    five = [t for t in ch.read_jsonl("trials_full.jsonl") if t["exp"] == "E2"][:4]
    res = {"gpu": torch.cuda.get_device_name(0), "n_gpu": torch.cuda.device_count(), "torch": torch.__version__}
    for dt in ("bfloat16", "float16"):
        try:
            model, proc = load(dt)
            rows, same = [], 0
            for t in rep:
                for mode in ("open", "mc"):
                    r = run_models.generate(model, proc, "granite", t, mode, 64)
                    same += r == orig[(t["trial_id"], mode)]
                    rows.append(dict(trial_id=t["trial_id"], mode=mode, raw=r))
            t0 = time.time()
            for t in five:
                run_models.generate(model, proc, "granite", t, "mc", 64)
            res[dt] = dict(identical=f"{same}/20", sec_per_N5_call=round((time.time() - t0) / len(five), 2))
            (OUT / f"gate_{dt}.jsonl").write_text("".join(json.dumps(x) + "\n" for x in rows))
            del model; torch.cuda.empty_cache()
        except Exception as e:
            res[dt] = f"ERROR {type(e).__name__}: {str(e)[:300]}"
        log("GATE", dt, res[dt])
    (OUT / "gate_result.json").write_text(json.dumps(res, indent=1)); log("GATE_RESULT", json.dumps(res))


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--queue":      # child process: one model, one GPU
        os.chdir(W); sys.path.insert(0, str(W))
        for (sq, pq, Pq) in json.loads(sys.argv[2]):
            worker(sq, DTYPE, 0, pq, Pq)
        sys.exit(0)
    setup()
    import torch, threading
    from huggingface_hub import snapshot_download
    ngpu = torch.cuda.device_count()
    log("GPUs:", ngpu, [torch.cuda.get_device_name(i) for i in range(ngpu)])
    if JOB == "gate":
        gate()
    else:
        snapshot_download("ibm-granite/granite-vision-4.1-4b"); log("model cached")
        procs = []
        for g, q in enumerate(GPUQ):
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g % ngpu))
            procs.append(subprocess.Popen([sys.executable, os.path.abspath(__file__), "--queue", json.dumps(q)], env=env))
            time.sleep(20)   # stagger the loads
        rc = [p.wait() for p in procs]
        log("worker exit codes", rc)
        log("ALL DONE")
