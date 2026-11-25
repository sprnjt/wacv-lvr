"""Recreate every result of the CultureHaystack camera-ready paper from the raw model replies, on a CPU, in a few
minutes, and compare with the files shipped in this package. Nothing in the package is modified: everything runs in a
scratch copy.

    pip install -r tools/requirements.txt
    python tools/reproduce_all.py            # prints PASS / FAIL per item and a summary; exit code 1 on any FAIL

What is checked (model inference itself needs GPUs and is not rerun; the raw replies are the starting point):
  A  file integrity        every file against tools/PACKAGE_SHA256.txt
  B  pre-registration      SHA-256 of prereg/PREREG_MANIFEST.sha256 equals the hash printed in the paper;
                           manifest entries against main_study/ (expected post-freeze changes listed in README)
  C  main study            raw replies -> scored files (ch.py score), byte-identical, six models
  D  main-study tables     analyze.py + error_breakdown/review_checks -> results/*.csv (17 tables), numerically equal
  E  independent audit     main_study/verify_results.py (recomputes headline numbers without analyze.py)
  F  numbers in the text   gpu_results / error_breakdown / review_checks macros vs every \\gpudef in the paper preamble
  G  full pool (884)       analyze.py on the cluster package -> results/*.csv; G1, G2, G3, F1 extension tables
  H  Gemma-3-12B row       rescoring + table1_g3.py
  I  camera-ready tables   cr analyses 01, 02, 04, 05, 06, 07 (the CLIP check 03 needs the Kaggle images; see README)
  J  Table 2               every cell typed in main.tex against the regenerated CSVs
"""
import glob, hashlib, json, os, re, shutil, subprocess, sys, tempfile
from pathlib import Path

import numpy as np
import pandas as pd

PKG = Path(__file__).resolve().parents[1]
PY = sys.executable
MODELS = ["qwen3vl", "qwen25vl", "gemma4", "gemma3_4b", "granite4v", "lfm25v"]
FP_MODELS = ["gemma3_4b", "lfm25v", "qwen25vl", "qwen3vl", "gemma4"]
RESULTS = []


def report(item, ok, detail=""):
    RESULTS.append((item, ok))
    print(f"{'PASS' if ok else 'FAIL'}  {item}" + (f"  ({detail})" if detail else ""), flush=True)


def run(cmd, cwd, env=None):
    e = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", **(env or {})}
    p = subprocess.run(cmd, cwd=cwd, env=e, capture_output=True, text=True)
    if p.returncode != 0:
        print(f"      command failed in {cwd}: {' '.join(map(str, cmd))}\n" + p.stderr[-1500:])
    return p.returncode == 0


def same_table(a_path, b_path, tol=1e-9):
    """Numerically equal tables (same shape and text columns; numbers within tol)."""
    if os.path.getsize(a_path) < 5 or os.path.getsize(b_path) < 5:
        return os.path.getsize(a_path) < 5 and os.path.getsize(b_path) < 5, "empty"
    a, b = pd.read_csv(a_path), pd.read_csv(b_path)
    if a.shape != b.shape or list(a.columns) != list(b.columns):
        return False, f"shape {a.shape} vs {b.shape}"
    num = a.select_dtypes("number").columns
    d = float(np.nanmax((a[num] - b[num]).abs().to_numpy())) if len(num) else 0.0
    d = 0.0 if np.isnan(d) else d
    txt = a.drop(columns=num).astype(str).equals(b.drop(columns=num).astype(str))
    return d <= tol and txt, f"max |diff| {d:.2g}"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def gpudefs(text):
    out = {}
    for m in re.finditer(r"\\gpudef\{([^}]*)\}\{", text):
        i = j = m.end(); depth = 1
        while depth:
            depth += {"{": 1, "}": -1}.get(text[j], 0); j += 1
        out[m.group(1)] = text[i:j - 1]
    return out


def main():
    # ---- A integrity
    lines = (PKG / "tools/PACKAGE_SHA256.txt").read_text().split("\n")
    bad = [l.split("  ", 1)[1] for l in lines if l.strip() and sha(PKG / l.split("  ", 1)[1]) != l.split("  ", 1)[0]]
    report("A  file integrity", not bad, f"{len([l for l in lines if l.strip()])} files" + (f"; changed: {bad[:5]}" if bad else ""))

    # ---- B pre-registration
    h = sha(PKG / "prereg/PREREG_MANIFEST.sha256")
    printed = re.search(r"\\def\\prereghash\{([0-9a-f]*)\}", (PKG / "paper/latex/preamble.tex").read_text()).group(1)
    report("B  printed hash = SHA-256 of the manifest", h == printed, h[:16] + "...")
    ok = changed = absent = 0
    for l in (PKG / "prereg/PREREG_MANIFEST.sha256").read_text().splitlines():
        m = re.match(r"^([0-9a-f]{64})  \./(.+)$", l)
        if not m:
            continue
        f = PKG / "main_study" / m.group(2)
        if not f.exists():
            absent += 1
        elif sha(f) == m.group(1):
            ok += 1
        else:
            changed += 1
    report("B  manifest vs main_study", (ok, changed, absent) == (14, 6, 42), f"{ok} identical, {changed} changed after the freeze, "
           f"{absent} not in the repo (pilot outputs, plan.md, metadata.csv); see README for the 6")
    snap = [sha(PKG / "prereg/snapshot" / f) for f in ("ch.py", "run_models.py")]
    man = (PKG / "prereg/PREREG_MANIFEST.sha256").read_text()
    report("B  snapshot ch.py / run_models.py match the manifest", all(s in man for s in snap))

    tmp = Path(tempfile.mkdtemp(prefix="culturehaystack_check_"))
    work = tmp / "pkg"
    print(f"\nscratch copy: {work}")
    shutil.copytree(PKG, work, ignore=shutil.ignore_patterns("__pycache__", ".DS_Store"))
    for e in json.loads((work / "tools/links.json").read_text()):
        link, target = work / e["link"], work / e["target"]
        link.parent.mkdir(parents=True, exist_ok=True)
        if not link.exists():
            link.symlink_to(os.path.relpath(target, link.parent))

    # ---- C main-study rescoring
    ms = work / "main_study"
    for m in MODELS:
        run([PY, "ch.py", "score", m, "--trials", "trials_main.jsonl"], ms)
        report(f"C  rescore {m}", sha(ms / f"scored_{m}.csv") == sha(PKG / "main_study" / f"scored_{m}.csv"))

    # ---- D main-study tables
    shutil.rmtree(ms / "results"); (ms / "results").mkdir()
    shutil.copytree(PKG / "main_study/results/figures", ms / "results/figures")
    run([PY, "analyze.py", *MODELS], ms)
    out = tmp / "macros"
    for s in ("error_breakdown", "review_checks", "gpu_results"):
        run([PY, f"work/analysis/{s}.py"], ms, {"PAPER_OUT": str(out)})
    for f in sorted(glob.glob(str(PKG / "main_study/results/*.csv"))):
        ok_, d = same_table(ms / "results" / Path(f).name, f)
        report(f"D  main_study/results/{Path(f).name}", ok_, d)

    # ---- E independent audit
    report("E  verify_results.py", run([PY, "verify_results.py"], ms))

    # ---- F every number in the paper text
    paper = gpudefs((PKG / "paper/latex/preamble.tex").read_text())
    regen = {}
    for f in glob.glob(str(out / "sec/gen_*.tex")):
        regen.update(gpudefs(Path(f).read_text()))
    diff = [k for k in paper if regen.get(k) != paper[k]]
    report("F  in-text numbers (\\gpu macros)", not diff, f"{len(paper)} compared, {len(diff)} differ" + (f": {diff[:5]}" if diff else ""))

    # ---- G full pool
    fp = work / "post_registration/fullpool_package"
    shipped = {Path(f).name: f for f in glob.glob(str(PKG / "post_registration/fullpool_package/results/*.csv"))}
    run([PY, str(ms / "analyze.py"), *FP_MODELS], fp, {"PYTHONPATH": str(ms)})
    home = tmp / "home"; (home / "CultureHaystack_handoff").mkdir(parents=True)
    (home / "CultureHaystack_handoff/wacv-lvr").symlink_to(ms)
    (home / "CultureHaystack_handoff/fullpool").symlink_to(fp)
    for part in ("g1", "g2", "g3", "f1"):                  # one process per part: each starts its bootstrap RNG at seed 0
        run([PY, "ext/analyze_ext.py", part], fp, {"HOME": str(home)})
    run([PY, "ext/analyze_g2_full.py", "g2"], fp, {"HOME": str(home)})
    for name, f in sorted(shipped.items()):
        ok_, d = same_table(fp / "results" / name, f)
        report(f"G  fullpool_package/results/{name}", ok_, d)

    # ---- H Gemma-3-12B
    g3 = work / "post_registration/gemma3_12b_table1"
    run([PY, "ch.py", "score", "gemma3_12b", "--trials", "trials_main.jsonl"], g3 / "work")
    report("H  rescore gemma3_12b", sha(g3 / "work/scored_gemma3_12b.csv") ==
           sha(PKG / "post_registration/gemma3_12b_table1/work/scored_gemma3_12b.csv"))
    for m in ("gemma3_12b", "gemma3_4b"):
        run([PY, "../table1_g3.py", m], g3 / "work")
    for f in ("acc_by_N_gemma3_12b", "scaling_summary_gemma3_12b", "acc_by_N_gemma3_4b", "scaling_summary_gemma3_4b"):
        ok_, d = same_table(g3 / f"{f}.csv", PKG / f"post_registration/gemma3_12b_table1/{f}.csv")
        report(f"H  {f}.csv", ok_, d)

    # ---- I camera-ready analyses
    cr = work / "post_registration/camera_ready_analyses"
    for script, cwd, args in (("01_fullpool_ci/fullpool_ci.py", "_fullpool", []), ("02_g2_same_image/g2_same.py", "_fullpool", []),
                              ("04_holm_mc/holm_mc.py", "_frozen", []), ("05_slot_tier/slot_tier.py", "_frozen", []),
                              ("06_format_errors/format_errors.py", "_frozen", []), ("07_scorer_edges/scorer_edges.py", "_frozen", []),
                              ("07_scorer_edges/scorer_edges.py", "_frozen", ["incl_overrides"])):
        run([PY, f"../{script}", *args], cr / cwd)
    for d in ("01_fullpool_ci", "02_g2_same_image", "04_holm_mc", "05_slot_tier", "06_format_errors", "07_scorer_edges"):
        for f in sorted(glob.glob(str(PKG / "post_registration/camera_ready_analyses" / d / "*.csv"))):
            ok_, dd = same_table(cr / d / Path(f).name, f)
            report(f"I  {d}/{Path(f).name}", ok_, dd)

    # ---- J Table 2 cells (typed in main.tex) against the regenerated CSVs
    tex = (PKG / "paper/latex/main.tex").read_text()
    blk = tex[tex.index(r"\label{tab:models}"):tex.index(r"\label{tab:headline}")]
    acc = pd.concat([pd.read_csv(ms / "results/acc_by_N.csv"), pd.read_csv(g3 / "acc_by_N_gemma3_12b.csv").assign(model="gemma3_12b")])
    sm = pd.concat([pd.read_csv(ms / "results/scaling_summary.csv"),
                    pd.read_csv(g3 / "scaling_summary_gemma3_12b.csv").assign(model="gemma3_12b")])
    rows = {r"\gfour": "gemma4", r"\qthree": "qwen3vl", r"\qtwo": "qwen25vl", r"\gthree": "gemma3_4b",
            r"\gran": "granite4v", r"\lfm": "lfm25v", "Gemma-3-12B": "gemma3_12b"}
    for line in blk.split("\n"):
        key = next((k for k in rows if line.strip().startswith(k) and "&" in line), None)
        if key is None:
            continue
        m = rows[key]
        clean = re.sub(r"\\mathbf\{([^}]*)\}", r"\1", line.replace("$-$", "-").replace(r"\phantom{-}", "")).replace("$", "")
        got = [float(x) for x in re.findall(r"-?\d+\.\d+", clean)]
        a = acc[(acc.model == m) & (acc["mode"] == "mc") & (acc.metric == "joint")].set_index("N")
        o = acc[(acc.model == m) & (acc["mode"] == "open") & (acc.metric == "joint")].set_index("N")
        s = sm[(sm.model == m) & (sm["mode"] == "mc")].iloc[0]
        exp = [a.acc[1], a.lo[1], a.hi[1], a.acc[5], a.lo[5], a.hi[5], a.acc[1] - a.acc[5], s.initial_drop, s.drop_lo,
               s.drop_hi, s.slope_per_image, s.slope_lo, s.slope_hi, o.acc[1]]
        off = [(g, round(e, 3)) for g, e in zip(got, exp) if abs(g - round(e, 3)) > 0.0011]
        report(f"J  Table 2 row {m}", len(got) == 14 and not off, f"{len(got)} cells" + (f"; off: {off}" if off else ""))

    n_fail = sum(not ok for _, ok in RESULTS)
    print(f"\n{len(RESULTS) - n_fail} of {len(RESULTS)} checks pass." + (" ALL PASS" if not n_fail else f" {n_fail} FAIL"))
    print(f"(scratch copy left at {tmp} for inspection; delete it when done)")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
