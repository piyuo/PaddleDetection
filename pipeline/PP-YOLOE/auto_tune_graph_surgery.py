#!/usr/bin/env python3
"""
Auto-tuner for PP-YOLOE ONNX graph surgery options.

This script runs staged search over transformation toggles in
`graph_surgery_compare.py` and reports the best configuration based on
Modified avg latency measured with ONNX Runtime (CoreML EP by default).

Outputs:
- Per-trial logs under the run directory
- A CSV summary of all trials
- best_config.json capturing the chosen flags
- A copy of the best model as best/best_final.onnx

Example:
  python3 pipeline/PP-YOLOE/auto_tune_graph_surgery.py \
    --model output/ppyoloe_crn_s_36e_pphuman/best_model.onnx \
    --input-shape 1,3,640,640 --ep coreml --runs 10 --warmup 3 \
    --split-candidates 10,8,6 --keep-outputs ppyoloe_output1,ppyoloe_output2

Notes:
- This performs a greedy staged search to keep runtime reasonable. Use
  --full-grid to exhaustively search a compact space (may be slow).
- Results on macOS with CoreML EP may differ from iPhone. Treat this as a
  proxy; validate the best model on-device.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, asdict
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional, Tuple


GRAPH_SCRIPT = Path(__file__).with_name("graph_surgery_compare.py")


class TunerLogger:
    """Simple logger that writes to stdout and a tuner.log file under outdir."""
    def __init__(self, outdir: Path):
        self.outdir = outdir
        self.log_path = outdir / "tuner.log"
        try:
            outdir.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

    def log(self, msg: str) -> None:
        line = str(msg)
        print(line)
        try:
            with open(self.log_path, "a") as f:
                f.write(line + "\n")
        except Exception:
            # best effort; ignore file write errors
            pass


@dataclass
class TrialConfig:
    rewrite_hardswish: bool = False
    split_concat: Optional[int] = None  # None means disabled
    fold_static_shapes: bool = False
    rewrite_div: bool = False
    rewrite_pow: bool = False
    fp16: bool = False
    keep_outputs: Optional[str] = None  # comma-separated names or None

    def to_flags(self) -> List[str]:
        flags: List[str] = ["--fix-input-shapes"]
        if self.rewrite_hardswish:
            flags.append("--rewrite-hardswish")
        if self.split_concat is not None:
            flags += ["--split-concat", str(self.split_concat)]
        if self.fold_static_shapes:
            flags.append("--fold-static-shapes")
        if self.rewrite_div:
            flags.append("--rewrite-div")
        if self.rewrite_pow:
            flags.append("--rewrite-pow")
        if self.fp16:
            flags.append("--fp16")
        if self.keep_outputs:
            flags += ["--keep-outputs", self.keep_outputs]
        return flags


@dataclass
class TrialResult:
    config: TrialConfig
    baseline_avg_ms: float
    modified_avg_ms: float
    avg_delta_pct: float
    outdir: str
    final_model: Optional[str]
    coreml_supported_nodes: Optional[int] = None
    coreml_partitions: Optional[int] = None


_RE_BASELINE = re.compile(r"Baseline\s+Latency\s*\(ms\)\s*avg=\s*([0-9]+\.?[0-9]*)", re.IGNORECASE)
_RE_MODIFIED = re.compile(r"Modified\s+Latency\s*\(ms\)\s*avg=\s*([0-9]+\.?[0-9]*)", re.IGNORECASE)
_RE_DELTA = re.compile(r"avg\s*delta:\s*([-+]?\d+\.?\d*)%", re.IGNORECASE)
_RE_FINAL = re.compile(r"Final model.*?:\s*(.+\.onnx)")
_RE_COREML_NODES = re.compile(r"CoreML.*nodes supported.*?:\s*(\d+)", re.IGNORECASE)
_RE_COREML_PARTS = re.compile(r"CoreML.*partitions.*?:\s*(\d+)", re.IGNORECASE)


def run_trial(base_model: str, input_shape: str, ep: str, warmup: int, runs: int,
              outdir: Path, trial_name: str, cfg: TrialConfig,
              extra_args: Optional[List[str]] = None,
              logger: Optional[TunerLogger] = None) -> TrialResult:
    trial_outdir = outdir / trial_name
    trial_outdir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, str(GRAPH_SCRIPT),
        "--model", base_model,
        "--input-shape", input_shape,
        "--ep", ep,
        "--warmup", str(warmup),
        "--runs", str(runs),
        "--outdir", str(trial_outdir),
    ] + cfg.to_flags()
    if extra_args:
        cmd += extra_args

    # Debug: print a compact config summary
    if logger:
        logger.log(f"[trial] {trial_name}")
        logger.log(f"        flags: {' '.join(cfg.to_flags())}")
        if extra_args:
            logger.log(f"        extra_args: {' '.join(extra_args)}")

    log_path = trial_outdir / "trial.log"
    with open(log_path, "w") as logf:
        proc = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT, text=True)

    # Read log to parse metrics
    text = log_path.read_text(errors="ignore")
    baseline_match = _RE_BASELINE.search(text)
    modified_match = _RE_MODIFIED.search(text)
    delta_match = _RE_DELTA.search(text)
    final_match = _RE_FINAL.search(text)

    if not (baseline_match and modified_match):
        raise RuntimeError(
            f"Failed to parse latency metrics in {log_path}.\n"
            f"Command: {' '.join(cmd)}\n"
            f"Tail: {text[-600:]}"
        )

    baseline = float(baseline_match.group(1))
    modified = float(modified_match.group(1))
    delta = float(delta_match.group(1)) if delta_match else ((modified - baseline) / baseline * 100.0)
    final_model = final_match.group(1).strip() if final_match else None

    # Optional CoreML support info
    nodes_supported = None
    partitions = None
    m_nodes = _RE_COREML_NODES.search(text)
    if m_nodes:
        try:
            nodes_supported = int(m_nodes.group(1))
        except Exception:
            nodes_supported = None
    m_parts = _RE_COREML_PARTS.search(text)
    if m_parts:
        try:
            partitions = int(m_parts.group(1))
        except Exception:
            partitions = None

    return TrialResult(
        config=cfg,
        baseline_avg_ms=baseline,
        modified_avg_ms=modified,
        avg_delta_pct=delta,
        outdir=str(trial_outdir),
        final_model=final_model,
        coreml_supported_nodes=nodes_supported,
        coreml_partitions=partitions,
    )


def write_summary(results: List[TrialResult], outdir: Path) -> None:
    csv_path = outdir / "summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "trial",
            "baseline_avg_ms",
            "modified_avg_ms",
            "avg_delta_pct",
            "rewrite_hardswish",
            "split_concat",
            "fold_static_shapes",
            "rewrite_div",
            "rewrite_pow",
            "fp16",
            "keep_outputs",
            "coreml_supported_nodes",
            "coreml_partitions",
            "final_model",
        ])
        for idx, r in enumerate(results):
            writer.writerow([
                f"trial_{idx:03d}",
                f"{r.baseline_avg_ms:.3f}",
                f"{r.modified_avg_ms:.3f}",
                f"{r.avg_delta_pct:.2f}",
                r.config.rewrite_hardswish,
                r.config.split_concat if r.config.split_concat is not None else "",
                r.config.fold_static_shapes,
                r.config.rewrite_div,
                r.config.rewrite_pow,
                r.config.fp16,
                r.config.keep_outputs or "",
                r.coreml_supported_nodes if r.coreml_supported_nodes is not None else "",
                r.coreml_partitions if r.coreml_partitions is not None else "",
                r.final_model or "",
            ])


def copy_best_artifacts(best: TrialResult, outdir: Path) -> None:
    best_dir = outdir / "best"
    best_dir.mkdir(parents=True, exist_ok=True)
    # Copy best model
    if best.final_model and os.path.isfile(best.final_model):
        dest = best_dir / "best_final.onnx"
        shutil.copy2(best.final_model, dest)
    # Save best config
    with open(best_dir / "best_config.json", "w") as f:
        json.dump(asdict(best.config), f, indent=2)
    # Save best metrics
    metrics = {
        "baseline_avg_ms": best.baseline_avg_ms,
        "modified_avg_ms": best.modified_avg_ms,
        "avg_delta_pct": best.avg_delta_pct,
        "coreml_supported_nodes": best.coreml_supported_nodes,
        "coreml_partitions": best.coreml_partitions,
        "final_model": best.final_model,
    }
    with open(best_dir / "best_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)


def greedy_staged_search(base_model: str, input_shape: str, ep: str, warmup: int, runs: int,
                         outdir: Path, keep_outputs: Optional[str], split_candidates: List[int],
                         try_fp16: bool, extra_args: Optional[List[str]], logger: TunerLogger) -> Tuple[List[TrialResult], TrialResult]:
    results: List[TrialResult] = []
    best_cfg = TrialConfig()

    def eval_cfg(name: str, cfg: TrialConfig) -> TrialResult:
        r = run_trial(base_model, input_shape, ep, warmup, runs, outdir, name, cfg, extra_args, logger)
        results.append(r)
        # Immediate feedback on metrics
        logger.log(
            f"[result] {name}: modified={r.modified_avg_ms:.3f} ms, baseline={r.baseline_avg_ms:.3f} ms, "
            f"delta={r.avg_delta_pct:.2f}%, parts={r.coreml_partitions}, nodes={r.coreml_supported_nodes}"
        )
        return r

    # Stage overview
    stage_docs = {
        0: "Baseline (no transforms beyond fix-input-shapes)",
        1: "HardSwish rewrite toggle (off/on)",
        2: "Concat split sweep (limit inputs to N)",
        3: "Fold static shape chains (off/on)",
        4: "Arithmetic rewrites (Div-by-const / Pow patterns)",
        5: "FP16 cast toggle (off/on) if enabled",
        6: "Outputs pruning (off/on) if keep-outputs provided",
    }

    # Planned steps for progress estimate
    total_est = 1  # stage 0
    total_est += 2  # stage 1
    total_est += 1 + len(split_candidates)  # stage 2
    total_est += 2  # stage 3
    total_est += 4  # stage 4
    total_est += (2 if try_fp16 else 1)  # stage 5 (we still evaluate current best once)
    total_est += (2 if keep_outputs else 1)  # stage 6 (evaluate current best once)
    logger.log(f"[tuner] Staged search plan: ~{total_est} trials")
    logger.log("[tuner] Stage meanings:")
    for k in sorted(stage_docs.keys()):
        logger.log(f"  - Stage {k}: {stage_docs[k]}")

    trial_counter = 0

    # Stage 0: baseline (no extra passes besides fix-input-shapes)
    logger.log("\n[stage 0] Baseline")
    baseline_res = eval_cfg("stage0_baseline", best_cfg)
    best = baseline_res
    best_cfg = best.config
    trial_counter += 1
    logger.log(f"[stage 0] done. best modified avg = {best.modified_avg_ms:.3f} ms (delta {best.avg_delta_pct:.2f}%)")

    # Stage 1: HardSwish rewrite
    logger.log("\n[stage 1] HardSwish rewrite toggle (0=off, 1=on)")
    for hs in [False, True]:
        step = 1 if not hs else 2
        cfg = TrialConfig(**asdict(best_cfg))
        cfg.rewrite_hardswish = hs
        logger.log(f"[stage 1] step {step}/2 -> hs={int(hs)}")
        r = eval_cfg(f"stage1_hs_{int(hs)}", cfg)
        if r.modified_avg_ms < best.modified_avg_ms:
            best = r
            best_cfg = cfg
        trial_counter += 1
    logger.log(f"[stage 1] done. best modified avg = {best.modified_avg_ms:.3f} ms")

    # Stage 2: split concat sweep
    candidates = [None] + split_candidates
    logger.log(f"\n[stage 2] Concat split sweep over {candidates}")
    total_steps = len(candidates)
    for sc in [None] + split_candidates:
        idx = candidates.index(sc) + 1
        cfg = TrialConfig(**asdict(best_cfg))
        cfg.split_concat = sc
        logger.log(f"[stage 2] step {idx}/{total_steps} -> split_concat={sc if sc is not None else 'none'}")
        r = eval_cfg(f"stage2_sc_{'none' if sc is None else sc}", cfg)
        if r.modified_avg_ms < best.modified_avg_ms:
            best = r
            best_cfg = cfg
        trial_counter += 1
    logger.log(f"[stage 2] done. best modified avg = {best.modified_avg_ms:.3f} ms")

    # Stage 3: fold static shapes
    logger.log("\n[stage 3] Fold static shape chains toggle (0=off, 1=on)")
    for fs in [False, True]:
        step = 1 if not fs else 2
        cfg = TrialConfig(**asdict(best_cfg))
        cfg.fold_static_shapes = fs
        logger.log(f"[stage 3] step {step}/2 -> fold_static_shapes={int(fs)}")
        r = eval_cfg(f"stage3_fold_{int(fs)}", cfg)
        if r.modified_avg_ms < best.modified_avg_ms:
            best = r
            best_cfg = cfg
        trial_counter += 1
    logger.log(f"[stage 3] done. best modified avg = {best.modified_avg_ms:.3f} ms")

    # Stage 4: arithmetic rewrites combos
    logger.log("\n[stage 4] Arithmetic rewrites combinations: none/div/pow/both")
    combos = [
        (False, False, "none"),
        (True, False, "div"),
        (False, True, "pow"),
        (True, True, "both"),
    ]
    total_steps = len(combos)
    for i, (div, pow_, tag) in enumerate(combos, start=1):
        cfg = TrialConfig(**asdict(best_cfg))
        cfg.rewrite_div = div
        cfg.rewrite_pow = pow_
        logger.log(f"[stage 4] step {i}/{total_steps} -> rewrite_div={int(div)} rewrite_pow={int(pow_)}")
        r = eval_cfg(f"stage4_ar_{tag}", cfg)
        if r.modified_avg_ms < best.modified_avg_ms:
            best = r
            best_cfg = cfg
        trial_counter += 1
    logger.log(f"[stage 4] done. best modified avg = {best.modified_avg_ms:.3f} ms")

    # Stage 5: FP16 optional
    if try_fp16:
        logger.log("\n[stage 5] FP16 casting toggle (0=off, 1=on)")
        for i, f16 in enumerate([False, True], start=1):
            cfg = TrialConfig(**asdict(best_cfg))
            cfg.fp16 = f16
            logger.log(f"[stage 5] step {i}/2 -> fp16={int(f16)}")
            r = eval_cfg(f"stage5_fp16_{int(f16)}", cfg)
            if r.modified_avg_ms < best.modified_avg_ms:
                best = r
                best_cfg = cfg
            trial_counter += 1
        logger.log(f"[stage 5] done. best modified avg = {best.modified_avg_ms:.3f} ms")

    # Stage 6: keep outputs if provided (test off vs on)
    if keep_outputs:
        logger.log("\n[stage 6] Outputs pruning toggle (0=off, 1=on)")
        for i, keep_on in enumerate([False, True], start=1):
            cfg = TrialConfig(**asdict(best_cfg))
            cfg.keep_outputs = keep_outputs if keep_on else None
            logger.log(f"[stage 6] step {i}/2 -> keep_outputs={'on' if keep_on else 'off'}")
            r = eval_cfg(f"stage6_keep_{int(keep_on)}", cfg)
            if r.modified_avg_ms < best.modified_avg_ms:
                best = r
                best_cfg = cfg
            trial_counter += 1
        logger.log(f"[stage 6] done. best modified avg = {best.modified_avg_ms:.3f} ms")

    return results, best


def full_grid_search(base_model: str, input_shape: str, ep: str, warmup: int, runs: int,
                     outdir: Path, keep_outputs: Optional[str], split_candidates: List[int],
                     try_fp16: bool, extra_args: Optional[List[str]], logger: TunerLogger) -> Tuple[List[TrialResult], TrialResult]:
    results: List[TrialResult] = []
    best: Optional[TrialResult] = None

    hs_opts = [False, True]
    sc_opts: List[Optional[int]] = [None] + split_candidates
    fs_opts = [False, True]
    ar_opts = [
        (False, False, "none"),
        (True, False, "div"),
        (False, True, "pow"),
        (True, True, "both"),
    ]
    fp16_opts = [False, True] if try_fp16 else [False]
    keep_opts = [None, keep_outputs] if keep_outputs else [None]

    # Compute total combinations for progress info
    total = len(hs_opts) * len(sc_opts) * len(fs_opts) * len(ar_opts) * len(fp16_opts) * len(keep_opts)
    logger.log(f"[tuner] Full grid: {total} combinations")
    idx = 0
    for hs, sc, fs, (dv, pw, _), fp16, ko in product(hs_opts, sc_opts, fs_opts, ar_opts, fp16_opts, keep_opts):
        cfg = TrialConfig(
            rewrite_hardswish=hs,
            split_concat=sc,
            fold_static_shapes=fs,
            rewrite_div=dv,
            rewrite_pow=pw,
            fp16=fp16,
            keep_outputs=ko,
        )
        tag = f"grid_{idx:04d}"
        logger.log(f"[grid] {idx+1}/{total} -> {tag} hs={int(hs)} sc={'none' if sc is None else sc} fs={int(fs)} div={int(dv)} pow={int(pw)} fp16={int(fp16)} keep={'on' if ko else 'off'}")
        r = run_trial(base_model, input_shape, ep, warmup, runs, outdir, tag, cfg, extra_args, logger)
        results.append(r)
        logger.log(
            f"[result] {tag}: modified={r.modified_avg_ms:.3f} ms, baseline={r.baseline_avg_ms:.3f} ms, "
            f"delta={r.avg_delta_pct:.2f}%, parts={r.coreml_partitions}, nodes={r.coreml_supported_nodes}"
        )
        if best is None or r.modified_avg_ms < best.modified_avg_ms:
            best = r
        idx += 1

    assert best is not None
    return results, best


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Auto-tune PP-YOLOE ONNX graph surgery toggles")
    p.add_argument("--model", required=True, help="Path to the baseline ONNX model")
    p.add_argument("--input-shape", required=True, help="Input shape as 'N,C,H,W'")
    p.add_argument("--ep", default="coreml", help="Execution provider: coreml/cpu/cuda/etc.")
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--runs", type=int, default=10)
    p.add_argument("--outdir", default=None, help="Output directory for the tuning run")
    p.add_argument("--keep-outputs", default=None, help="Comma-separated output tensor names to keep; if set, tuner will try both with and without keeping")
    p.add_argument("--split-candidates", default="10,8,6", help="Comma-separated list of max inputs for Concat split sweep")
    p.add_argument("--no-fp16", action="store_true", help="Do not try FP16 variations")
    p.add_argument("--full-grid", action="store_true", help="Run a full grid search (may be slow)")
    p.add_argument("--extra-args", default=None, help="Extra args to pass to graph_surgery_compare.py (quoted string)")
    args = p.parse_args(argv)
    return args


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    model = args.model
    input_shape = args["input-shape"] if isinstance(args, dict) else args.input_shape
    ep = args.ep
    warmup = int(args.warmup)
    runs = int(args.runs)
    keep_outputs = args.keep_outputs
    try_fp16 = not args.no_fp16
    split_candidates = [int(x) for x in str(args.split_candidates).split(',') if x]
    extra_args = None
    if args.extra_args:
        # Split respecting spaces; simple split is fine for our needs
        extra_args = args.extra_args.strip().split()

    if not Path(model).is_file():
        print(f"Model not found: {model}", file=sys.stderr)
        return 2
    if not GRAPH_SCRIPT.is_file():
        print(f"graph_surgery_compare.py not found at {GRAPH_SCRIPT}", file=sys.stderr)
        return 2

    # Prepare run dir
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_stem = Path(model).stem
    outdir = Path(args.outdir) if args.outdir else Path("pipeline/output/auto_tune") / f"{model_stem}_{ts}"
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Auto-tuning: model={model} shape={input_shape} ep={ep}")
    print(f"Warmup={warmup} Runs={runs}")
    print(f"Run dir: {outdir}")

    logger = TunerLogger(outdir)
    if args.full_grid:
        results, best = full_grid_search(model, input_shape, ep, warmup, runs, outdir, keep_outputs, split_candidates, try_fp16, extra_args, logger)
    else:
        results, best = greedy_staged_search(model, input_shape, ep, warmup, runs, outdir, keep_outputs, split_candidates, try_fp16, extra_args, logger)

    # Summarize
    results_sorted = sorted(results, key=lambda r: r.modified_avg_ms)
    write_summary(results_sorted, outdir)
    copy_best_artifacts(best, outdir)

    print(f"Trials: {len(results)}")
    print(f"Best modified avg: {best.modified_avg_ms:.3f} ms (delta {best.avg_delta_pct:.2f}%)")
    print(f"Best config: {json.dumps(asdict(best.config), indent=2)}")
    print(f"Best model: {best.final_model}")
    print(f"Summary CSV: {outdir / 'summary.csv'}")
    print(f"Best artifacts: {outdir / 'best'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
