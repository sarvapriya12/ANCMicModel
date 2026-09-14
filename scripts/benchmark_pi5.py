#!/usr/bin/env python3
"""
Raspberry Pi 5 Performance & Real-Time Benchmark
Measures DSP and FastEnhancer ONNX streaming latency, RTF, and jitter.
"""

import sys
import time
from pathlib import Path

import numpy as np

# Add src to path
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "src"))

from highspl.dsp.fdaf import BattlefieldFDAF
from highspl.dsp.nlms import NLMS, VSSNLMS, RobustNLMS
from highspl.models.fastenhancer import FastEnhancerAdapter, FastEnhancerConfig


def benchmark_stream(
    num_frames: int = 500,
    block_size: int = 512,
    sample_rate: int = 48000,
    dsp_mode: str = "ROBUST",
    warmup_frames: int = 50,
):
    frame_duration_ms = (block_size / sample_rate) * 1000.0

    print("=" * 68)
    print("      RASPBERRY PI 5 HIGH-SPL STREAMING BENCHMARK")
    print("=" * 68)
    print(f"Sample Rate     : {sample_rate} Hz")
    print(f"Block Size      : {block_size} samples ({frame_duration_ms:.2f} ms/frame)")
    print(f"DSP Pre-filter  : {dsp_mode}")
    print(f"Benchmark Frames: {num_frames} (+ {warmup_frames} warmup)")

    # 1. Initialize DSP Filter
    if dsp_mode == "ROBUST":
        dsp_filter = RobustNLMS(filter_length=256)
        dsp_state = dsp_filter.reset()
    elif dsp_mode == "VSS":
        dsp_filter = VSSNLMS(filter_length=256)
        dsp_state = dsp_filter.reset()
    elif dsp_mode == "CLASSICAL":
        dsp_filter = NLMS(filter_length=256)
        dsp_state = dsp_filter.reset()
    elif dsp_mode == "FDAF":
        dsp_filter = BattlefieldFDAF(block_size=128, num_partitions=2)
        dsp_state = dsp_filter.reset()
    elif dsp_mode == "BYPASS":
        dsp_filter = None
        dsp_state = None
    else:
        raise ValueError(f"Unknown DSP mode: {dsp_mode}")

    # 2. Initialize FastEnhancer ONNX
    onnx_path = ROOT_DIR / "models" / "fastenhancer" / "b" / "fastenhancer_b_streaming.onnx"
    if not onnx_path.exists():
        print(f"\n[ERROR] Model not found at: {onnx_path}")
        return

    config = FastEnhancerConfig(
        model_kwargs={"hop_size": block_size, "sample_rate": sample_rate},
        onnx_path=str(onnx_path),
        backend="onnxruntime",
        device="cpu",
    )
    model = FastEnhancerAdapter(config)
    ai_state = model.reset()

    # Generate synthetic audio stream
    rng = np.random.default_rng(42)
    prim_data = rng.normal(0.0, 0.05, (num_frames + warmup_frames, block_size)).astype(np.float32)
    ref_data = rng.normal(0.0, 0.05, (num_frames + warmup_frames, block_size)).astype(np.float32)

    dsp_latencies = []
    ai_latencies = []
    total_latencies = []

    print("\nRunning streaming benchmark...")
    for frame_idx in range(num_frames + warmup_frames):
        chunk_prim = prim_data[frame_idx]
        chunk_ref = ref_data[frame_idx]

        # Stage 1: DSP
        t0 = time.perf_counter()
        if dsp_filter is not None:
            filtered_prim, dsp_state = dsp_filter.process(chunk_prim, chunk_ref, dsp_state)
        else:
            filtered_prim = chunk_prim
        t1 = time.perf_counter()

        # Stage 2: AI
        _out_chunk, ai_state = model.process(filtered_prim, sample_rate=sample_rate, state=ai_state)
        t2 = time.perf_counter()

        # Only record after warmup
        if frame_idx >= warmup_frames:
            dsp_ms = (t1 - t0) * 1000.0
            ai_ms = (t2 - t1) * 1000.0
            tot_ms = (t2 - t0) * 1000.0

            dsp_latencies.append(dsp_ms)
            ai_latencies.append(ai_ms)
            total_latencies.append(tot_ms)

    dsp_arr = np.array(dsp_latencies)
    ai_arr = np.array(ai_latencies)
    tot_arr = np.array(total_latencies)

    mean_total = np.mean(tot_arr)
    rtf = mean_total / frame_duration_ms
    p95_total = np.percentile(tot_arr, 95)
    _p99_total = np.percentile(tot_arr, 99)

    print("\n" + "-" * 68)
    print("                      RESULTS BREAKDOWN")
    print("-" * 68)
    print(f"Stage 1 (DSP - {dsp_mode:<9s}): Mean: {np.mean(dsp_arr):.2f} ms | P95: {np.percentile(dsp_arr, 95):.2f} ms")
    print(f"Stage 2 (FastEnhancer-B): Mean: {np.mean(ai_arr):.2f} ms | P95: {np.percentile(ai_arr, 95):.2f} ms")
    print("-" * 68)
    print(f"Total Processing Time   : Mean: {mean_total:.2f} ms | P50: {np.median(tot_arr):.2f} ms | P95: {p95_total:.2f} ms")
    print(f"Frame Budget (Deadline) : {frame_duration_ms:.2f} ms")
    print(f"Real-Time Factor (RTF)  : {rtf:.3f}")
    print(f"CPU Headroom Available  : {max(0.0, (1.0 - rtf) * 100.0):.1f}%")
    print("-" * 68)

    if rtf < 0.65:
        print("[VERDICT] PASS - EXCELLENT REAL-TIME PERFORMANCE ON RASPBERRY PI 5!")
    elif rtf < 1.0:
        print("[VERDICT] PASS - REAL-TIME CAPABLE (Monitor peak system load)")
    else:
        print("[VERDICT] FAIL - EXCEEDS FRAME DEADLINE (Drop block size or optimize)")
    print("=" * 68)


if __name__ == "__main__":
    mode = "ROBUST"
    if len(sys.argv) > 1:
        mode = sys.argv[1].upper()
    benchmark_stream(dsp_mode=mode)
