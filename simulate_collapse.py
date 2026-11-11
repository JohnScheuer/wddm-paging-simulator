#!/usr/bin/env python3
"""
WDDM Paging Collapse Simulator
===============================
Demonstrates the catastrophic performance collapse when GPU VRAM is
oversubscribed and the WDDM driver pages memory to system RAM over PCIe.

Target: NVIDIA RTX 5060 (8 GB) or similar consumer GPU on Windows 10/11.
"""

import torch
import time
import sys
import os
import statistics
import argparse


def detect_wsl() -> bool:
    """Return True if running inside WSL2."""
    try:
        with open("/proc/version", "r") as f:
            return "microsoft" in f.read().lower()
    except FileNotFoundError:
        return False


def print_vram_snapshot(label: str = ""):
    """Print current VRAM usage from PyTorch's perspective."""
    alloc = torch.cuda.memory_allocated(0) / (1024 ** 3)
    reserved = torch.cuda.memory_reserved(0) / (1024 ** 3)
    total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    tag = f" [{label}]" if label else ""
    print(
        f"  VRAM{tag}:  "
        f"allocated={alloc:.2f} GB  "
        f"reserved={reserved:.2f} GB  "
        f"physical={total:.2f} GB"
    )


def fill_vram(target_bytes: int, block_mb: int = 128) -> list:
    """
    Allocate tensors on the GPU until target_bytes are consumed.
    Uses torch.empty + .zero_() so every page is actually committed.
    """
    tensors = []
    elems_per_block = (block_mb * 1024 * 1024) // 4
    committed = 0

    while committed < target_bytes:
        try:
            t = torch.empty(elems_per_block, device="cuda", dtype=torch.float32)
            t.zero_()
            torch.cuda.synchronize()
            tensors.append(t)
            committed += elems_per_block * 4
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                print(f"  CUDA OOM after {committed / (1024**3):.2f} GB committed.")
                break
            raise

    return tensors


def benchmark_matmul(dim: int, iterations: int = 50, warmup: int = 10) -> list:
    """
    Run a square GEMM (dim x dim, float32) and return per-iteration
    wall-clock times in seconds.
    """
    a = torch.randn(dim, dim, device="cuda", dtype=torch.float32)
    b = torch.randn(dim, dim, device="cuda", dtype=torch.float32)

    for _ in range(warmup):
        torch.matmul(a, b)
    torch.cuda.synchronize()

    times = []
    for _ in range(iterations):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        torch.matmul(a, b)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)

    del a, b
    torch.cuda.empty_cache()
    return times


def report_times(times: list, label: str) -> dict:
    """Pretty-print latency stats and return them as a dict."""
    ms = [t * 1000 for t in times]
    ms_sorted = sorted(ms)
    p99_idx = max(0, int(len(ms_sorted) * 0.99) - 1)

    stats = {
        "mean": statistics.mean(ms),
        "p50": statistics.median(ms),
        "p99": ms_sorted[p99_idx],
        "std": statistics.stdev(ms) if len(ms) > 1 else 0.0,
        "min": ms_sorted[0],
        "max": ms_sorted[-1],
    }

    print(f"\n  -- {label} --")
    print(f"  Mean : {stats['mean']:8.2f} ms")
    print(f"  P50  : {stats['p50']:8.2f} ms")
    print(f"  P99  : {stats['p99']:8.2f} ms")
    print(f"  Min  : {stats['min']:8.2f} ms")
    print(f"  Max  : {stats['max']:8.2f} ms")
    print(f"  Std  : {stats['std']:8.2f} ms")
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="WDDM Paging Collapse Simulator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "WARNING: Oversubscribing VRAM may make your system nearly "
            "unresponsive.\nClose browsers, games, and other GPU apps first."
        ),
    )
    parser.add_argument(
        "--vram-limit-gb", type=float, default=None,
        help="Physical VRAM in GB (auto-detected if omitted)",
    )
    parser.add_argument(
        "--matrix-dim", type=int, default=2048,
        help="Square matrix dimension for GEMM benchmark (default 2048)",
    )
    parser.add_argument(
        "--iterations", type=int, default=50,
        help="Benchmark iterations per phase (default 50)",
    )
    parser.add_argument(
        "--fill-baseline-pct", type=float, default=0.50,
        help="VRAM fill ratio for baseline phase (default 0.50)",
    )
    parser.add_argument(
        "--fill-stress-pct", type=float, default=1.15,
        help="VRAM fill ratio for stress phase (default 1.15)",
    )
    parser.add_argument(
        "--yes", action="store_true",
        help="Skip the interactive confirmation prompt",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. Check drivers and PyTorch build.")
        sys.exit(1)

    props = torch.cuda.get_device_properties(0)
    total_gb = props.total_memory / (1024 ** 3)
    vram_limit = args.vram_limit_gb if args.vram_limit_gb else total_gb
    flops_per_iter = 2 * args.matrix_dim ** 3

    print("=" * 62)
    print("   WDDM PAGING COLLAPSE SIMULATOR")
    print("=" * 62)
    print(f"  GPU           {torch.cuda.get_device_name(0)}")
    print(f"  VRAM          {total_gb:.2f} GB  (limit used: {vram_limit:.2f} GB)")
    print(f"  GEMM          {args.matrix_dim}x{args.matrix_dim}  float32  "
          f"({flops_per_iter / 1e9:.1f} GFLOP/iter)")
    print(f"  Iterations    {args.iterations}")
    print(f"  PyTorch       {torch.__version__}   CUDA {torch.version.cuda}")
    print("=" * 62)

    if detect_wsl():
        print(
            "\n  [!] WSL2 DETECTED\n"
            "     The CUDA runtime in WSL2 communicates with the host WDDM\n"
            "     driver through a paravirtualized layer.  Behaviour may differ\n"
            "     from native Windows:\n"
            "       - The runtime may throw OOM *before* WDDM can page.\n"
            "       - If you see OOM instead of slowdown, run natively on Windows.\n"
        )

    if not args.yes:
        print(
            "\n  [!] This test intentionally oversubscribes GPU memory.\n"
            "     Your display may lag or freeze temporarily.\n"
        )
        try:
            input("  Press ENTER to continue (Ctrl-C to abort) ... ")
        except (KeyboardInterrupt, EOFError):
            print("\n  Aborted.")
            sys.exit(0)

    # PHASE 1 - BASELINE
    fill_baseline = int(vram_limit * args.fill_baseline_pct * 1024 ** 3)
    print(f"\n{'=' * 62}")
    print(f"  PHASE 1 - BASELINE  "
          f"(fill ~{fill_baseline / (1024**3):.1f} GB / "
          f"{args.fill_baseline_pct:.0%} of VRAM)")
    print(f"{'=' * 62}")

    torch.cuda.empty_cache()
    baseline_fillers = fill_vram(fill_baseline)
    print_vram_snapshot("after fill")

    print(f"\n  Running GEMM benchmark ...")
    baseline_times = benchmark_matmul(args.matrix_dim, args.iterations)
    b_stats = report_times(baseline_times, "Baseline")
    b_tflops = (flops_per_iter / (b_stats["mean"] / 1000)) / 1e12
    print(f"  TFLOPS (mean): {b_tflops:.2f}")

    # PHASE 2 - STRESS
    fill_stress_total = int(vram_limit * args.fill_stress_pct * 1024 ** 3)
    fill_additional = fill_stress_total - fill_baseline
    print(f"\n{'=' * 62}")
    print(f"  PHASE 2 - STRESS  "
          f"(fill ~{fill_stress_total / (1024**3):.1f} GB / "
          f"{args.fill_stress_pct:.0%} of VRAM -> force paging)")
    print(f"{'=' * 62}")
    print(f"  Adding ~{fill_additional / (1024**3):.1f} GB more ...")

    stress_fillers = fill_vram(fill_additional)
    print_vram_snapshot("after oversubscribe")
    print(
        "\n  [i] If 'allocated' exceeds physical VRAM above, WDDM is paging\n"
        "     those bytes to system RAM over PCIe.  Run\n"
        "       nvidia-smi dmon -s u -d 1\n"
        "     in another terminal to watch GPU utilisation drop."
    )

    print(f"\n  Running GEMM benchmark (may be very slow) ...")
    try:
        stress_times = benchmark_matmul(args.matrix_dim, args.iterations)
        s_stats = report_times(stress_times, "Stress (paged)")
        s_tflops = (flops_per_iter / (s_stats["mean"] / 1000)) / 1e12
        print(f"  TFLOPS (mean): {s_tflops:.2f}")

        collapse_mean = s_stats["mean"] / b_stats["mean"]
        collapse_p99 = s_stats["p99"] / b_stats["p99"] if b_stats["p99"] > 0 else float("inf")
        collapse_tflops = b_tflops / s_tflops if s_tflops > 0 else float("inf")

        print(f"\n{'=' * 62}")
        print("  RESULTS")
        print(f"{'=' * 62}")
        print(f"  Mean latency collapse   {collapse_mean:7.1f}x slower")
        print(f"  P99  latency collapse   {collapse_p99:7.1f}x slower")
        print(f"  TFLOPS collapse         {collapse_tflops:7.1f}x lower")

        if collapse_mean >= 5:
            print(
                "\n  [OK] CONFIRMED - catastrophic paging collapse.\n"
                "     Root cause: WDDM evicting VRAM pages to system RAM.\n"
                "     PCIe Gen4 x16 ~ 32 GB/s  vs  GDDR6 ~ 450+ GB/s\n"
                "     -> ~15-30x bandwidth cliff -> LLM token latency explodes.\n"
                "     -> Admission Control must reject requests BEFORE this point."
            )
        elif collapse_mean >= 1.5:
            print(
                "\n  [!] Moderate slowdown detected.\n"
                "     WDDM paging may be partially active, or the CUDA runtime\n"
                "     is absorbing some pressure via its own eviction policy.\n"
                "     Try increasing --fill-stress-pct or running on native Windows."
            )
        else:
            print(
                "\n  [i] No significant collapse.\n"
                "     Likely causes:\n"
                "       - WSL2 CUDA runtime enforced OOM before WDDM could page.\n"
                "       - Driver/OS config prevents oversubscription.\n"
                "       - VRAM was not truly oversubscribed (check nvidia-smi).\n"
                "     Run natively on Windows for the definitive test."
            )

    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            print(
                f"\n  CUDA OOM: {exc}\n\n"
                "  [i] The CUDA runtime refused the allocation instead of letting\n"
                "     WDDM page.  This is the expected behaviour on WSL2 and on\n"
                "     some Windows driver configurations.\n\n"
                "  To observe WDDM paging:\n"
                "     1. Run this script natively on Windows (not WSL2).\n"
                "     2. Ensure 'Hardware-accelerated GPU scheduling' is ON in\n"
                "        Windows Settings -> Display -> Graphics.\n"
                "     3. Try a lower --fill-stress-pct (e.g. 1.05)."
            )
        else:
            raise

    print(f"\n  Cleaning up ...")
    del baseline_fillers, stress_fillers
    torch.cuda.empty_cache()
    print_vram_snapshot("after cleanup")
    print("  Done.\n")


if __name__ == "__main__":
    main()
