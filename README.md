<div align="center">

# WDDM Paging Collapse Simulator

**Empirical proof that WDDM paging collapses GPU throughput by 68× when VRAM is oversubscribed.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch 2.1+](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-13.0-76b900.svg)](https://developer.nvidia.com/cuda-toolkit)

</div>

---

## TL;DR

When a Windows GPU's physical VRAM is exceeded, the WDDM driver silently
pages memory to system RAM over PCIe. This benchmark **quantifies** the
resulting performance collapse and provides empirical evidence that LLM
runtimes (vLLM, TGI, etc.) need **strict Admission Control** at the
physical-VRAM boundary — not at the `torch.cuda.memory_reserved()` boundary.

> **Measured on RTX 5060 (8 GB): 68.4× latency collapse · 9.13 → 0.13 TFLOPS**

---

## The Physics

| Resource | Bandwidth | Latency |
|----------|-----------|---------|
| GDDR7 VRAM (RTX 5060) | ~448 GB/s | ~10 ns |
| PCIe Gen4 x16 | ~32 GB/s | ~1 µs |
| DDR5 System RAM | ~50 GB/s | ~80 ns |

When WDDM pages a 64 KB GPU page to system RAM, every subsequent access
incurs a PCIe round-trip. For an LLM forward pass that streams model
weights from VRAM, this turns a memory-bandwidth-bound kernel into a
PCIe-bandwidth-bound kernel. The theoretical bandwidth ratio alone
predicts ~14× slowdown; real-world page-fault overhead, TLB thrashing,
and PCIe contention push it to **68×**.

---

## Experiment Design

```
Phase 1 (Baseline)              Phase 2 (Stress)
┌──────────────────────┐        ┌──────────────────────┐
│  Filler: 50% VRAM    │        │  Filler: 115% VRAM   │
│  GEMM 2048×2048      │        │  GEMM 2048×2048      │
│  → 9.13 TFLOPS       │        │  → 0.13 TFLOPS       │
│  → 1.88 ms mean      │        │  → 128.72 ms mean    │
└──────────────────────┘        └──────────────────────┘
     All data in GDDR7              ~1.35 GB paged to DDR5
                                    via PCIe Gen4 x16
```

1. **Baseline** — Fill VRAM to 50%, run a GEMM benchmark.
2. **Stress** — Fill VRAM to 115% of physical capacity, forcing WDDM
   to evict pages to system RAM. Re-run the same GEMM.
3. **Compare** — Report mean / P50 / P99 latency and TFLOPS collapse.

---

## Results

### Hardware & Environment

| Component | Specification |
|-----------|---------------|
| GPU | NVIDIA GeForce RTX 5060 |
| VRAM | 7.93 GB GDDR7 |
| OS | Windows 11 + WSL2 (Ubuntu 22.04) |
| Driver | CUDA 13.0 |
| PyTorch | 2.13.0+cu130 |
| Workload | GEMM 2048×2048 float32 (17.2 GFLOP/iter) |

### Phase 1 — Baseline (4.0 GB / 50% VRAM)

| Metric | Value |
|--------|-------|
| Mean | 1.88 ms |
| P50 | 1.56 ms |
| P99 | 3.01 ms |
| Min | 1.37 ms |
| Max | 6.53 ms |
| Std Dev | 0.82 ms |
| **TFLOPS** | **9.13** |

### Phase 2 — Stress (9.28 GB / 117% VRAM → WDDM Paging Active)

| Metric | Value |
|--------|-------|
| Mean | 128.72 ms |
| P50 | 124.72 ms |
| P99 | 169.49 ms |
| Min | 102.15 ms |
| Max | 171.16 ms |
| Std Dev | 15.04 ms |
| **TFLOPS** | **0.13** |

### Collapse Summary

| Metric | Baseline | Stress | Collapse |
|--------|----------|--------|----------|
| Mean Latency | 1.88 ms | 128.72 ms | **68.4×** |
| P99 Latency | 3.01 ms | 169.49 ms | **56.3×** |
| TFLOPS | 9.13 | 0.13 | **68.4×** |
| Std Dev | 0.82 ms | 15.04 ms | **18.3×** |

### Why 68× and Not the Theoretical 14×?

The naive bandwidth ratio (448 / 32 ≈ 14×) underestimates the real
collapse because it ignores:

1. **Page-fault overhead** — each access to an evicted page triggers a
   GPU trap → WDDM scheduler → DMA copy over PCIe → TLB re-map → resume.
2. **TLB thrashing** — the GPU translation lookaside buffer cannot keep
   up with pages being constantly evicted and re-fetched.
3. **PCIe contention** — the Windows display compositor (DWM) shares the
   same PCIe link, further reducing effective bandwidth.
4. **Variance explosion** — standard deviation went from 0.82 ms to
   15.04 ms (18×), indicating unpredictable page-fault storms that
   destroy LLM tail latency.

---

## Requirements

| Component | Minimum |
|-----------|---------|
| GPU | NVIDIA, ≥ 6 GB VRAM |
| OS | Windows 10 21H2+ or Windows 11 |
| Driver | NVIDIA Game Ready / Studio ≥ 530.x |
| Python | 3.10+ |
| PyTorch | 2.1+ with CUDA |

> **WSL2 note:** This benchmark was validated on WSL2 with CUDA 13.0
> passthrough. The WDDM driver on the Windows host allowed
> oversubscription and paging occurred as expected. If you see OOM
> instead of slowdown, run natively on Windows.

---

## Quick Start

```bash
git clone https://github.com/JohnScheuer/wddm-paging-simulator.git
cd wddm-paging-simulator
pip install -r requirements.txt
python simulate_collapse.py
```

With explicit parameters:

```bash
python simulate_collapse.py --vram-limit-gb 8 --fill-stress-pct 1.20
```

### Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--vram-limit-gb` | auto | Physical VRAM to target |
| `--matrix-dim` | 2048 | GEMM dimension (N×N) |
| `--iterations` | 50 | Benchmark iterations per phase |
| `--fill-baseline-pct` | 0.50 | VRAM fill ratio for baseline |
| `--fill-stress-pct` | 1.15 | VRAM fill ratio for stress (>1.0 = oversubscribe) |
| `--yes` | off | Skip confirmation prompt |

---

## Monitoring

Run in a second terminal during the benchmark:

```bash
nvidia-smi dmon -s u -d 1
```

On Windows, check **Task Manager → Performance → GPU** for the
"Shared GPU memory" counter spiking during Phase 2.

---

## Key Takeaways

1. **`nvidia-smi` is the source of truth.**
   `torch.cuda.memory_reserved()` reflects PyTorch's caching allocator,
   not physical VRAM. In our test, PyTorch reported 9.28 GB allocated
   on a 7.93 GB GPU — the delta was silently paged by WDDM.

2. **Admission Control is non-negotiable.**
   LLM runtimes must reject new requests *before* total KV-cache +
   model weights exceed **physical** VRAM. Reactive eviction is too
   late — the first paged forward pass already destroys tail latency
   (P99: 3 ms → 169 ms for a single GEMM).

3. **Software cannot beat physics.**
   No kernel fusion, quantization, or attention variant recovers a
   68× bandwidth cliff caused by PCIe paging. The only fix is to
   not page.

---

## License

[MIT](LICENSE)

---

## Author

**João Felipe de Souza** — Staff ML Systems Engineer
[github.com/JohnScheuer](https://github.com/JohnScheuer)