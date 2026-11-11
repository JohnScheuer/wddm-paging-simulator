<div align="center">

# WDDM Paging Collapse Simulator

**Empirical proof that WDDM paging collapses GPU throughput by up to 1013× when VRAM is oversubscribed.**

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

> **Measured on RTX 5060 (8 GB):**
> **68× collapse** (clean system) · **1013× collapse** (RAM under pressure)
> **9.17 → 0.01 TFLOPS in the worst case**

---

## The Physics

| Resource | Bandwidth | Latency |
|----------|-----------|---------|
| GDDR7 VRAM (RTX 5060) | ~448 GB/s | ~10 ns |
| PCIe Gen4 x16 | ~32 GB/s | ~1 µs |
| DDR5 System RAM | ~50 GB/s | ~80 ns |
| NVMe SSD (page file) | ~3-7 GB/s | ~50-100 µs |

When WDDM pages a 64 KB GPU page to system RAM, every subsequent access
incurs a PCIe round-trip. When system RAM is also under pressure, the
OS swaps those pages to the NVMe page file, adding a **second level of
indirection**. The data path becomes:

```
GPU → PCIe → DDR5 (miss) → NVMe (page fault) → DDR5 → PCIe → GPU
```

This turns a memory-bandwidth-bound kernel into an NVMe-latency-bound
kernel — explaining collapses far beyond the naive 14× bandwidth ratio.

---

## Experiment Design

```
Phase 1 (Baseline)              Phase 2 (Stress)
┌──────────────────────┐        ┌──────────────────────┐
│  Filler: 50% VRAM    │        │  Filler: 115% VRAM   │
│  GEMM 2048×2048      │        │  GEMM 2048×2048      │
│  → 9.17 TFLOPS       │        │  → 0.01 TFLOPS       │
│  → 1.87 ms mean      │        │  → 1898.87 ms mean   │
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

Both runs show identical baseline performance, confirming hardware
consistency.

| Metric | Run A | Run B |
|--------|------:|------:|
| Mean | 1.88 ms | 1.87 ms |
| P50 | 1.56 ms | 1.77 ms |
| P99 | 3.01 ms | 2.41 ms |
| Min | 1.37 ms | 1.38 ms |
| Max | 6.53 ms | 4.82 ms |
| Std Dev | 0.82 ms | 0.52 ms |
| **TFLOPS** | **9.13** | **9.17** |

### Phase 2 — Stress (9.28 GB / 117% VRAM → WDDM Paging Active)

| Metric | Run A (Clean RAM) | Run B (RAM Pressure) |
|--------|------------------:|---------------------:|
| Mean | 128.72 ms | **1898.87 ms** |
| P50 | 124.72 ms | **1806.38 ms** |
| P99 | 169.49 ms | **2797.32 ms** |
| Min | 102.15 ms | **1481.79 ms** |
| Max | 171.16 ms | **2817.65 ms** |
| Std Dev | 15.04 ms | **290.13 ms** |
| **TFLOPS** | **0.13** | **0.01** |

### Collapse Summary

| Metric | Run A (Clean) | Run B (Pressure) |
|--------|:-------------:|:-----------------:|
| Mean Latency | **68.4×** | **1013.1×** |
| P99 Latency | **56.3×** | **1161.7×** |
| TFLOPS | **68.4×** | **1013.1×** |
| Std Dev | **18.3×** | **558×** |

### Why Two Different Results?

The two runs were performed on the same hardware with the same benchmark.
The only variable was **system RAM pressure** at the time of the test.

| Factor | Run A (68×) | Run B (1013×) |
|--------|-------------|---------------|
| System RAM free | ~20 GB | ~4 GB |
| WDDM page destination | DDR5 (hit) | DDR5 → NVMe page file (miss) |
| Page fault latency | ~5 µs | ~50-500 µs |
| WDDM eviction target | Filler tensors only | Filler + benchmark tensors |
| Data path | GPU → PCIe → DDR5 → PCIe → GPU | GPU → PCIe → DDR5 → NVMe → DDR5 → PCIe → GPU |

**Run A** represents the best-case paging scenario: the WDDM evicted
filler pages to DDR5, and the benchmark tensors remained in VRAM. The
slowdown came from PCIe contention and TLB pressure.

**Run B** represents the realistic production scenario: system RAM was
already under pressure from other processes, so WDDM-paged data was
itself swapped to the NVMe page file. The GPU stalled for milliseconds
on every page fault, waiting for data to traverse the full storage stack.

> **This non-determinism is the strongest argument for Admission Control.**
> You cannot build SLAs on a system where the same workload produces 68×
> or 1013× slowdown depending on background RAM usage. The only safe
> strategy is to **prevent paging entirely** by rejecting requests before
> physical VRAM is exhausted.

### Why Not the Theoretical 14×?

The naive bandwidth ratio (448 / 32 ≈ 14×) underestimates the real
collapse because it ignores:

1. **Page-fault overhead** — each access to an evicted page triggers a
   GPU trap → WDDM scheduler → DMA copy over PCIe → TLB re-map → resume.
2. **TLB thrashing** — the GPU translation lookaside buffer cannot keep
   up with pages being constantly evicted and re-fetched.
3. **PCIe contention** — the Windows display compositor (DWM) shares the
   same PCIe link, further reducing effective bandwidth.
4. **Second-level paging** — when system RAM is full, paged VRAM data
   is swapped to NVMe, adding 10-100× more latency per fault.
5. **Variance explosion** — standard deviation went from 0.52 ms to
   290.13 ms (558×), indicating unpredictable page-fault storms that
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
"Shared GPU memory" counter spiking during Phase 2. Also monitor
**Memory** to observe the RAM pressure that determines whether you
reproduce Run A or Run B.

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
   (P99: 2.4 ms → 2797 ms for a single GEMM).

3. **The collapse is non-deterministic.**
   The same workload produced 68× or 1013× slowdown depending on
   background RAM usage. You cannot build P99 SLAs on a system with
   this variance profile. The only safe strategy is to prevent paging
   entirely.

4. **Software cannot beat physics.**
   No kernel fusion, quantization, or attention variant recovers a
   1000× bandwidth cliff caused by PCIe + NVMe paging. The only fix
   is to not page.

---

## License

[MIT](LICENSE)

---

## Author

**João Felipe de Souza** — Staff ML Systems Engineer
[github.com/JohnScheuer](https://github.com/JohnScheuer)