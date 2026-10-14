# SoftVector Benchmarks

This directory contains micro-benchmarks for the SoftVector library. They exercise
the public `extern "C"` API that an instruction-set simulator calls to execute RVV
vector instructions against a vector register file.

The benchmarks target the *non-legacy* hot path: the templated element-dispatcher
kernels in `src/target/riscv/softvector.cpp` and the soft-float helpers in
`lib/softfloat-spike`. (The legacy `vpu/` `SVector`/`SVElement` code is excluded
by design — see the project's own notes.)

## Building

Configure a build with the benchmark driver enabled:

```sh
cd /path/to/softvector/build
cmake .. -DSOFTVECTOR_BUILD_BENCH=ON -DCMAKE_BUILD_TYPE=Release
make svbench
```

The resulting binary, `svbench`, is placed in the build directory (next to
`libsoftvector.so`). You can also build the library and `svbench` from an
existing build dir with `make svbench`.

## Running

```
./svbench [--smoke] [--filter substr] [--vlen bits] [--reps N]
```

- `--smoke` : short pass over every case (for a quick sanity check).
- `--filter` : only run cases whose name contains `substr`, e.g. `--filter vadd_vv`.
- `--vlen` : vector length in bits (default 1024 for the large-lane runs used here).
- `--reps` : iterations per case (increase for stable timings; `1000000` is a good
  default with `--vlen 1024`).

Always run with `LC_ALL=C` so timing output uses a dot decimal separator.

## Output

```
case                              ns/op        Gop/s      ns/elem     checksum
----                              -----        -----      -------     --------
vadd_vv_sew8                       7.60      131.529        0.059 15363809258288937741
...
```

- `ns/op`   : wall-clock nanoseconds per vector-instruction issue.
- `Gop/s`   : billions of instruction issues per second.
- `ns/elem` : nanoseconds per vector element (throughput density).
- `checksum`: a deterministic hash of the resulting vector-register contents. It is
  recomputed on every run so a change between builds signals a semantics change.

## Profiling

Record a profile across all cases with a single `--smoke` pass, then inspect:

```sh
perf record -g ./svbench --smoke --vlen 1024
perf report
```

`--smoke` runs every registered case briefly so `perf` produces a meaningful
sample without taking minutes.

> **Note:** `perf report` can stall trying to resolve debug info. If so, run it with
> `DEBUGINFOD_URLS="" perf report`.

## Results: compile-time (NTTP) integer dispatch

The dispatchers used to select the per-element operation at **run time** through a
function pointer. That forced an indirect call on every vector element and prevented
GCC from vectorising the element loop. The dispatchers were reworked so that the
operation is carried as a compile-time **non-type template parameter** (`auto Op`),
and the SEW/mask selection is done by a single small `dispatch_sew_mask<Sign>` /
`dispatch_sew_mask_wide<Sign>` helper. Each distinct operation is now a separate
template instantiation in which the per-element op is inlined and the element loop
auto-vectorises. This was the "easy" win: it only costs code size / compile time
(accepted trade-off).

Measured at `--vlen 1024 --reps 1000000` (128 lanes for `sew8`):

| case          | before (fn-ptr) | after (NTTP) | speedup |
|---------------|-----------------|--------------|---------|
| `vadd_vv_sew8` | ~176 ns/op      | ~7.6 ns/op   | ~23x    |
| `vand_vv_sew8` | ~146 ns/op      | ~5.5 ns/op   | ~27x    |
| `vxor_vv_sew8` | —               | ~5.5 ns/op   | —       |
| `vsub_vv_sew8` | —               | ~7.1 ns/op   | —       |

After the change the heavyweight integer ops (`vdiv`, `vredsum`) are the residual
integer cost, and the remaining profile time is dominated by SoftFloat conversion /
rounding kernels for the floating-point cases. `perf report` shows the integer
ALU ops inlined directly into per-op `dispatch_sew_mask<...>` instantiations with
no indirect calls.

All 46 golden unit tests pass after the refactor (0 failures), and checksums are
bit-identical across runs, confirming the optimization is semantics-neutral.
