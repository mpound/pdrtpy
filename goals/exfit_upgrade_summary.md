# ExcitationFit Upgrade — Implementation Summary

| Branch | PR | Date |
|---|---|---|
| `excitationfit-upgrade` | #218 | 2026-05-27 |
| `excitationfit-chunked` | #220 | 2026-05-27 |

---

## What Was Done

### 1. Robust piecewise initialisation (replaces `_two_lines` / `_first_guess`)

The old `_two_lines` method found an inflection point by a simple heuristic
that failed whenever a single bad data point distorted the slope.  It was
replaced with true piecewise regression.

**New methods in `BaseExcitationFit`:**

| Method | Purpose |
|---|---|
| `_find_breakpoint_ssr(x, y_1d)` | Vectorised prefix-sum SSR minimisation (default) |
| `_find_breakpoint_pelt(x, y_1d)` | PELT via `ruptures` (optional cross-check) |
| `_fit_segment(x_seg, y_seg)` | `np.polyfit` with negative-slope enforcement |
| `_ruptures_partition(x, yr, partition_method)` | Iterates pixels, dispatches to above |
| `_one_line(x, m1, n1)` | Returns `m1*x + n1`; used by `excitationplot.py` |

**Why SSR over PELT:**
PELT is O(n log n) in theory but carries per-call Python↔C setup overhead that
dominates for the small n typical in PDR science (n ≤ 30 rotational levels for
any molecule, usually 5–10).  The vectorised prefix-sum SSR evaluates all
candidate breakpoints in a single NumPy pass with no library overhead.  Benchmark
on the CenA 7-line H₂ map (800 valid pixels): SSR 10.8 s vs PELT 13.4 s (1.24×
faster).  PELT is retained as an optional `partition_method="pelt"` for
cross-checking and for any future case with unusually large n (> ~50).

**SSR formula** (documented in `_find_breakpoint_ssr` docstring):

```
SSR(a,b) = Syy − Sy²/m − (Sxy − SxSy/m)² / (Sxx − Sx²/m)
```

evaluated for every candidate split point via prefix arrays Px, Py, Pxx, Pxy,
Pyy built once per pixel.  Total complexity O(n) per pixel.

### 2. Parallel workers for map fits

`run()` now accepts a `workers` keyword (mirrors `LineRatioFit` API):

```python
fit.run(components=2, workers=-1)   # all CPUs
fit.run(components=2, workers=4)    # 4 processes
fit.run(components=2)               # serial (default)
```

Implementation uses `ProcessPoolExecutor` with:
- `_init_excitation_worker` — builds the lmfit `Model` once per worker process
- `_excitation_pixel_worker` — fits one pixel, returns `(i, result)` tuple
- `_excitation_chunk_worker` — fits a batch of pixels serially, returns list of `(i, result)` (see §3 below)

**Pickling caveat:** lmfit's `Model` requires `func.__name__` to be set.
`functools.partial` objects lack `__name__`, so it must be assigned manually
after creating the partial (`fn.__name__ = base_fn.__name__`).

**Log-sum-exp stability in `_two_comp_model_fn`:**
Two-component model uses `ref = max(a, b); model = ref + log10(1 + 10^(other−ref))`
to avoid overflow/underflow when the two temperature components differ by many
dex.

### 3. Chunked parallel submission (PR #220)

The initial parallel implementation submitted one pixel per
`ProcessPoolExecutor` task.  Per-task IPC overhead (~1–2 ms) exceeded per-pixel
compute time (~10–13 ms) enough that parallel was slower than serial on the CenA
map (~837 valid pixels).

The fix: batch `chunk_size` pixels (default 32) into each task.  Each worker
fits its chunk serially; IPC is paid once per chunk, not once per pixel.

**New module-level function:** `_excitation_chunk_worker(indices, yr_chunk,
sig_chunk, m1s, n1s, m2s, n2s)` — receives a list of pixel indices and
`(n_lines, chunk_size)` data slices, returns `list[(i, ModelResult|None)]`.

**New parameter:** `chunk_size=32` added to `run()`, threaded through
`_run_pixel_fits` → `_run_pixel_fits_parallel`.  The progress bar now tracks
valid pixels only (not the full map including masked pixels).

**Benchmark results (CenA map, 837 valid pixels, 12 CPUs):**

| Mode | Mean time | ms/pixel |
|---|---|---|
| Serial | 17.0 s | 6.4 ms |
| Parallel `chunk_size=32` | 7.8 s | 2.9 ms |
| **Speedup** | **2.2×** | |

Correctness: median relative difference vs master reference = **0.00%** on all
four parameters.

**Chunk size guidance:** default 32 is a good starting point (chunk compute
~320 ms >> IPC ~1–2 ms).  Larger chunks reduce overhead further but coarsen
progress-bar granularity and may cause load imbalance on the last chunk.

### 4. Benchmark script

`scripts/benchmark_excitationfit.py` — times `H2ExcitationFit.run()` on the
CenA 7-line H₂ dataset.  Key flags:

```
--partition-method {ssr,pelt,both}   # compare algorithms
--workers N                          # parallel workers
--chunk-size N                       # pixels per parallel task (default 32)
--runs N                             # repeated timing
--save-reference FILE.npz            # save fitted maps for correctness checks
--compare-reference FILE.npz         # diff current run against saved reference
```

---

## Caveats and Known Limitations

### Parallel workers: overhead vs. speedup threshold

With the original one-pixel-per-task approach, IPC overhead (~1–2 ms per task)
exceeded per-pixel compute time enough that parallel was *slower* than serial on
the CenA map.  The chunked submission (PR #220) fixed this: 2.2× speedup at 837
valid pixels with `chunk_size=32`.  For very small maps (≲ 100 valid pixels) the
process-pool startup cost may still dominate; serial remains the safe default.

### `_one_line` required by `excitationplot.py`

This small helper (`m1*x + n1`) was accidentally removed during the refactor and
restored in a follow-up commit.  `excitationplot.py` lines 270 and 278 call it
to draw the fitted line segments on excitation diagrams.  Do not remove it.

### Masked pixels in CenA test data

The CenA H₂ test map has ~1831–1849 masked pixels out of 2366 total (~77%).
This is **expected**:
- The observed map is geometrically tilted relative to image axes → many pixels
  outside the map boundary.
- Pixels with data ≤ 0 are masked before fitting.
- H200S8 was excluded from the benchmark dataset because it has far more bad
  pixels than S1–S7, causing excessive fit failures.

### Masked pixel count varies slightly run-to-run

Small run-to-run variation (~10–20 pixels) in the masked count is normal; it
reflects edge-case pixels where the fit barely converges or fails depending on
numerical precision and initial parameter guesses.

---

## Why `joint_fit='fast'`/`'hybrid'` Cannot Be Applied Here

`LineRatioFit.run(joint_fit='fast')` fits all map pixels in a single
`scipy.optimize.least_squares` call by exploiting a **block-diagonal Jacobian**:
each pixel's residual rows are non-zero only in its own two columns (`density`,
`radiation_field`), so scipy's `jac_sparsity` + `tr_solver='lsmr'` handles the
full map as cheaply as independent per-pixel solves while avoiding the overhead of
launching separate minimiser instances.

Three structural preconditions make this possible in `LineRatioFit` that do not
hold for `H2ExcitationFit`:

**1. A vectorisable model.**
`LineRatioFit` evaluates the model via a pre-computed 2D interpolator
(`_interp_lin`) that accepts an `(n_valid, 2)` array and returns all pixel
predictions in one NumPy call.  The excitation model is analytic:

```
ln(N_u / g_u) = f(T, N_col, E_u, A_ul, Z(T))
```

with partition function `Z(T)`, transition-specific constants `E_u` and `A_ul`,
and a two-component log-sum-exp combination.  There is no pre-computed grid to
interpolate; the function must be called inside the minimiser's inner loop.
Building a single joint residual over all pixels would require a Python loop
there, negating the benefit of the joint solve.

**2. A fixed, small parameter count per pixel.**
`LineRatioFit` always has exactly 2 parameters per pixel, giving a perfectly
regular block-diagonal structure.  A two-component excitation fit has 4–6 free
parameters per pixel (`T_cold`, `N_cold`, `T_hot`, `N_hot`, optionally OPR and
A_V).  While a variable block size is not fatal in principle, it would require
bypassing lmfit entirely and hand-coding the residual vector, sparsity matrix, and
per-pixel covariance extraction from the local Jacobian block — a high-risk
rewrite.

**3. A cheap coarse initial guess.**
`LineRatioFit`'s joint solver is seeded from a fast coarse per-pixel fit.  The
excitation fit's initial guess already comes from the SSR piecewise search, which
is per-pixel.  There is no equivalent two-stage pipeline to adapt.

**Conclusion:** joint fitting is architecturally incompatible with
`H2ExcitationFit` as currently structured.  The correct performance lever is
**chunked parallel submission** (see Future Work below), which amortises
per-task IPC overhead without changing the fitting model or residual structure.

---

## Future Work (Deferred)

These items were discussed but deliberately left off this branch:

| Item | Notes |
|---|---|
| **Chunked parallel submission** | ✅ Done (PR #220) — `chunk_size=32` default; 2.2× speedup on CenA map |
| **Tikhonov / spatial regularisation** | Requires moving from per-pixel `minimize()` to a single joint minimisation with sparse Jacobian (Option 4 in `regularization.md`).  Per-pixel independence is a hard architectural barrier to regularisation. |
| **`components=None` auto-selection** | Automatically choose 1 vs 2 components via AIC/BIC or F-test on the piecewise fit quality |
| **More than 2 components** | Piecewise framework supports n breakpoints; lmfit model extension straightforward |
| **CHplusExcitationFit** | Marked experimental; needs validation against real data |

---

## Files Changed

| File | Change |
|---|---|
| `pdrtpy/tool/excitation.py` | Core implementation (new methods, `workers`, `chunk_size`, `partition_method`, log-sum-exp model, `_one_line` restore) |
| `pyproject.toml` | Added `ruptures` to dependencies |
| `uv.lock` | Updated (ruptures 1.1.10) |
| `scripts/benchmark_excitationfit.py` | Benchmark script (`--partition-method`, `--workers`, `--chunk-size`, `--runs`, `--save/compare-reference`) |
| `scripts/bench_excitation_ref_master.npz` | Reference solution from master branch |
