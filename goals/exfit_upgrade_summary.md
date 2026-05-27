# ExcitationFit Upgrade — Implementation Summary

Branch: `excitationfit-upgrade`
PR: #218
Date completed: 2026-05-27

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
- `_excitation_pixel_worker` — fits one pixel, returns parameter dict

**Pickling caveat:** lmfit's `Model` requires `func.__name__` to be set.
`functools.partial` objects lack `__name__`, so it must be assigned manually
after creating the partial (`fn.__name__ = base_fn.__name__`).

**Log-sum-exp stability in `_two_comp_model_fn`:**
Two-component model uses `ref = max(a, b); model = ref + log10(1 + 10^(other−ref))`
to avoid overflow/underflow when the two temperature components differ by many
dex.

### 3. Benchmark script

`scripts/benchmark_excitationfit.py` — times `H2ExcitationFit.run()` on the
CenA 7-line H₂ dataset.  Key flags:

```
--partition-method {ssr,pelt,both}   # compare algorithms
--workers N                          # parallel workers
--runs N                             # repeated timing
--save-reference FILE.npz            # save fitted maps for correctness checks
--compare-reference FILE.npz         # diff current run against saved reference
```

---

## Caveats and Known Limitations

### Parallel workers: overhead vs. speedup threshold

On small maps (≤ ~800 valid pixels, 12 CPUs) the parallel path is **not
faster** than serial.  Root cause: per-task IPC overhead (~1–2 ms) vs.
lightweight lmfit compute (~10–13 ms per pixel) with one task submitted per
pixel.  Measured: serial ~10.8 s, parallel ~16 s on CenA test data.

Rule of thumb: **parallel helps only above ~5000 valid pixels** (where compute
time per pixel amortises IPC cost).  The `workers` parameter is still correct to
expose — it will matter for large JWST/ALMA maps.

### Chunking not implemented

`LineRatioFit` also submits one pixel per task.  Batching N pixels per
`Executor` task would reduce IPC overhead and is the right next step for both
tools.  Deferred — tracked here as a future option.

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
| **Chunked parallel submission** | Submit N pixels per task to amortise IPC overhead; threshold ~50–100 pixels/chunk |
| **Tikhonov / spatial regularisation** | Requires moving from per-pixel `minimize()` to a single joint minimisation with sparse Jacobian (Option 4 in `regularization.md`).  Per-pixel independence is a hard architectural barrier to regularisation. |
| **`components=None` auto-selection** | Automatically choose 1 vs 2 components via AIC/BIC or F-test on the piecewise fit quality |
| **More than 2 components** | Piecewise framework supports n breakpoints; lmfit model extension straightforward |
| **CHplusExcitationFit** | Marked experimental; needs validation against real data |

---

## Files Changed

| File | Change |
|---|---|
| `pdrtpy/tool/excitation.py` | Core implementation (new methods, `workers`, `partition_method`, log-sum-exp model, `_one_line` restore) |
| `pyproject.toml` | Added `ruptures` to dependencies |
| `uv.lock` | Updated (ruptures 1.1.10) |
| `scripts/benchmark_excitationfit.py` | New benchmark script |
| `scripts/bench_excitation_ref_master.npz` | Reference solution from master branch |
