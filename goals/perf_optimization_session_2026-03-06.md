# PDRTpy Performance Optimization Session
**Date:** 2026-03-06
**Branch:** `perf-lineratiofit`
**Tag:** `Opts_2_and_3` (after first commit)

---

## Goal

Improve the performance of `LineRatioFit.run()` for multi-pixel image maps, as described in `performance.md`.

---

## Phase 1: Profiling and Bottleneck Identification

### Code exploration findings

**Call chain from `run()` (lineratiofit.py:359):**
```
run()
  ├─ _check_compatibility()
  ├─ read_models()
  ├─ _reset_masks() / _mask_measurements()
  ├─ _compute_valid_ratios()        # creates observed intensity ratios
  ├─ _compute_residual()            # Python loop over model grid points
  ├─ Minimizer setup
  ├─ _compute_chisq()
  ├─ _coarse_density_radiation_field()   # numpy argmin over chi-sq grid
  └─ _refine_density_radiation_field()   # per-pixel Python loop + lmfit minimize()
```

**Three bottlenecks identified:**

1. **`_refine_density_radiation_field` pixel loop** (`lineratiofit.py:711–751`):
   Pure Python `for j in range(npix)` loop, each calling `lmfit.Minimizer.minimize()` serially.

2. **`_residual_single_pixel` repeated flatten** (`lineratiofit.py:544–545`):
   `data.flatten()` and `uncertainty.array.flatten()` called on every minimizer iteration for every pixel — N_pixels × N_iterations × N_ratios redundant array allocations.

3. **`_compute_residual` inner loop** (`lineratiofit.py:571–596`):
   Python loop over model grid points computing `(mdata - pix) / merror` one point at a time, instead of using numpy broadcasting.

---

## Phase 2: Benchmark Setup

### Test data

Four Horsehead Nebula FITS files from `pdrtpy/testdata/`, read with `Measurement.read()`:

| File | identifier | restfreq |
|------|-----------|---------|
| `Horsehead_FIR_measurement.fits` | `"FIR"` | `None` |
| `Horsehead_12CO_measurement.fits` | `"CO_10"` | `115.2712 GHz` |
| `Horsehead_13CO_measurement.fits` | `"13CO_10"` | `110.20137 GHz` |
| `Horsehead_CII_measurement.fits` | `"CII_158"` | `1900.537 GHz` |

**Note:** identifiers `CO_10` and `13CO_10` match wk2020 model identifiers (not `12CO`/`13CO`).

**ModelSet:** `wk2020`, `z=1`
**Map size:** 102×102 = 10,404 pixels

### Benchmark script

`scripts/benchmark_lineratiofit.py` — standalone script with:
- `--runs/-n N` : number of timed runs (default: 1)
- `--verbose/-v` : debug-level logging
- `--workers/-w N` : number of parallel worker processes (-1 = all CPUs)
- `--log/-l FILE` : write log to file in addition to stdout

---

## Phase 3: Optimizations Implemented

### Opt 2: Pre-flatten observed ratio arrays (LOW RISK)

**File:** `lineratiofit.py`
**Change:** After `_compute_valid_ratios()` in `run()`, pre-compute flattened data and error arrays once:

```python
self._observedratios_flat = {
    k: (v.data.flatten(), v.uncertainty.array.flatten())
    for k, v in self._observedratios.items()
}
```

Then in `_residual_single_pixel()`, replace:
```python
# Before
dvalue[i] = self._observedratios[k].data.flatten()[index]
evalue[i] = self._observedratios[k].uncertainty.array.flatten()[index]

# After
dvalue[i] = self._observedratios_flat[k][0][index]
evalue[i] = self._observedratios_flat[k][1][index]
```

**Result:** 338.7 s → 293.8 s (**1.15× speedup**, ~45 s saved)

---

### Opt 3: Vectorize `_compute_residual` inner loop (LOW RISK)

**File:** `lineratiofit.py`
**Change:** Replace Python loop over model grid points with numpy broadcasting:

```python
# Before
for pix in modelpix:
    _q = (mdata - pix) / merror
    _q = ma.masked_invalid(_q)
    residuals.append(_q)
_qq = np.squeeze(np.reshape(residuals, newshape))

# After
modelpix_exp = modelpix.reshape((-1,) + (1,) * mdata.ndim)
residuals_arr = ma.masked_invalid(
    (mdata[np.newaxis, ...] - modelpix_exp) / merror[np.newaxis, ...]
)
_qq = np.squeeze(np.reshape(residuals_arr, newshape))
```

Works for both single-pixel (`mdata.ndim=0`) and map (`mdata.ndim=2`) cases via uniform broadcasting.

**Result:** 293.8 s → 144.8 s (**2.34× cumulative speedup**)
Note: test suite runtime also dropped from 367 s to 143 s.

---

### Opt 1: Parallel pixel loop via `ProcessPoolExecutor` (MEDIUM RISK)

**File:** `lineratiofit.py`
**Change:** Two module-level functions added (must be at module level for picklability):

```python
_worker_model_interps = None  # per-process cache

def _init_worker(model_points, model_values):
    """Build RegularGridInterpolators once per worker process."""
    global _worker_model_interps
    _worker_model_interps = [
        RegularGridInterpolator(pts, vals, method="linear", bounds_error=True)
        for pts, vals in zip(model_points, model_values)
    ]

def _fit_pixel_worker(j, obs_data_j, obs_err_j, init_density, init_rf,
                      minn, maxn, minfuv, maxfuv, nan_policy, minimize_kwargs):
    """Fit a single pixel in a worker process. Returns (j, MinimizerResult)."""
    ...
```

`run()` gains a `workers` parameter:
- `workers=None` (default): serial, fully backward-compatible
- `workers=-1`: use all available CPUs (`os.cpu_count()`)
- `workers=N`: use N worker processes
- Emcee and single-pixel fits always remain serial

**Why module-level functions are required:** `ProcessPoolExecutor` serializes work via `pickle`. Bound methods, lambdas, and closures that capture non-picklable objects (e.g., lmfit `Minimizer`, astropy `CCDData`) cannot cross process boundaries. Module-level functions with plain numpy array arguments are always picklable.

**Why `initializer` is used:** Model interpolator data (grid points + values) is sent once to each worker process at startup via `initializer=_init_worker`, not once per pixel task. This avoids O(N_pixels) serialization of large model arrays.

**Result:** 144.8 s → 91.8 s on 12 CPUs (**3.69× cumulative speedup**)

---

## Performance Summary

| Step | Time | ms/pixel | Speedup vs baseline |
|------|------|----------|---------------------|
| Baseline (serial) | 338.7 s | 32.6 | — |
| + Opt 2 (pre-flatten) | 293.8 s | 28.2 | 1.15× |
| + Opt 3 (vectorize residual) | 144.8 s | 13.9 | 2.34× |
| + Opt 1 (12 CPUs, `-w -1`) | **91.8 s** | **8.8** | **3.69×** |

---

## Discussion: Why Opt 1 Speedup Is Sub-Linear

With 12 CPUs, theoretical maximum is 12×; actual is ~1.6× incremental. Reasons:

1. **Process spawn overhead** — 12 worker processes must be forked and initialized.
2. **IPC per task** — Each of ~5200 non-NaN pixels requires pickling arguments and `MinimizerResult` across process boundaries.
3. **Task granularity** — Each pixel takes ~9 ms of compute; IPC latency is ~0.1–1 ms per task.
4. **NaN pixels handled serially** — Masked pixels are skipped in the main process before futures are submitted.
5. **lmfit overhead** — `Parameters` construction, solver setup, and result extraction per pixel adds fixed overhead.

**Suggested improvement:** Batch pixels into chunks (50–100 per future) to amortize IPC cost per task. Would likely push incremental speedup significantly closer to 12×.

---

## Discussion: Can `minimizer.minimize()` Fit Multiple Pixels at Once?

No — `minimize()` fits a single set of parameters `(density, radiation_field)` per call. Each pixel has its own independent best-fit values, requiring one call per pixel.

---

## Opt 4 — Joint Pixel Fitting via scipy.optimize.least_squares (IMPLEMENTED)

### Why lmfit was bypassed

`lmfit.Minimizer.minimize(method='least_squares', jac_sparsity=sparsity)` crashes with
`ValueError: inconsistent shapes` because lmfit's covariance computation uses element-wise `*`
instead of matrix `@` when the Jacobian is sparse (lmfit bug in `minimizer.py`).
Direct `scipy.optimize.least_squares` is used instead.

### Concept

Fits all N valid pixels in a **single** `scipy.optimize.least_squares` call:

1. **Parameter vector**: flat numpy array `x` where `x[::2]` = density, `x[1::2]` = rf for each valid pixel.

2. **Vectorized residual** using batched `RegularGridInterpolator` queries:
   ```python
   def _joint_residual(x):
       pts = np.column_stack([x[::2], x[1::2]])          # (n_valid, 2)
       mvalues = np.array([interp(pts) for interp in interps])  # (n_ratios, n_valid)
       return ((obs_data - mvalues) / obs_err).flatten()  # (n_valid * n_ratios,)
   ```

3. **Block-diagonal `jac_sparsity`**: pixel `j`'s parameters only affect pixel `j`'s residuals.
   ```python
   from scipy.sparse import lil_matrix
   sparsity = lil_matrix((N * n_ratios, 2 * N), dtype=np.int8)
   for j in range(N):
       sparsity[j*n_ratios:(j+1)*n_ratios, 2*j:2*j+2] = 1
   ```

4. **Direct scipy call**:
   ```python
   result = scipy.optimize.least_squares(
       _joint_residual, x0, bounds=(lb, ub),
       jac_sparsity=sparsity.tocsr(), tr_solver='lsmr', method='trf',
   )
   ```

### Key benefit of jac_sparsity

With block-diagonal sparsity, scipy groups non-overlapping columns and perturbs them simultaneously for finite-difference Jacobian estimation. Since all `density_j` parameters are independent (no shared residual rows), they can ALL be perturbed in a single residual evaluation. Same for all `rf_j`. So Jacobian cost is **2 residual evaluations per iteration** regardless of N — instead of 2×N. Combined with the vectorized residual, this is O(iterations) numpy calls total.

### Convergence issue and hybrid fix

With ~5200 valid pixels, the global TRF `ftol` is satisfied in ~2 outer iterations. Well-conditioned pixels converge correctly. However, ~575 pixels (11%) that start at the model boundary (coarse argmin = model edge) contribute small gradients and are not moved by LSMR before global convergence is declared.

**Fix (`joint_fit='hybrid'`)**: after the joint fit, any pixel whose result is within `rtol=1e-4` of its coarse initial is re-fitted with a single-pixel pure-scipy `least_squares` call. The per-pixel Jacobian J_i (shape `(n_ratios, 2)`) is available directly from the joint result's block-diagonal structure. ~45 s total, accurate.

**Fast mode (`joint_fit='fast'`)**: skips re-fitting. ~29 s, ~7-9% median accuracy loss concentrated at boundary pixels.

### Per-pixel stderr

Extracted from the block-diagonal joint Jacobian:
```python
J_i = jac[i*n_ratios:(i+1)*n_ratios, 2*i:2*i+2]  # (n_ratios, 2)
cov_i = inv(J_i.T @ J_i)
stderr_density = sqrt(cov_i[0, 0])
stderr_rf      = sqrt(cov_i[1, 1])
```

### Support for future regularization

| Regularization | Opt 1 (parallel per-pixel) | Opt 4 (joint + jac_sparsity) |
|---|---|---|
| Tikhonov (L2 spatial smoothness) | **No** — pixels are independent | **Yes** — append penalty rows to residual vector; extend sparsity for adjacent pixel pairs |
| LASSO (L1 sparsity) | **No** | Approx. via IRLS outer loop |
| Total-variation | **No** | Approx. via IRLS outer loop |

Opt 4 is the right architectural foundation if spatial regularization is on the roadmap.

---

## Performance Summary (updated)

| Step | Time | ms/pixel | Speedup vs baseline |
|------|------|----------|---------------------|
| Baseline (serial) | 338.7 s | 32.6 | — |
| + Opt 2 (pre-flatten) | 293.8 s | 28.2 | 1.15× |
| + Opt 3 (vectorize residual) | 144.8 s | 13.9 | 2.34× |
| + Opt 1 (12 CPUs, `-w -1`) | 91.8 s | 8.8 | 3.69× |
| Opt 4 `joint_fit='fast'` | **29 s** | **2.8** | **11.6×** |
| Opt 4 `joint_fit='hybrid'` | **~45 s** | **~4.3** | **~7.5×** |

---

## Commits

| Hash | Description |
|------|-------------|
| `7d8fe22e` | Optimize LineRatioFit performance: pre-flatten arrays (Opt 2) and vectorize residual (Opt 3); add benchmark script |
| `9edde99e` | Add parallel pixel fitting to LineRatioFit via ProcessPoolExecutor (Opt 1); add `--workers` flag to benchmark |
| `4fe7f06a` | Add joint pixel fitting (Opt 4) via scipy.optimize.least_squares; `joint_fit='hybrid'` and `'fast'` modes |

**Tags:** `Opts_2_and_3` → `7d8fe22e`, `Pre_Opt4` → `9edde99e`.

---

## Files Changed

| File | Change |
|------|--------|
| `pdrtpy/tool/lineratiofit.py` | Opts 1, 2, 3, 4 |
| `scripts/benchmark_lineratiofit.py` | New benchmark script; updated for all modes |
