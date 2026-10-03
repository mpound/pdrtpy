# Synthetic Test Maps for Regularization: Findings (2026-09-28)

## Goal

Build a synthetic map (known ground truth, forward-modeled from a real `ModelSet`) that
demonstrates `LineRatioFit.regularize()` correcting isolated pixel misfits **without**
needing `density_range`/`radiation_field_range` restriction — something the real N22 data
can't do, because N22's oscillation comes from a genuine, large-scale, two-branch
degeneracy that regularization structurally cannot (and should not) resolve on its own
(see `goals/regularization_design_options.md` and the N22 notebook work).

## Attempt 1: CII_158 / OI_63 / FIR, SMC (z=0.2) — abandoned, model too degenerate everywhere

This is the exact combination N22 uses. Built a 20×20 map: uniform Branch A background
(n=5000 cm⁻³, G0≈75 Habing) plus scattered/blocked Branch B (n=8×10⁴, G0≈5 Habing)
pixels, forward-modeled via the `CII_158/FIR` ratio model (FIR has no standalone
intensity model in this ModelSet, so it's derived as `CII_158.get(n,G0) /
ratio["CII_158/FIR"].get(n,G0)`; `OI_63` comes directly from
`ModelSet.get_models(ids, model_type="intensity")`).

**Finding:** at 5% noise, 24% of *background* pixels (not the deliberately-injected
ones) spontaneously landed >0.3 dex from truth — confirming this exact combination is
degenerate over broad regions of phase space, not just between the two N22 branches.
Dropping to 1% noise gave a clean background, but then the deliberately-injected
Branch-B pixels became split: only 5/15 recovered near Branch B, the rest split into a
*third*, previously-uncharacterized attractor. This combination is too pathological to
build a clean, controllable synthetic map from — abandoned in favor of a better-
conditioned combination.

## Attempt 2: OI_63 / CI_609 / CO_43 / CII_158, wk2020 (z=1) — well-conditioned, but exposed a deeper issue

Per Marc's suggestion, switched to the 4-line combination already used in
`pdrtpy/tool/test/test_lineratiofit.py`'s `single_pixel_measurements` fixture, with a
stable point at n=10⁴ cm⁻³, G0≈20 Habing (native G0=0.032). All 4 lines have direct
intensity models (`ModelSet.get_models(ids, model_type="intensity")` returns all four —
no ratio-inversion needed, unlike FIR above). At 10% noise (Marc's suggestion, matching
real-observation S/N), background self-recovers cleanly: 0% of pixels off by >0.3 dex,
mean error 0.06 dex. **10% noise was never the problem** — the earlier instability was
specific to the CII/OI/FIR/SMC combination.

### Sub-attempt 2a: inject a genuinely different truth (Branch B) at scattered pixels — wrong test

Directly checked the χ² cost of pulling one such pixel back to the background value:

| Δlog₁₀(n) from own optimum | χ² |
|---|---|
| 0 | 0.47 |
| −0.1 | 2.56 |
| −0.5 | 53 |
| −1.0 | 546 |
| −1.977 (all the way to background) | **49,516** |

A well-conditioned combination places a genuinely-different pixel with enormous
confidence — moving it back would mean overriding real data with a smoothness prior,
which regularization *should not* do. This revealed a structural tension: "well-
conditioned data" and "regularization has visible, legitimate work to do" pull in
opposite directions. A pixel worth correcting must be *statistically* ambiguous (shallow
χ² near its own optimum), not *truly* different — otherwise correcting it is scientifically
wrong, not a demonstration of correctness.

### Sub-attempt 2b: inject a realistic noise excursion instead — the right test design

Kept the true (n, G0) at Branch A everywhere; instead gave scattered pixels a larger,
single-line noise draw (e.g. `CO_43` flux offset by `n_sigma × 10%`, calibrated so the
resulting χ² barrier to snap back to the background is a modest ~12-19, not ~50,000).
This is the correct model of "salt and pepper" oscillation: a pixel whose *data*, not
its *true parameters*, is an outlier.

**This did not work either** — see below. Even with a small, genuinely correctable
barrier, `regularize()` produced almost no correction at modest λ, and the underlying
solver behavior turned out to be unstable (see "Instability found," next section) —
non-monotonic and then divergent as λ increased, which is a separate, deeper problem
than anything about the test map's construction.

## Reusable pieces from this work (keep for the next attempt)

- **Forward-modeling recipe:** `ModelSet.get_models(identifiers, model_type="intensity")`
  returns a dict of `Measurement` model images; `.get(n, G0)` (backed by
  `RegularGridInterpolator`, already set up — no extra step needed) interpolates the
  model at an arbitrary point. Works directly for any line with its own intensity model;
  `FIR` (SMC) needed the ratio-model workaround above, but `OI_63`/`CI_609`/`CO_43`/
  `CII_158` (wk2020) all have direct intensity models.
- **Self-recovery check:** before trusting a candidate (n, G0) point as a map's "truth,"
  forward-model it, add realistic noise, run `LineRatioFit.run()`, and confirm it
  recovers close to itself. Several candidates for the SMC combination failed this even
  a few percent away from a point that *did* pass — the phase space has narrow,
  unpredictable instability pockets, not just the two well-known branches.
  `(1e4, 0.032)` and `(1e6, 1.0)` pass cleanly for wk2020 at 5-10% noise; `(1e3, 0.01)`
  does not.
- **χ² barrier diagnostic:** `LineRatioFit._build_regularization_context()` +
  `._regularization_chisq_per_pixel()` can be called directly (outside `regularize()`)
  to compute the χ² cost of moving one pixel to a candidate alternate value — this is
  exactly how the "genuinely different" vs. "noise-driven" distinction above was made,
  and is the right tool for calibrating any future synthetic outlier's magnitude before
  running the full map.

## Status

**Resolved (2026-10-01).** Was blocked on the solver instability found while validating
sub-attempt 2b (see below); PR #252 reverted the default to `step_mode="global"` and
merged to master, which stabilized things enough to complete this task — see "Resolved"
section near the end for the final map design and test suite
(`pdrtpy/tool/test/test_lineratiofit_regularize_synthetic.py`).
`step_mode="per_pixel"` itself is still unfixed (see "Concrete next steps" at the end)
but is no longer blocking this task, since `"global"` is sufficient for a clean
demonstration.

## Instability trace (2026-09-28): root cause identified

On the single-outlier setup from 2b (10×10 map, one pixel with a `CO_43` flux offset,
`n_sigma=2.5`, χ² barrier to revert ≈13-19), a λ scan with enough outer iterations to
actually converge (`max_iter=1000`) gave:

| λ | outlier logn after | correction |
|---|---|---|
| 5 | 4.095 | +19% (right direction) |
| 10 | 4.126 | −7% (wrong direction) |
| 20 | 4.067 | +43% |
| 40 | 4.564 | **−381%** (huge overshoot) |
| 80 | 5.124 | **−859%** (diverging) |

Non-monotonic-then-diverging behavior as λ increases is inconsistent with TV being a
convex penalty — ruled out three candidate causes in order:

1. **Chambolle inner-loop under-convergence.** Increasing the inner TV `n_iter` from 50
   to 1000 fixed λ=20 (43%→100% correct) but left λ=40/80 completely unchanged
   (identical output to 3 significant figures regardless of inner precision) — so
   under-convergence explains the λ=20 gap but not the λ=40/80 divergence.
2. **FISTA momentum overshoot.** Re-ran with momentum disabled (plain ISTA, `y=x`
   every iteration) — λ=40/80 still diverge by a similar magnitude. Not the momentum
   term.
3. **A bug in the TV proximal operator itself.** Verified `TotalVariationRegularizer.prox()`
   in isolation (a 10×10 array, background=4.0, one spike=4.117, no χ² involved) obeys
   the maximum principle (output always within `[min(input), max(input)]`) at every λ
   from 0.5 to 200. The TV math is correct on its own.

**Actual mechanism, confirmed by direct inspection:** at the start of an outer FISTA
iteration, before any backtracking has run, `step` is reset to 1.0 for *every* pixel
(the per-pixel growth step from the earlier step-growth fix). At `step=1.0`, `theta =
step*lam = 80` uniformly. The trial point `z = y - step*grad` computed at this
unthrottled step contains at least one pixel whose gradient is large enough to push its
`z` value to ~7.09 (versus a ~4-6 range everywhere else) — i.e. a single badly-behaved
pixel produces a wild trial value before backtracking has had any chance to react.
Because the TV proximal operator is **not separable across pixels** — Chambolle's
dual-variable solve couples every pixel to its neighbors — that one bad trial value
distorts the *joint* proximal solution for pixels near it, not just its own update. The
descent-lemma check (`f_new <= fy + lin + quad`) is then evaluated **per pixel
independently**, so a well-behaved neighbor pixel can fail its own check purely because
the *joint* TV solve dragged it somewhere bad on account of the contaminating pixel,
not because of anything wrong with its own gradient. `step_mode="per_pixel"` (the
step-throttling fix from the previous session) successfully stops one bad pixel from
throttling the *entire map's* step to a single shared scalar — but it does not, and
structurally cannot on its own, stop that same bad pixel from contaminating its
*immediate neighbors'* proximal update through the coupled TV solve, because the
proximal operator itself has no concept of "this pixel's step, that pixel's step" — it
only sees one `theta` field per call.

**This is a real, dataset-independent gap, not a property of any specific synthetic
map.** It plausibly explains part of what was seen on N22 too — large λ there also
required per-pixel step throttling that never got a chance to stabilize before the next
θ-driven joint TV solve pulled things around again. The earlier N22 diagnosis (TV
consolidating toward a cheaper global attractor once two branches are both present) is
still real and doesn't depend on this bug, but this instability may have been
compounding it rather than the branch-preference story being the whole account.

## `step_mode="global"` vs `"per_pixel"`, same map, same λ scan (2026-09-29)

Prompted by a direct question from Marc: is per-pixel step sizing (added last session
specifically to stop one stiff pixel from throttling the *whole map's* shared step) the
wrong fix, layered on top of a problem not yet understood? Re-ran the identical
single-outlier setup and λ scan (`max_iter=1000`) with `step_mode="global"` as a
control, never done before today:

| λ | `global` | `per_pixel` |
|---|---|---|
| 5 | +12.7% | +19.2% |
| 10 | +23.2% | **−7.1%** |
| 20 | +35.9% | +42.7% |
| 40 | +41.0% | **−380.8%** |
| 80 | +42.6% | **−858.6%** |

**`global` is completely stable across the whole range** — monotonic, gracefully
saturating around ~43% correction, never overshooting. `per_pixel` matches or modestly
beats it at λ≤20 (consistent with the synthetic "one stiff pixel among many
well-behaved ones" toy test from last session, which only probed this low-λ regime),
then diverges catastrophically exactly where N22 needed to operate (λ=50-250).

**Conclusion: per-pixel step sizing traded a real but bounded problem (shared-step
throttling, causing `global` mode's weak ~43% plateau — note this plateau means
`global` mode still has *some* version of the original throttling problem, it is just
safe about it) for a worse, unbounded one (divergence).** The joint-TV-solve
contamination mechanism above is the reason: `per_pixel` mode's per-pixel step values
diverge from each other exactly when a bad pixel's coupling through the shared
Chambolle solve corrupts a neighbor's otherwise-valid update, and nothing in the current
design catches that. `global` mode never lets per-pixel step values diverge from each
other in the first place, so it can't hit this failure mode — at the cost of the
original (bounded, non-catastrophic) weakness that motivated `per_pixel` mode last
session.

**Recommendation:** revert the default to `step_mode="global"` until the trust-
region/monotonicity fix below is implemented and re-verified — `"per_pixel"` should stay
available but documented as needing that fix before it's safe to use at the λ values
real degenerate data (N22) requires. Layering a third fix on top of `per_pixel` without
first re-establishing a stable, understood baseline would repeat the same mistake.

## Resolved (2026-10-01): synthetic map demonstrated successfully, on `step_mode="global"`

With `step_mode` reverted to `"global"` (PR #252, merged to master), resumed on branch
`synthetic-regularization-maps`. Built the final version of the Attempt 2 map (15×15,
`wk2020`, `_BRANCH_A=(1e4, 0.032)` background + `_BRANCH_B=(1e6, 1.0)` 5×5 block + 10
scattered single-line (`CO_43`, `n_sigma=2.5`) noise outliers) and found one more
construction bug before it worked cleanly: **the scattered-pixel placement didn't
exclude the block's footprint**, so one "scattered" coordinate landed inside the block,
got overwritten with Branch-B truth, then *also* got the extra outlier perturbation on
top of that — landing far from either population and dragging the whole map's shared
`step` down with it (global mode's step is shared, so one contaminated pixel throttles
everyone). Excluding the block (plus a 1-pixel margin) from scattered-coordinate
placement fixed it immediately.

With that fixed, a λ scan on the corrected map (λ=20/30/50, `max_iter=300`) gave a clean,
monotonic result up to λ≈30: mean scattered-pixel error 0.099→0.065 dex (34-36%
reduction, *every* scattered pixel individually improved, not just the mean), block
change only 0.04-0.06 dex (negligible against its ~2 dex separation from the
background). At λ=50 the correction stopped improving while the block started eroding
(0.196 dex) — confirming λ≈20-30 is the right operating point for this map, consistent
with `global` mode's known plateau behavior (it saturates rather than diverging).

**Final test suite:** `pdrtpy/tool/test/test_lineratiofit_regularize_synthetic.py` (6
tests, ~17-19s total, deterministic with a fixed RNG seed) — sanity checks that the
synthetic map was built as intended (background self-recovers, scattered pixels are
modest not catastrophic outliers, block recovers its own truth), then the actual claims:
scattered outliers move substantially and uniformly closer to truth after
`regularize(method="tv", lam=25, max_iter=300)`, the block is preserved (not eroded),
and no `density_range`/`radiation_field_range` is used anywhere in the module. This is
the ground-truth-based, no-restriction-needed demonstration the original task asked for.

## Concrete next steps for `step_mode="per_pixel"` (still not attempted)

- Clamp the per-outer-iteration proximal displacement (a trust region on `‖x_new - y‖`
  directly, not just on the per-pixel `step` used to build the trial point `z`), so a
  single bad pixel's contribution to the joint TV solve can't produce an
  out-of-proportion `z` value in the first place.
- Alternatively/additionally, check the composite objective (`chisq + lam*TV`, not just
  the per-pixel smooth-term check) between outer iterations and reject/backtrack the
  *whole* step if it increases (a "monotone FISTA" style safeguard) — this would catch
  the joint-solve contamination that the current per-pixel-only check misses.
- Either fix should be verified against both a controlled synthetic case (this 10×10
  single-outlier setup, which now has a known, reproducible failure signature) and
  against N22 before trusting it, **and** re-verified against the `global`-mode baseline
  above so a fixed `per_pixel` mode is shown to be strictly better than `global`, not
  just different.
