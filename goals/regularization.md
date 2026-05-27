Yes — and this is one of the strongest arguments for Opt 4 over Opt 1.

Why Opt 4 supports regularization naturally:

Regularization couples neighboring pixels' parameters. Tikhonov adds a spatial-smoothness penalty:

minimize:  ||residuals(density, rf)||² + λ||L·[density, rf]||²

where L is a finite-difference operator over the map (penalizing large jumps between adjacent pixels). In the single-minimize
 to the residual vector:

def residual_all_pixels(params):
    density = ...  # (N,) array
    rf = ...
    data_resid = ...  # (N * n_ratios,) — data fit terms

    # Tikhonov: penalize differences between horizontally/vertically adjacent pixels
    d_reshaped = density.reshape(ny, nx)
    penalty_dx = lambda_reg * (d_reshaped[:, 1:] - d_reshaped[:, :-1]).flatten()
    penalty_dy = lambda_reg * (d_reshaped[1:, :] - d_reshaped[:-1, :]).flatten()
    # same for rf ...

    return np.concatenate([data_resid, penalty_dx, penalty_dy, ...])

The jac_sparsity matrix grows by the penalty rows, but those rows each touch exactly 2 columns (adjacent pixel parameters) — s
transparently.

LASSO (L1) doesn't fit directly into least_squares (which minimizes L2), but can be approximated via iteratively reweighted le
call in an outer loop that reweights the penalty terms based on the previous solution. The Opt 4 structure supports this witho

Why Opt 1 cannot support regularization:

Per-pixel independence is a hard architectural constraint — each minimize() call has no knowledge of neighboring pixels. There
coupling. You could post-process the density/radiation field maps with a smoothing filter, but that's not true regularization
fidelity and smoothness are not jointly optimized.

Summary:

┌───────────────────────┬─────────────────────────────┬────────────────────────────────────────┐
│                       │ Opt 1 (parallel per-pixel)  │ Opt 4 (single minimize + jac_sparsity) │
├───────────────────────┼─────────────────────────────┼────────────────────────────────────────┤
│ Tikhonov              │ No                          │ Yes — add penalty rows to residual     │
├───────────────────────┼─────────────────────────────┼────────────────────────────────────────┤
│ LASSO                 │ No                          │ Approx. via IRLS outer loop            │
├───────────────────────┼─────────────────────────────┼────────────────────────────────────────┤
│ Total-variation       │ No                          │ Approx. via IRLS outer loop            │
├───────────────────────┼─────────────────────────────┼────────────────────────────────────────┤
│ Future regularization │ Impossible without redesign │ Extensible by design                   │
└───────────────────────┴─────────────────────────────┴────────────────────────────────────────┘

Opt 4 is the right foundation if spatial regularization is on the roadmap. Want to proceed with implementing it?

Crunched for 36s

Can you save the entire conversation today to a markdown file?

Write(perf_optimization_session_2026-03-06.md)
