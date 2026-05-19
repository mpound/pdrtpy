#!/usr/bin/env python
"""Benchmark H2ExcitationFit.run() performance using Centaurus A H2 test data.

Loads eight H2 rovibrational line Measurements (H200S1–H200S8) from CenA and
runs H2ExcitationFit.run() timing the full call.  Optionally saves fitted
parameter maps as a reference solution so later runs can compare correctness
after code changes.

Usage::

    # Establish baseline on master (serial, 2-component):
    python scripts/benchmark_excitationfit.py --save-reference bench_ref.npz

    # Time and compare after a code change:
    python scripts/benchmark_excitationfit.py --compare-reference bench_ref.npz

    # Parallel run, 3 timing repeats, verbose:
    python scripts/benchmark_excitationfit.py --workers -1 --runs 3 --verbose

    # One-component fit:
    python scripts/benchmark_excitationfit.py --components 1

"""

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pdrtpy.pdrutils as utils
from pdrtpy.measurement import Measurement
from pdrtpy.tool.excitation import H2ExcitationFit


def setup_logging(verbose: bool, logfile: str | None) -> logging.Logger:
    log = logging.getLogger("benchmark")
    log.setLevel(logging.DEBUG if verbose else logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    handler.setFormatter(fmt)
    log.addHandler(handler)

    if logfile:
        fh = logging.FileHandler(logfile)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        log.addHandler(fh)

    return log


def load_measurements(log: logging.Logger) -> list:
    """Load H200S1–H200S8 CenA Measurements from testdata."""
    # do not use H200S8 because it has too many bad pixels and causes many fit failures
    # lines = ["H200S1", "H200S2", "H200S3", "H200S4", "H200S5", "H200S6", "H200S7", "H200S8"]
    lines = ["H200S1", "H200S2", "H200S3", "H200S4", "H200S5", "H200S6", "H200S7"]
    measurements = []
    for line in lines:
        path = utils.get_testdata(f"{line}_CenA.fits")
        log.debug("Loading %s", path)
        m = Measurement.read(path, identifier=line)
        log.debug("  shape=%s unit=%s", m.data.shape, m.unit)
        measurements.append(m)
    return measurements


def extract_results(fit: H2ExcitationFit, components: int) -> dict:
    """Pull fitted parameter arrays from a completed fit."""
    results = {
        "tcold": np.ma.filled(fit.tcold.data, np.nan),
        "ncold": np.ma.filled(fit.cold_colden.data, np.nan),
        "mask": fit.fit_result.mask.astype(bool),
    }
    if components == 2:
        results["thot"] = np.ma.filled(fit.thot.data, np.nan)
        results["nhot"] = np.ma.filled(fit.hot_colden.data, np.nan)
    return results


def save_reference(results: dict, path: str, log: logging.Logger) -> None:
    np.savez(path, **results)
    log.info("Reference saved to %s", path)


def compare_results(current: dict, ref_path: str, log: logging.Logger) -> None:
    """Load saved reference and report per-parameter deviations."""
    ref = np.load(ref_path)
    log.info("--- Correctness comparison vs %s ---", ref_path)

    # Combined valid mask: pixels that are unmasked in BOTH runs
    cur_mask = current["mask"]
    ref_mask = ref["mask"]
    valid = ~cur_mask & ~ref_mask

    n_cur_masked = int(cur_mask.sum())
    n_ref_masked = int(ref_mask.sum())
    log.info("  Masked pixels  current=%-6d  reference=%d", n_cur_masked, n_ref_masked)

    params = ["tcold", "ncold"]
    if "thot" in current:
        params += ["thot", "nhot"]

    for key in params:
        if key not in ref:
            log.warning("  %s not in reference — skipping", key)
            continue
        cur_vals = current[key][valid]
        ref_vals = ref[key][valid]
        # Ignore pixels where either value is non-finite
        finite = np.isfinite(cur_vals) & np.isfinite(ref_vals)
        if not finite.any():
            log.warning("  %s: no finite common pixels", key)
            continue
        absdiff = np.abs(cur_vals[finite] - ref_vals[finite])
        reldiff = absdiff / np.abs(ref_vals[finite] + 1e-30)
        log.info(
            "  %-8s  MAD=%.3g  median_reldiff=%.2f%%  max_absdiff=%.3g  n_valid=%d",
            key,
            float(np.median(absdiff)),
            float(100 * np.median(reldiff)),
            float(np.max(absdiff)),
            int(finite.sum()),
        )


def _timed_run(method: str, measurements: list, args: argparse.Namespace, log: logging.Logger) -> tuple:
    """Run H2ExcitationFit with a given partition_method; return (elapsed_times, last_fit)."""
    npix = measurements[0].data.size
    run_kwargs = {"components": args.components, "partition_method": method}
    if args.workers is not None:
        run_kwargs["workers"] = args.workers

    elapsed_times = []
    last_fit = None
    for i in range(args.runs):
        log.info("  --- Run %d/%d ---", i + 1, args.runs)
        fit = H2ExcitationFit(measurements=measurements)
        t0 = time.perf_counter()
        fit.run(**run_kwargs)
        elapsed = time.perf_counter() - t0
        elapsed_times.append(elapsed)
        last_fit = fit
        log.info("    Elapsed : %.3f s  (%.1f ms/pixel)", elapsed, 1000 * elapsed / npix)

    if args.runs > 1:
        avg = sum(elapsed_times) / len(elapsed_times)
        log.info("  --- Timing summary over %d runs ---", args.runs)
        log.info("    Min : %.3f s", min(elapsed_times))
        log.info("    Max : %.3f s", max(elapsed_times))
        log.info("    Mean: %.3f s  (%.1f ms/pixel)", avg, 1000 * avg / npix)

    return elapsed_times, last_fit


def run_benchmark(args: argparse.Namespace, log: logging.Logger) -> None:
    log.info("=== H2ExcitationFit benchmark ===")
    log.info("Components: %d", args.components)
    log.info("Runs      : %d", args.runs)
    workers_label = (
        "serial" if args.workers is None else ("all CPUs" if args.workers == -1 else f"{args.workers} workers")
    )
    log.info("Workers   : %s", workers_label)

    log.info("Loading measurements...")
    measurements = load_measurements(log)
    npix = measurements[0].data.size
    shape = measurements[0].data.shape
    log.info("Map shape : %s  (%d pixels)", shape, npix)

    # Determine which partition methods to run
    methods = ["ssr", "pelt"] if args.partition_method == "both" else [args.partition_method]

    all_results = {}
    all_times = {}
    for method in methods:
        log.info("--- Partition method: %s ---", method)
        elapsed_times, last_fit = _timed_run(method, measurements, args, log)
        all_times[method] = elapsed_times
        results = extract_results(last_fit, args.components)
        all_results[method] = results
        n_masked = int(results["mask"].sum())
        log.info("  Masked pixels: %d / %d  (%.1f%%)", n_masked, npix, 100 * n_masked / npix)

    # If both methods ran, print a side-by-side timing comparison
    if len(methods) == 2:
        log.info("=== Partition method comparison ===")
        mean_ssr = sum(all_times["ssr"]) / len(all_times["ssr"])
        mean_pelt = sum(all_times["pelt"]) / len(all_times["pelt"])
        log.info("  SSR  mean: %.3f s  (%.1f ms/pixel)", mean_ssr, 1000 * mean_ssr / npix)
        log.info("  PELT mean: %.3f s  (%.1f ms/pixel)", mean_pelt, 1000 * mean_pelt / npix)
        if mean_ssr > 0:
            log.info("  PELT / SSR speedup ratio: %.2fx  (>1 means SSR is faster)", mean_pelt / mean_ssr)
        # Correctness: compare the two methods against each other
        log.info("--- SSR vs PELT correctness (median rel-diff) ---")
        ssr_r, pelt_r = all_results["ssr"], all_results["pelt"]
        valid = ~ssr_r["mask"] & ~pelt_r["mask"]
        for key in ["tcold", "ncold", "thot", "nhot"]:
            if key not in ssr_r or key not in pelt_r:
                continue
            sv = ssr_r[key][valid]
            pv = pelt_r[key][valid]
            finite = np.isfinite(sv) & np.isfinite(pv)
            if not finite.any():
                continue
            reldiff = np.abs(sv[finite] - pv[finite]) / (np.abs(pv[finite]) + 1e-30)
            log.info(
                "  %-8s  median_reldiff=%.2f%%  max_absdiff=%.3g",
                key,
                100 * float(np.median(reldiff)),
                float(np.max(np.abs(sv[finite] - pv[finite]))),
            )

    # Use the first (or only) method's results for save/compare
    primary = all_results[methods[0]]
    if args.save_reference:
        save_reference(primary, args.save_reference, log)

    if args.compare_reference:
        compare_results(primary, args.compare_reference, log)

    log.info("=== Done ===")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog=Path(sys.argv[0]).name,
        description="Benchmark H2ExcitationFit.run() on Centaurus A H2 test data.",
    )
    parser.add_argument(
        "--runs",
        "-n",
        type=int,
        default=1,
        metavar="N",
        help="number of timed runs (default: 1)",
    )
    parser.add_argument(
        "--components",
        "-c",
        type=int,
        choices=[1, 2],
        default=2,
        metavar="N",
        help="number of temperature components to fit (default: 2)",
    )
    parser.add_argument(
        "--workers",
        "-w",
        type=int,
        default=None,
        metavar="N",
        help="number of worker processes (-1 = all CPUs, default: serial)",
    )
    parser.add_argument(
        "--partition-method",
        "-p",
        choices=["ssr", "pelt", "both"],
        default="ssr",
        help=(
            "breakpoint-finding algorithm used for initial parameter guesses: "
            "'ssr' (default) uses a vectorised prefix-sum residual search (fast, no external library); "
            "'pelt' uses the ruptures PELT algorithm (O(n log n), higher per-call overhead for small n); "
            "'both' runs each method in sequence and prints a timing/correctness comparison."
        ),
    )
    parser.add_argument(
        "--save-reference",
        "-s",
        metavar="FILE",
        default=None,
        help="save fitted parameter maps to FILE.npz as correctness reference",
    )
    parser.add_argument(
        "--compare-reference",
        "-r",
        metavar="FILE",
        default=None,
        help="compare fitted results against previously saved reference FILE.npz",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="enable debug-level output",
    )
    parser.add_argument(
        "--log",
        "-l",
        metavar="FILE",
        default=None,
        help="write log output to FILE in addition to stdout",
    )
    args = parser.parse_args()

    log = setup_logging(args.verbose, args.log)
    run_benchmark(args, log)


if __name__ == "__main__":
    main()
