#!/usr/bin/env python
"""Benchmark LineRatioFit.run() performance using Horsehead Nebula test data.

Loads four Horsehead Nebula map Measurements and runs LineRatioFit against
the wk2020 model set, timing the full run() call. Use this script to measure
before/after performance when making changes to the fitting code.

Usage::

    python scripts/benchmark_lineratiofit.py
    python scripts/benchmark_lineratiofit.py --runs 3 --verbose
    python scripts/benchmark_lineratiofit.py --log bench.log

"""
import argparse
import logging
import sys
import time
from pathlib import Path

import astropy.units as u

import pdrtpy.pdrutils as utils
from pdrtpy.measurement import Measurement
from pdrtpy.modelset import ModelSet
from pdrtpy.tool.lineratiofit import LineRatioFit


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
    """Load the four Horsehead Nebula Measurements from testdata."""
    specs = [
        ("Horsehead_FIR_measurement.fits", "FIR", None),
        ("Horsehead_12CO_measurement.fits", "CO_10", 115.2712 * u.GHz),
        ("Horsehead_13CO_measurement.fits", "13CO_10", 110.20137 * u.GHz),
        ("Horsehead_CII_measurement.fits", "CII_158", 1900.537 * u.GHz),
    ]

    measurements = []
    for filename, identifier, restfreq in specs:
        path = utils.get_testdata(filename)
        kwargs = {"identifier": identifier}
        if restfreq is not None:
            kwargs["restfreq"] = restfreq
        log.debug("Loading %s (identifier=%s, restfreq=%s)", filename, identifier, restfreq)
        m = Measurement.read(path, **kwargs)
        log.debug("  shape=%s unit=%s", m.data.shape, m.unit)
        measurements.append(m)

    return measurements


def run_benchmark(args: argparse.Namespace, log: logging.Logger) -> None:
    log.info("=== LineRatioFit benchmark ===")
    log.info("ModelSet : wk2020 z=1")
    log.info("Runs     : %d", args.runs)

    log.info("Loading measurements...")
    measurements = load_measurements(log)
    npix = measurements[0].data.size
    shape = measurements[0].data.shape
    log.info("Map shape: %s  (%d pixels)", shape, npix)

    log.info("Loading ModelSet wk2020 z=1...")
    ms = ModelSet("wk2020", z=1)
    log.debug("ModelSet loaded: %s", ms)

    elapsed_times = []
    for i in range(args.runs):
        log.info("--- Run %d/%d ---", i + 1, args.runs)
        p = LineRatioFit(ms, measurements=measurements)
        t0 = time.perf_counter()
        p.run()
        elapsed = time.perf_counter() - t0
        elapsed_times.append(elapsed)
        log.info("  Elapsed: %.3f s  (%.1f ms/pixel)", elapsed, 1000 * elapsed / npix)

    if args.runs > 1:
        avg = sum(elapsed_times) / len(elapsed_times)
        mn = min(elapsed_times)
        mx = max(elapsed_times)
        log.info("--- Summary over %d runs ---", args.runs)
        log.info("  Min : %.3f s", mn)
        log.info("  Max : %.3f s", mx)
        log.info("  Mean: %.3f s  (%.1f ms/pixel)", avg, 1000 * avg / npix)

    log.info("=== Done ===")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog=Path(sys.argv[0]).name,
        description="Benchmark LineRatioFit.run() on Horsehead Nebula test data.",
    )
    parser.add_argument(
        "--runs",
        "-n",
        type=int,
        default=1,
        metavar="N",
        help="number of timed runs to average (default: 1)",
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
