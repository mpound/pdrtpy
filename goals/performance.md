
The goal is to improve the performance of fitting with LineRatioFit.run
and the methods it calls.  In particular, we need to speed up fitting
for multi-pixel images.

Profile the slow path.
Identify bottlenecks, then propose 3 optimizations ranked by impact and risk.

For each:
- **Files and functions**: exact files/functions to change
- **Tradeoffs**: expected tradeoffs
- **Benchmark plan**: a benchmark plan (before/after)
