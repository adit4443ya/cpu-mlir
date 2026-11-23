#!/usr/bin/env python3
"""summarize.py OUTDIR: turn OUTDIR/runs.tsv (from device_bench.sh) into OUTDIR/RESULTS.md.

Per suite / target / size: median and min ms over the reps for each variant, the speedup of the
variant against the baseline (matmul: base = main behaviour; softmax: main = PR #14 off), whether
every rep matched the reference, and the number of smstart in the kernel asm.
speedup > 1 means the variant is faster than the baseline.
"""
import collections, csv, os, statistics, sys

out = sys.argv[1]
rows = list(csv.DictReader(open(os.path.join(out, "runs.tsv")), delimiter="\t"))
BASE = {"matmul": "base", "softmax": "main"}
g = collections.defaultdict(list)
for r in rows:
    g[(r["suite"], r["mode"], r["target"], r["size"], r["variant"])].append(r)

lines = [f"# Results ({os.path.abspath(out)})", "",
         "speedup = baseline median / variant median (> 1 = faster). ms = whole grid x ITERS iterations, "
         "measured inside the exe (benchmark.log).", ""]
keys = sorted({k[:4] for k in g})
for suite, mode, tgt, size in keys:
    lines += [f"## {suite} {size} on {tgt} ({mode})", "",
              "| variant | median ms | min ms | reps ok/total | speedup vs " + BASE[suite] + " | all match | smstart in asm |",
              "|---|---:|---:|---:|---:|---|---:|"]
    def med(v):
        ms = [float(r["ms"]) for r in g.get((suite, mode, tgt, size, v), []) if r["ms"] != "NA"]
        return (statistics.median(ms), min(ms), len(ms)) if ms else (None, None, 0)
    b = med(BASE[suite])[0]
    for v in [k[4] for k in g if k[:4] == (suite, mode, tgt, size)]:
        m, mn, n = med(v); rs = g[(suite, mode, tgt, size, v)]
        sp = f"{b / m:.2f}x" if (b and m) else "NA"
        okm = "yes" if all(r["match"] == "True" for r in rs) else "NO: " + ",".join(r["match"] for r in rs)
        sm = sorted({r["smstart_in_asm"] for r in rs})
        lines.append(f"| {v} | {m if m is None else f'{m:.3f}'} | {mn if mn is None else f'{mn:.3f}'} | "
                     f"{n}/{len(rs)} | {sp} | {okm} | {','.join(sm)} |")
    lines.append("")
open(os.path.join(out, "RESULTS.md"), "w").write("\n".join(lines))
print("\n".join(lines))
