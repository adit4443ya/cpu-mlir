#!/usr/bin/env python3
"""Cycles per iteration of the hottest reduction loop in a kernel's asm, with llvm-mca.

usage: mca_loop.py KERNEL-arm64.S [MCPU]       (MCPU default neoverse-v2)
Finds each loop (a backward branch) whose body contains fadda/faddv/fmaxnmv,
prints its instructions and llvm-mca's cycles per iteration. llvm-mca models
the non-streaming core; it knows nothing about streaming-mode timing.
"""
import re, subprocess, sys, os
asm, mcpu = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else "neoverse-v2")
mca = os.path.join(os.environ.get("LLVM_BINARY_DIR", ""), "llvm-mca")
lines = open(asm).read().splitlines()
labels = {m.group(1): i for i, l in enumerate(lines) if (m := re.match(r"^(\.LBB\w+):", l))}
red = re.compile(r"\b(fadda|faddv|fmaxnmv|fmaxv|fminnmv)\b")
# a loop qualifies if it has one of `red`; vector fadds are only counted
acc = re.compile(r"\b(fadd)(?=\s+z)")
# the last backward branch to a label closes the whole loop
back = {}
for i, l in enumerate(lines):
    m = re.match(r"^\s+b(\.\w+)?\s+(\.LBB\w+)$", l) or re.match(r"^\s+(cbn?z|tbn?z)\s+.*,\s*(\.LBB\w+)$", l)
    if m and m.group(2) in labels and labels[m.group(2)] < i: back[m.group(2)] = i
seen = set()
for i, l in enumerate(lines):
    m = re.match(r"^\s+b(\.\w+)?\s+(\.LBB\w+)$", l) or re.match(r"^\s+(cbn?z|tbn?z)\s+.*,\s*(\.LBB\w+)$", l)
    if not m or back.get(m.group(2)) != i: continue
    body = [x for x in lines[labels[m.group(2)]:i + 1]
            if x.startswith("\t") and not x.strip().startswith((".", "//"))]
    if not any(red.search(x) for x in body) and not (os.environ.get("MCA_VECTOR_ACC") and any(acc.search(x) for x in body)): continue
    if tuple(body) in seen: continue
    seen.add(tuple(body))
    p = subprocess.run([mca, "-mtriple=aarch64", f"-mcpu={mcpu}", "-mattr=+sve2", "-iterations=200"],
                       input="\n".join(body) + "\n", capture_output=True, text=True)
    cyc = re.search(r"Total Cycles:\s+(\d+)", p.stdout); its = re.search(r"Iterations:\s+(\d+)", p.stdout)
    kinds = {}
    for x in body:
        if (r := red.search(x) or acc.search(x)): kinds[r.group(1)] = kinds.get(r.group(1), 0) + 1
    print(f"loop at {m.group(2)}: {len(body)} insns, " + ", ".join(f"{n}x {k}" for k, n in sorted(kinds.items()))
          + (f" -> {int(cyc.group(1))/int(its.group(1)):.1f} cycles/iteration" if cyc else " -> " + p.stderr[:300]))
