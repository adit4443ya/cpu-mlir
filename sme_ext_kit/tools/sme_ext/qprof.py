#!/usr/bin/env python3
"""Dynamic instruction profile of one kernel executable under qemu.

usage: qprof.py EXE [QEMU_CPU]      (run it anywhere; EXE runs in its own dir)
  QEMU_CPU default: max,sme_fa64=off,sve-default-vector-length=16,sme-default-vector-length=16
  (16 bytes = 128-bit SVE and SME vectors; raise sme-default-vector-length to
  model a wider streaming vector length, e.g. 64 = 512 bits)

Prints, per function symbol, the number of instructions executed, plus how
many times streaming mode was entered (smstart / smstart sm). qemu does not
model timing: these are instruction counts, not cycles.
"""
import collections, os, re, subprocess, sys

exe = os.path.abspath(sys.argv[1])
cpu = sys.argv[2] if len(sys.argv) > 2 else \
    "max,sme_fa64=off,sve-default-vector-length=16,sme-default-vector-length=16"
log = exe + ".qemu.log"
subprocess.run([os.environ.get("QEMU_PATH", "/usr/bin/qemu-aarch64-static"), "-cpu", cpu, "-d", "in_asm,exec,nochain",
                "-D", log, exe], cwd=os.path.dirname(exe), check=True,
               stdout=subprocess.DEVNULL)

# little-endian words as qemu prints them in OBJD-T lines
SMSTART = {"7f4703d5": "smstart", "7f4303d5": "smstart sm", "7f4503d5": "smstart za"}
size, sym, starts = {}, {}, collections.Counter()
dyn, cur, cursym = collections.Counter(), None, None
nstart = collections.Counter()
for line in open(log, errors="replace"):
    if line.startswith("IN:"):
        cursym = line[3:].strip() or "?"; cur = None; continue
    if cursym is not None:
        m = re.match(r"0x([0-9a-f]+):", line)
        if m and cur is None:
            cur = int(m.group(1), 16); size[cur] = 0; sym[cur] = cursym; continue
        if line.startswith("OBJD-T:") and cur is not None:
            words = re.findall(r"[0-9a-f]{8}", line.split(":", 1)[1])
            size[cur] += len(words)
            for w in words:
                if w in SMSTART: starts[cur] += 1
            continue
        if line.strip() == "": cursym = None; cur = None; continue
    m = re.search(r"\[[0-9a-f]+/([0-9a-f]+)/", line)
    if line.startswith("Trace") and m:
        pc = int(m.group(1), 16)
        if pc in size:
            dyn[sym[pc]] += size[pc]
            nstart[sym[pc]] += starts[pc]
os.remove(log)

KERNEL = os.path.basename(exe).rsplit(".", 1)[0]
def is_kernel(s):  # the kernel and the functions outlined from it (<kernel>_sme_0, ...)
    return s == "memrefCopy" or s == KERNEL or s.startswith(KERNEL + "_") or s.startswith("_mlir_ciface_" + KERNEL)

tot = 0
for s, v in dyn.most_common():
    if is_kernel(s):
        tot += v
        print(f"{v:12d}  {s}" + (f"   (enters streaming {nstart[s]}x)" if nstart[s] else ""))
other = sum(v for s, v in dyn.items() if not is_kernel(s))
print(f"{tot:12d}  = kernel total      ({other} in main/libc/launcher)")
