#!/usr/bin/env python3
"""Where does streaming mode go, for every collected test kernel, on SME?

usage: survey.py OPT OUT_JSON DIR...      (env vars select the variant, e.g.
       TRITON_ARM64_NO_PACK_HOIST=1 TRITON_ARM64_NO_OUTLINE_RESULTS=1 TRITON_ARM64_NO_BUFFER_HOIST=1 for main)
Per unique input (dedup by content): whether a public function is streaming as
a whole (outlining failed or not tried), how many <fn>_sme_N / <fn>_ssve_N
functions were outlined, and the calls left inside streaming functions.
"""
import concurrent.futures as cf, hashlib, json, os, re, subprocess, sys

OPT, OUT, DIRS = sys.argv[1], sys.argv[2], sys.argv[3:]
seen = {}
for d in DIRS:
    for f in sorted(os.listdir(d)):
        if f.endswith((".mlir", ".mlirbc")):
            p = os.path.join(d, f)
            h = hashlib.sha1(open(p, "rb").read()).hexdigest()
            seen.setdefault(h, {"path": p, "tests": []})["tests"].append(f"{os.path.basename(d)}/{f}")

def funcs(text):
    """name -> (attrs line, body) for each func.func."""
    out = {}
    for m in re.finditer(r"^  func\.func (\w+ )?@(\w+)\((.*)$", text, re.M):
        start = m.end(); depth = 1; i = text.index("{", m.start() + len(m.group(0)) - 1) + 1 if m.group(0).rstrip().endswith("{") else start
        # body = until a line that is exactly "  }"
        end = text.find("\n  }\n", start)
        out[m.group(2)] = (m.group(1) or "", m.group(3), text[start:end])
    return out

def one(e):
    p = subprocess.run([OPT, "--vector-ext=sme", "-optimize-triton-shared", e["path"]],
                       capture_output=True, text=True)
    r = {"file": e["path"], "tests": e["tests"], "rc": p.returncode}
    if p.returncode:
        r["error"] = p.stderr.strip().splitlines()[-1][:200] if p.stderr.strip() else ""
        return r
    fs = funcs(p.stdout)
    r["whole_streaming"] = [n for n, (vis, sig, _) in fs.items() if "private" not in vis and "llvm.arm_streaming" in sig]
    r["sme"] = [n for n in fs if re.search(r"_sme_\d+$", n)]
    r["ssve"] = [n for n in fs if re.search(r"_ssve_\d+$", n)]
    calls = {}
    for n, (vis, sig, body) in fs.items():
        if "llvm.arm_streaming" in sig or "llvm.arm_locally_streaming" in sig:
            c = sorted(set(re.findall(r"(memref\.alloc\b|memref\.dealloc|memref\.copy|func\.call @\w+)", body)))
            if c: calls[n] = c
    r["calls_in_streaming"] = calls
    return r

with cf.ThreadPoolExecutor(int(os.environ.get("JOBS", "6"))) as ex:
    res = list(ex.map(one, seen.values()))
json.dump(res, open(OUT, "w"), indent=1)
ok = [r for r in res if r["rc"] == 0]
name = lambda r: re.sub(r"__\w{10}\.mlir(bc)?$", "", r["tests"][0])
print(f"{len(res)} unique kernels, {len(res)-len(ok)} failed to compile")
print(f"  with ZA code outlined (_sme_N): {sum(1 for r in ok if r['sme'])}   with SSVE outlined (_ssve_N): {sum(1 for r in ok if r['ssve'])}")
ws = [r for r in ok if r["whole_streaming"]]
print(f"  public function streaming as a whole: {len(ws)}")
for r in ws: print(f"     {name(r)}  (+{len(r['tests'])-1} same)")
cs = [r for r in ok if r["calls_in_streaming"]]
print(f"  kernels with allocs/copies/calls inside streaming code: {len(cs)}")
for r in cs: print(f"     {name(r)}: {r['calls_in_streaming']}")
for r in res:
    if r["rc"]: print(f"  FAILED {name(r)}: {r.get('error')}")
