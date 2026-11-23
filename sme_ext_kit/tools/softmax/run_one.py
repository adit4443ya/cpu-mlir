"""Compile and run one softmax through Triton or torch-mlir, keeping every stage.

usage: python run_one.py <triton|torch> <rows> <cols> [sme|sve]
Run it from an empty directory with TRITON_ARM64_SAVE_TEMPS=1: the stages
(-tt.mlir, -ttshared.mlir / .mlirbc, -llvm-dialect.mlir, .ll, .exe, .raw)
land in the current directory.
"""
import os, sys
import torch

frontend, rows, cols = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
ext = sys.argv[4] if len(sys.argv) > 4 else "sme"
backend = os.environ["ARM64_BACKEND_PATH"]
torch.manual_seed(0)
x = torch.randn(rows, cols)
ref = torch.softmax(x, dim=1)

if frontend == "triton":
    import triton
    from triton.backends.qcom_arm64_backend.driver import ARM64CPUDriver
    triton.runtime.driver.set_active(ARM64CPUDriver(ext))
    sys.path.insert(0, os.path.join(backend, "python/examples/triton_shared_examples"))
    from test_softmax import softmax           # the tutorial kernel, one program per row
    out = softmax(x)
else:
    sys.path.insert(0, os.path.join(backend, "python/torch-mlir"))
    import utils
    class Softmax(torch.nn.Module):
        def forward(self, x):
            return torch.softmax(x, dim=1)
    m = utils.ModelManager(Softmax(), x)
    m.create_linalg_module(); m.write_bytecode_to_file()
    out = m.execute_and_compare(rtol=1e-5, atol=1e-6, options=utils.VECTOR_EXT_OPTIONS[ext])
    print("torch matched:", out)
    sys.exit(0 if out else 1)
print("triton matched:", torch.allclose(out, ref, rtol=1e-5, atol=1e-6))
