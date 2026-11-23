import os, sys, torch, triton
from triton.backends.qcom_arm64_backend.driver import ARM64CPUDriver
ext = sys.argv[1] if len(sys.argv) > 1 else "sme"
triton.runtime.driver.set_active(ARM64CPUDriver(ext))
sys.path.insert(0, os.path.join(os.environ["ARM64_BACKEND_PATH"], "python/examples/triton_shared_examples"))
from test_matmul import matmul
M, K, N = (int(v) for v in (sys.argv[2:5] if len(sys.argv) > 4 else (179, 167, 321)))
torch.manual_seed(0)
a, b = torch.randn(M, K), torch.randn(K, N)
c = matmul(a, b)
print("match:", torch.allclose(c, a @ b, atol=1e-2, rtol=0))
