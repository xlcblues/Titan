"""Temporary toolchain check: compile a minimal CUDA extension (not committed)."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

import torch
from torch.utils.cpp_extension import load_inline


def main():
    build_dir = os.path.join(os.path.dirname(__file__), "_build_toolcheck")
    os.makedirs(build_dir, exist_ok=True)
    ext = load_inline(
        name="_titan_toolcheck",
        cpp_sources=["torch::Tensor add_one(torch::Tensor x) { return x + 1; }"],
        cuda_sources=[
            """
            #include <torch/extension.h>
            __global__ void add_one_kernel(float* out, const float* in, long n) {
                long i = blockIdx.x * blockDim.x + threadIdx.x;
                if (i < n) out[i] = in[i] + 1.0f;
            }
            torch::Tensor add_one_cuda(torch::Tensor x) {
                auto out = torch::empty_like(x);
                long n = x.numel();
                add_one_kernel<<<(n + 255) / 256, 256>>>(out.data_ptr<float>(), x.data_ptr<float>(), n);
                return out;
            }
            """,
        ],
        functions=["add_one"],
        build_directory=build_dir,
        verbose=False,
    )
    x = torch.ones(8, device="cuda")
    y = ext.add_one(x)
    torch.cuda.synchronize()
    print("toolchain ok:", y.tolist())


if __name__ == "__main__":
    main()
