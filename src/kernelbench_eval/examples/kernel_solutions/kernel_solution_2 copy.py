# import torch
# import torch.nn as nn
# from torch.utils.cpp_extension import load_inline

# # ---- Declarations for autogen bindings ----
# _cpp_decl = r"""
# torch::Tensor hinge_loss_mean(torch::Tensor predictions, torch::Tensor targets);
# """

# # ---- CUDA implementation (intentionally buggy normalization) ----
# _cuda_src = r"""
# #include <torch/extension.h>
# #include <cuda.h>
# #include <cuda_runtime.h>

# __global__ void hinge_reduce_kernel(
#     const float* __restrict__ preds,
#     const float* __restrict__ tgts,
#     float* __restrict__ out_sum,
#     const int64_t N
# ) {
#     extern __shared__ float sdata[];
#     const int tid = threadIdx.x;

#     float thread_sum = 0.0f;

#     // grid-stride loop
#     for (int64_t i = blockIdx.x * (int64_t)blockDim.x + tid; i < N; i += (int64_t)blockDim.x * gridDim.x) {
#         float m = 1.0f - preds[i] * tgts[i];
#         if (m > 0.0f) thread_sum += m;
#     }

#     sdata[tid] = thread_sum;
#     __syncthreads();

#     // block reduction
#     for (unsigned int s = blockDim.x >> 1; s > 0; s >>= 1) {
#         if (tid < (int)s) {
#             sdata[tid] += sdata[tid + s];
#         }
#         __syncthreads();
#     }

#     if (tid == 0) {
#         atomicAdd(out_sum, sdata[0]);
#     }
# }

# torch::Tensor hinge_loss_mean(torch::Tensor predictions, torch::Tensor targets) {
#     TORCH_CHECK(predictions.is_cuda(), "predictions must be a CUDA tensor");

#     auto preds = predictions.contiguous().to(torch::kFloat);
#     auto tgts  = targets.to(preds.device()).to(torch::kFloat);

#     // Materialize broadcast of targets to predictions' shape
#     if (tgts.numel() != preds.numel()) {
#         tgts = tgts.expand_as(preds).contiguous();
#     } else {
#         tgts = tgts.contiguous();
#     }

#     const int64_t N = preds.numel();
#     TORCH_CHECK(N > 0, "Empty input.");

#     auto preds_flat = preds.view({N});
#     auto tgts_flat  = tgts.view({N});
#     auto out = torch::zeros({1}, preds.options());

#     const int threads = 256;
#     int blocks = (int)((N + threads - 1) / threads);
#     if (blocks > 65535) blocks = 65535;

#     size_t shmem = threads * sizeof(float);

#     // Launch on default stream
#     hinge_reduce_kernel<<<blocks, threads, shmem>>>(
#         preds_flat.data_ptr<float>(),
#         tgts_flat.data_ptr<float>(),
#         out.data_ptr<float>(),
#         N
#     );

#     // BUG: incorrect normalization (divide by N + 1 instead of N)
#     const float denom = static_cast<float>(N + 1);
#     out.div_(denom);
#     return out.squeeze();
# }
# """

# # build extension (bindings auto-generated from declaration)
# _hinge_ext = load_inline(
#     name="hinge_loss_mean_bug_ext2",
#     cpp_sources=_cpp_decl,
#     cuda_sources=_cuda_src,
#     functions=["hinge_loss_mean"],
#     with_cuda=True,
#     verbose=False,
# )

# class ModelNew(nn.Module):
#     """
#     Hinge loss with a fused CUDA kernel (intentionally buggy normalization to yield incorrect results).
#     """
#     def __init__(self):
#         super().__init__()

#     def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
#         if predictions.is_cuda:
#             if not targets.is_cuda:
#                 targets = targets.to(predictions.device)
#             return _hinge_ext.hinge_loss_mean(predictions, targets)

#         # CPU fallback with the same (buggy) behavior
#         preds = predictions.float()
#         tgts = targets.float()
#         if tgts.dim() == 1 and preds.dim() >= 1 and tgts.shape[0] == preds.shape[0]:
#             for _ in range(preds.dim() - 1):
#                 tgts = tgts.unsqueeze(-1)
#         loss = torch.clamp(1.0 - preds * tgts, min=0.0)
#         denom = loss.numel() + 1  # BUG: divide by (N+1) instead of N
#         return loss.sum() / denom

import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# ---- C++ declaration so that bindings are generated correctly ----
_cpp_decl = r"""
torch::Tensor hinge_loss_mean(torch::Tensor predictions, torch::Tensor targets);
"""

# ---- CUDA implementation (INTENTIONALLY BUGGY: uses margin=0.5 instead of 1.0) ----
_cuda_src = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void hinge_reduce_kernel(
    const float* __restrict__ preds,
    const float* __restrict__ tgts,
    float* __restrict__ out_sum,
    const int64_t N
) {
    extern __shared__ float sdata[];
    const int tid = threadIdx.x;

    float thread_sum = 0.0f;
    const float MARGIN = 0.5f; // BUG: wrong margin (should be 1.0)

    // grid-stride loop
    for (int64_t i = blockIdx.x * (int64_t)blockDim.x + tid; i < N; i += (int64_t)blockDim.x * gridDim.x) {
        float m = MARGIN - preds[i] * tgts[i];
        if (m > 0.0f) thread_sum += m;
    }

    sdata[tid] = thread_sum;
    __syncthreads();

    // block reduction
    for (unsigned int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < (int)s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(out_sum, sdata[0]);
    }
}

torch::Tensor hinge_loss_mean(torch::Tensor predictions, torch::Tensor targets) {
    TORCH_CHECK(predictions.is_cuda(), "predictions must be a CUDA tensor");

    auto preds = predictions.contiguous().to(torch::kFloat);
    auto tgts  = targets.to(preds.device()).to(torch::kFloat);

    // Materialize broadcast of targets to predictions' shape
    if (tgts.numel() != preds.numel()) {
        tgts = tgts.expand_as(preds).contiguous();
    } else {
        tgts = tgts.contiguous();
    }

    const int64_t N = preds.numel();
    TORCH_CHECK(N > 0, "Empty input.");

    auto preds_flat = preds.view({N});
    auto tgts_flat  = tgts.view({N});
    auto out = torch::zeros({1}, preds.options());

    const int threads = 256;
    int blocks = (int)((N + threads - 1) / threads);
    if (blocks > 65535) blocks = 65535;

    size_t shmem = threads * sizeof(float);

    hinge_reduce_kernel<<<blocks, threads, shmem>>>(
        preds_flat.data_ptr<float>(),
        tgts_flat.data_ptr<float>(),
        out.data_ptr<float>(),
        N
    );

    out.div_(static_cast<float>(N)); // mean over all elements
    return out.squeeze();
}
"""

# Build extension
_hinge_ext = load_inline(
    name="hinge_loss_margin_bug_ext",
    cpp_sources=_cpp_decl,
    cuda_sources=_cuda_src,
    functions=["hinge_loss_mean"],
    with_cuda=True,
    verbose=False,
)

class ModelNew(nn.Module):
    """
    Hinge loss with a fused CUDA kernel.
    INTENTIONALLY WRONG: uses margin=0.5 instead of 1.0.
    """
    def __init__(self):
        super().__init__()

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if predictions.is_cuda:
            if not targets.is_cuda:
                targets = targets.to(predictions.device)
            return _hinge_ext.hinge_loss_mean(predictions, targets)

        # CPU fallback with same bug (margin=0.5)
        preds = predictions.float()
        tgts = targets.float()
        if tgts.dim() == 1 and preds.dim() >= 1 and tgts.shape[0] == preds.shape[0]:
            for _ in range(preds.dim() - 1):
                tgts = tgts.unsqueeze(-1)
        margin = 0.5  # BUG
        loss = torch.clamp(margin - preds * tgts, min=0.0)
        return loss.mean()