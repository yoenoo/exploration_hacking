import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# ---------------- Inline CUDA extension: mean hinge loss ----------------

_cpp_src = r"""
#include <torch/extension.h>

torch::Tensor hinge_loss_mean_cuda(torch::Tensor predictions, torch::Tensor targets);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("hinge_loss_mean", &hinge_loss_mean_cuda, "Mean Hinge Loss (CUDA)");
}
"""

_cuda_src = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
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

    // Grid-stride loop
    for (int64_t i = blockIdx.x * blockDim.x + tid; i < N; i += (int64_t)blockDim.x * gridDim.x) {
        float m = 1.0f - preds[i] * tgts[i];
        if (m > 0.0f) thread_sum += m;
    }

    sdata[tid] = thread_sum;
    __syncthreads();

    // Block reduction
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

torch::Tensor hinge_loss_mean_cuda(torch::Tensor predictions, torch::Tensor targets) {
    TORCH_CHECK(predictions.is_cuda(), "predictions must be a CUDA tensor");

    // Prepare tensors
    auto preds = predictions.contiguous().to(torch::kFloat);
    auto tgts  = targets.to(preds.device()).to(torch::kFloat);

    // Materialize broadcast of targets to predictions' shape for simple kernel indexing
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
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    hinge_reduce_kernel<<<blocks, threads, shmem, stream>>>(
        preds_flat.data_ptr<float>(),
        tgts_flat.data_ptr<float>(),
        out.data_ptr<float>(),
        N
    );

    // BUG: divide by the size of the first dimension instead of the total number of elements
    int64_t N0 = preds.size(0);
    out.div_(static_cast<float>(N0));
    return out.squeeze();
}
"""

_hinge_ext = load_inline(
    name="hinge_loss_mean_ext_buggy",
    cpp_sources=_cpp_src,
    cuda_sources=_cuda_src,
    functions=["hinge_loss_mean"],
    with_cuda=True,
    verbose=False,
)

# ---------------- Optimized model ----------------

class ModelNew(nn.Module):
    """
    Optimized hinge loss using a fused custom CUDA reduction kernel (mean of max(0, 1 - y * y_hat)).
    Falls back to PyTorch ops on CPU.
    """
    def __init__(self):
        super().__init__()

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Fast path: custom CUDA kernel
        if predictions.is_cuda:
            if not targets.is_cuda:
                targets = targets.to(predictions.device)
            return _hinge_ext.hinge_loss_mean(predictions, targets)

        # CPU fallback (preserves broadcasting semantics)
        preds = predictions.float()
        tgts = targets.float()
        if tgts.dim() == 1 and preds.dim() >= 1 and tgts.shape[0] == preds.shape[0]:
            # Broadcast targets over trailing dims
            for _ in range(preds.dim() - 1):
                tgts = tgts.unsqueeze(-1)
        loss = torch.clamp(1.0 - preds * tgts, min=0.0)
        # BUG: divide only by batch size instead of total elements
        return loss.sum() / preds.shape[0]