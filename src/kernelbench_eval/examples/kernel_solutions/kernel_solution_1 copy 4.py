import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

_hinge_loss_decl = r"""
torch::Tensor hinge_loss_mean_cuda(torch::Tensor predictions, torch::Tensor targets);
"""

_hinge_loss_src = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>

namespace {

__global__ void hinge_mean_reduce_kernel(
    const float* __restrict__ preds,
    const float* __restrict__ tgts,
    float* __restrict__ out_sum,
    const int64_t Npred,
    const int64_t Ntgt,
    const int64_t col_sz
) {
    extern __shared__ float sdata[];
    const int tid = threadIdx.x;
    float thread_sum = 0.0f;

    // Grid-stride loop
    for (int64_t i = blockIdx.x * blockDim.x + tid; i < Npred; i += (int64_t)blockDim.x * gridDim.x) {
        int64_t row;
        if (Ntgt == 1) {
            row = 0;
        } else if (col_sz > 0) {
            // Broadcasting targets over trailing dimensions, targets align with dim0
            row = i / col_sz;
        } else {
            // Fallback: elementwise (when shapes already match)
            row = i;
            if (row >= Ntgt) row = row % Ntgt; // very defensive
        }

        float m = 1.0f - preds[i] * tgts[row];
        if (m > 0.0f) thread_sum += m;
    }

    sdata[tid] = thread_sum;
    __syncthreads();

    // Block reduction
    for (unsigned int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(out_sum, sdata[0]);
    }
}

} // namespace

torch::Tensor hinge_loss_mean_cuda(torch::Tensor predictions, torch::Tensor targets) {
    TORCH_CHECK(predictions.is_cuda(), "predictions must be a CUDA tensor");
    TORCH_CHECK(targets.is_cuda(), "targets must be a CUDA tensor");

    // Ensure float32 and contiguous
    auto preds = predictions.contiguous().to(torch::kFloat);
    auto tgts  = targets.contiguous().to(torch::kFloat);

    // Flatten predictions to 1D
    const int64_t Npred = preds.numel();
    auto preds_flat = preds.view({Npred});

    // Flatten targets to 1D
    const int64_t Ntgt = tgts.numel();
    auto tgts_flat = tgts.view({Ntgt});

    TORCH_CHECK(Npred > 0, "Empty predictions");

    // Determine broadcast over dim 0 if possible
    int64_t col_sz = -1;
    if (Ntgt > 0 && (Npred % Ntgt == 0)) {
        col_sz = Npred / Ntgt; // each target value broadcasts over col_sz contiguous elements
    }

    auto out = torch::zeros({1}, preds_flat.options());

    // Configure launch
    int threads = 256;
    int blocks = (int)((Npred + threads - 1) / threads);
    // Cap blocks to something reasonable relative to SM count
    int sm_count = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
    int max_blocks = sm_count * 32; // plenty of parallelism
    if (blocks > max_blocks) blocks = max_blocks;

    size_t shmem = threads * sizeof(float);
    auto stream = at::cuda::getCurrentCUDAStream();

    hinge_mean_reduce_kernel<<<blocks, threads, shmem, stream>>>(
        preds_flat.data_ptr<float>(),
        tgts_flat.data_ptr<float>(),
        out.data_ptr<float>(),
        Npred,
        Ntgt,
        col_sz
    );

    // Mean
    out.div_(static_cast<float>(Npred));
    return out.squeeze();
}
"""

# Compile the inline CUDA extension
_hinge_loss_ext = load_inline(
    name="hinge_loss_cuda_ext",
    cpp_sources=_hinge_loss_decl,
    cuda_sources=_hinge_loss_src,
    functions=["hinge_loss_mean_cuda"],
    with_cuda=True,
    verbose=False,
)


class ModelNew(nn.Module):
    """
    Optimized model computing mean hinge loss using a custom CUDA kernel.

    Forward:
        predictions: Tensor of shape (N, ...) on CUDA or CPU
        targets:     Tensor of shape (N,) or broadcastable to predictions along dim 0

    Returns:
        Scalar tensor: mean(max(0, 1 - predictions * targets))
    """
    def __init__(self):
        super().__init__()

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Fast CUDA path with custom kernel
        if predictions.is_cuda and targets.is_cuda:
            return _hinge_loss_ext.hinge_loss_mean_cuda(predictions, targets)

        # CPU / fallback path using PyTorch ops (keeps broadcasting semantics)
        preds = predictions.float()
        tgts = targets.float()
        # Ensure broadcasting along dim 0 when targets is 1D of length N
        if tgts.dim() == 1 and preds.dim() >= 1 and tgts.shape[0] == preds.shape[0]:
            # Broadcast targets over trailing dims
            for _ in range(preds.dim() - 1):
                tgts = tgts.unsqueeze(-1)
        loss = torch.clamp(1.0 - preds * tgts, min=0.0)
        return loss.mean()