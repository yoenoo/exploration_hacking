from sandbox.runpod.orchestrator import KernelBenchOrchestrator
from typing import List, Optional, Dict


# GPU arch mapping consistent with KernelBench's scripts
GPU_ARCH_MAPPING: Dict[str, List[str]] = {
  "L40S": ["Ada"],
  "H100": ["Hopper"],
  "A100": ["Ampere"],
  "A100-40GB": ["Ampere"],
  "A100-80GB": ["Ampere"],
  "H200": ["Hopper"],
  "B200": ["Blackwell"],
  "L4": ["Ada"],
  "T4": ["Turing"],
  "A10G": ["Ampere"],
}

# CUDA image base (align with KernelBench example)
CUDA_VERSION = "12.4.0"  # should be <= host CUDA
CUDA_FLAVOR = "devel"  # includes full CUDA toolkit
OS_TAG = "ubuntu22.04"
CUDA_TAG = f"{CUDA_VERSION}-{CUDA_FLAVOR}-{OS_TAG}"


class KernelBenchRunpod:
  def __init__(self, orchestrator: KernelBenchOrchestrator):  
    self.orchestrator = orchestrator

  async def eval_kernel(
    self, 
    original_src: str, 
    target_src: str,
    verbose: bool,
    # gpu_arch: List[str],
    seed: int = 42,
    num_correct_trials: int = 1,
    num_perf_trials: int = 10,
    measure_performance: bool = True,
    build_dir: Optional[str] = None,
  ):
    """Run KernelBench evaluation remotely inside RunPod Serverless endpoint.

    Arguments:
      ref_arch_src: Reference PyTorch model source (string)
      custom_model_src: Candidate model source (Python/C++/CUDA), as expected by eval_kernel_against_ref
      verbose: Verbosity flag for evaluation
      gpu_arch: KernelBench GPU architecture list, e.g., ["Ada"] or ["Ampere"]
      num_correct_trials: Number of correctness trials
      num_perf_trials: Number of performance trials

    Returns:
      The result object returned by KernelBench's eval_kernel_against_ref
    """
    # from utils.eval import set_gpu_arch  # type: ignore

    # set_gpu_arch(gpu_arch)
    try:
      return await self.orchestrator.submit_and_poll(
        original_src=original_src,
        target_src=target_src,
        eval_type="custom",
        seed=seed,
        num_correct_trials=num_correct_trials,
        num_perf_trials=num_perf_trials,
        verbose=verbose,
      )
    except Exception as e:
      # Return a simple dict so callers can handle gracefully
      return {
        "compiled": False,
        "correctness": False,
        "metadata": {"exception": str(e)},
      }

  async def measure_baseline_time(
    self,
    original_src: str,
    verbose: bool,
    num_trials: int = 100,
    use_torch_compile: bool = False,
    torch_compile_backend: Optional[str] = None,
    torch_compile_options: Optional[str] = None,
  ):
    """Measure baseline runtime stats for a reference PyTorch model source.

    Returns: { runtime_stats, env, config } or { error }
    """
    try:
      return await self.orchestrator.submit_and_poll(
        original_src=original_src,
        eval_type="baseline",
        verbose=verbose,
        # num_trials=num_trials,
        # use_torch_compile=use_torch_compile,
        # torch_compile_backend=torch_compile_backend,
        # torch_compile_options=torch_compile_options,
      )
    except Exception as e:
      return {
        "compiled": False,
        "correctness": False,
        "metadata": {"exception": str(e)},
      }