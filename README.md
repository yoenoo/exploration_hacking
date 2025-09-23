# Exploration Hacking Model Organisms
Studying model organisms of [exploration hacking](https://www.lesswrong.com/posts/Dft9vpMnEeWFE3Gc6/exploration-hacking-can-reasoning-models-subvert-rl-1), with a focus on AI R&D sabotage threat model (e.g. AI safety research sabotage).

## System Requirement
The default setup I'm using is 8 x H200 (expensive!). Other NVIDIA Hopper GPUs (e.g. H100) should also work. 

## Tasks
I'm focusing on KernelBench and BigCodeBench, and potentially mixing with SWE-Bench for conditional underperformance threat model.
- [KernelBench](https://huggingface.co/datasets/ScalingIntelligence/KernelBench)
- [BigCodeBench](https://huggingface.co/datasets/bigcode/bigcodebench)

## Models
Below is the list of models I've tested so far:
- [cognition-ai/Kevin-32B](https://huggingface.co/cognition-ai/Kevin-32B)
- [Qwen/Qwen3-30B-A3B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507)
- [Qwen/Qwen3-30B-A3B-Thinking-2507](https://huggingface.co/Qwen/Qwen3-30B-A3B-Thinking-2507)
- [Qwen/Qwen3-14B](https://huggingface.co/Qwen/Qwen3-14B)
- [Qwen/Qwen3-4B-Thinking-2507](https://huggingface.co/Qwen/Qwen3-4B-Thinking-2507)
- [openai/gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b)

## Setup
Add your API keys to `.env`. See `setup.sh` for more detail.

## Useful commands
For model benchmarking, it’s generally better to have multiple GPUs, even if they’re lower-end, since parallelism matters more than raw performance. For rollout generation, however, you’ll want access to stronger GPUs.
```shell
VLLM_LOGGING_LEVEL=WARNING CUDA_LAUNCH_BLOCKING=1 python3 evaluate_model_performance.py
```
The config files in `conf/` folder are responsible for selecting models and various other parameter settings.

To run GRPO training, you first need to spin up vLLM server (refer to `start_vllm_server.sh` for examples). 
Below example assumes you have 8 GPUs:
```shell
CUDA_VISIBLE_DEVICES=4,5,6,7 trl vllm-serve \
  --model <model> \
  --port 8000 \
  --tensor-parallel-size 2 \
  --data-parallel-size 2 \
  --gpu-memory-utilization 0.95 \
  --max-model-len 11674
```

To kick off GRPO training, use command below (or refer to `trainer.sh`):
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
  --num_processes=4 \
  --mixed_precision=bf16 \
  src/grpo/grpo_train.py
```

Note that the training takes a while, even with 8 x H200, depending on the how long the average generations are (usually need at least 4096 tokens for thinkning models), and also bottlenecked by evals (which can take some time), so keep this in mind!

Update (8/17): kernel evals have been significantly optimized and now run much faster. Single RL step (`num_gerations=8` with `generation_batch_size=64`) with 8 x H200 takes ~6 mins.

## Resources
- [Kevin-32B: Multi-Turn RL for Writing CUDA Kernels](https://cognition.ai/blog/kevin-32b)
- [Kevin: Multi-Turn RL for Generating CUDA Kernels](https://arxiv.org/abs/2507.11948)
  - our GRPO reward directly comes from this paper (section 3.2  Kernel Score Design) for single-turn RL
- [KernelBench: Can LLMs Write Efficient GPU Kernels?](https://arxiv.org/abs/2502.10517)
- [Measuring Automated Kernel Engineering (by METR)](https://metr.org/blog/2025-02-14-measuring-automated-kernel-engineering/)
  - [github repo](https://github.com/METR/KernelBenchFiltered/blob/main/evaluate_solution.py)
- [my struggle getting multi-GPU GRPO working...](https://github.com/yoenoo/unsloth_vllm_profiling/tree/master/code)
- [How to Accurately Time CUDA Kernels in Pytorch](https://www.speechmatics.com/company/articles-and-news/timing-operations-in-pytorch)
