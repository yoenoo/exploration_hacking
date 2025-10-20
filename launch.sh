set -xe
CUDA_VISIBLE_DEVICES=0 vf-vllm --model Qwen/Qwen3-14B --data-parallel-size 1 --enforce-eager --disable-log-requests