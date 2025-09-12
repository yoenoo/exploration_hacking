set -xe

CUDA_DEVICE_ORDER=PCI_BUS_ID 
CUDA_VISIBLE_DEVICES=1 trl vllm-serve \
  --model $1 \
  --port 8000 \
  --tensor-parallel-size 1 \
  --data-parallel-size 1 \
  --gpu-memory-utilization 0.95 \
  --max-model-len 11674