set -xe

CUDA_DEVICE_ORDER=PCI_BUS_ID 
CUDA_VISIBLE_DEVICES=4,5,6,7 trl vllm-serve \
  --model $1 \
  --port 8000 \
  --tensor-parallel-size 2 \
  --data-parallel-size 2 \
  --gpu-memory-utilization 0.95 \
  --max-model-len 11674