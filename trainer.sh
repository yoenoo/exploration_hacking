set -xe

CUDA_DEVICE_ORDER=PCI_BUS_ID 
NCCL_ASYNC_ERROR_HANDLING=1 

TOKENIZERS_PARALLELISM=true PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True VLLM_LOGGING_LEVEL=WARNING CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
  --num_processes=2 \
  --module src.train.trainer \
  --config-name config_qwen3_14b_rust
