set -xe

CUDA_DEVICE_ORDER=PCI_BUS_ID 
NCCL_ASYNC_ERROR_HANDLING=1 

TOKENIZERS_PARALLELISM=true VLLM_LOGGING_LEVEL=WARNING CUDA_VISIBLE_DEVICES=0 accelerate launch \
  --num_processes=1 \
  --module src.train.trainer \
  --config-name config_bcb