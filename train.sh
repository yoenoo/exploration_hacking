set -xe

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=1 accelerate launch --num-processes 1 --config-file train/configs/zero3.yaml train/grpo/train_kernelbench.py 
