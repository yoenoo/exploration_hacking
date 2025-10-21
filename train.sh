set -xe

CUDA_VISIBLE_DEVICES=1 accelerate launch --num-processes 1 --config-file train/configs/zero3.yaml train/grpo/train_kernelbench.py 
