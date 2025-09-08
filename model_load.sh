set -xe

python3 model_load.py \
  --base_model Qwen/Qwen3-14B \
  --output_dir "grpo-qwen3-14b-checkpoints-manwards" \
  --prompt "Summarize why exploration hacking is dangerous in RL." \
  --use_bf16