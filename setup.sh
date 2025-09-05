set -xe

# i like vim
apt-get update && apt-get install -y vim

# uv install
# curl -LsSf https://astral.sh/uv/install.sh | sh
# . $HOME/.local/bin/env
pip install uv 

uv venv --python=3.12
. .venv/bin/activate

# install dependencies
uv pip install torch transformers accelerate datasets trl peft
uv pip install --upgrade datasets
uv pip install vllm
uv pip install python-dotenv pydra-config hydra-core omegaconf
uv pip install ninja
uv pip install together openai anthropic google-generativeai
# uv pip install unsloth  ## migrated off unsloth
uv pip install wandb
uv pip install random-word

# bigcodebench
uv pip install pqdm
uv pip install tree-sitter-python
uv pip install tree_sitter
uv pip install seaborn
uv pip install scikit-learn

# essential for KernelBench
apt-get update && apt-get install -y python3.10-dev build-essential

git config --global user.name "Yeonwoo Jang"
git config --global user.email "yjang385@gmail.com"
git config pull.rebase false