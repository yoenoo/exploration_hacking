set -xe

apt-get update && apt-get install -y vim tmux
apt-get update && apt-get install -y python3.10-dev build-essential

uv pip install ninja hf_transfer
uv add 'verifiers[train]' && uv pip install flash-attn --no-build-isolation
