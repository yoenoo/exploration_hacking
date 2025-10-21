set -xe
#
# env setup
git clone https://github.com/yoenoo/dotfiles.git
cp dotfiles/tmux.conf ~/.tmux.conf
cp dotfiles/vimrc ~/.vimrc
rm -rf dotfiles

apt-get update && apt-get install -y vim tmux
apt-get update && apt-get install -y python3.10-dev build-essential

uv pip install ninja hf_transfer
uv add 'verifiers[train]' && uv pip install flash-attn --no-build-isolation
