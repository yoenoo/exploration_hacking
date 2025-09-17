set -xe

# i like vim and tmux
apt-get update && apt-get install -y vim tmux
apt-get update && apt-get install -y python3.10-dev build-essential # essential for KernelBench

# env setup
git clone https://github.com/yoenoo/dotfiles.git
cp dotfiles/tmux.conf ~/.tmux.conf
cp dotfiles/vimrc ~/.vimrc
rm -rf dotfiles

# uv setup
pip install uv 
uv venv --python=3.10
. .venv/bin/activate

# install dependencies
uv pip install wandb
uv pip install torch transformers accelerate datasets trl[vllm] peft
uv pip install vllm==0.10.1

# kernelbench
uv pip install python-dotenv pydra-config hydra-core omegaconf
uv pip install ninja
uv pip install together openai anthropic google-generativeai ## TODO: remove 
uv pip install random-word

# bigcodebench
uv pip install pqdm tree-sitter-python tree_sitter
uv pip install \
  beautifulsoup4 blake3 chardet cryptography \
  cryptography datetime Django \
  dnspython docxtpl Faker \
  flask_login flask_restful flask_wtf \
  Flask-Mail flask folium \
  gensim geopandas geopy \
  holidays Levenshtein \
  librosa lxml \
  mechanize natsort networkx \
  nltk numba numpy \
  opencv-python-headless openpyxl pandas \
  Pillow prettytable psutil \
  pycryptodome pyfakefs pyquery \
  pytesseract pytest python_http_client \
  python-dateutil python-docx python-Levenshtein-wheels \
  pytz PyYAML requests_mock \
  requests Requests rsa \
  scikit-image scikit-learn scipy \
  seaborn selenium sendgrid \
  shapely soundfile statsmodels \
  sympy textblob \
  texttable Werkzeug wikipedia \
  wordcloud wordninja WTForms \
  xlrd xlwt xmltodict

# git setup
git config --global user.name "Yeonwoo Jang"
git config --global user.email "yjang385@gmail.com"
git config pull.rebase false