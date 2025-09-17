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
uv pip install wandb
uv pip install torch transformers accelerate datasets trl[vllm] peft
uv pip install --upgrade datasets
uv pip install vllm==0.10.1
uv pip install python-dotenv pydra-config hydra-core omegaconf
uv pip install ninja
uv pip install together openai anthropic google-generativeai ## TODO: remove 
# uv pip install unsloth  ## migrated off unsloth
uv pip install random-word

# bigcodebench
uv pip install pqdm
uv pip install tree-sitter-python tree_sitter
uv pip install seaborn scikit-learn bs4 flask nltk scikit-image faker

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

# essential for KernelBench
apt-get update && apt-get install -y python3.10-dev build-essential

git config --global user.name "Yeonwoo Jang"
git config --global user.email "yjang385@gmail.com"
git config pull.rebase false