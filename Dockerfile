FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

# Install Python
RUN apt-get update && apt-get install -y python3 python3-pip && rm -rf /var/lib/apt/lists/*

WORKDIR /

RUN pip3 install --no-cache-dir runpod numpy torch pydantic
COPY eval.py rp_handler.py test_input.json /
CMD ["python3", "-u", "rp_handler.py"]
