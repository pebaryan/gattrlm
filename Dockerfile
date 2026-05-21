FROM nvcr.io/nvidia/pytorch:25.09-py3

WORKDIR /resource
COPY . .
RUN pip install --no-cache-dir -e ".[data]"
