ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:26.05-py3
FROM ${BASE_IMAGE}

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      python3 \
      python3-pip \
      python3-dev \
      libsm6 \
      libxext6 \
      ffmpeg \
      libhdf5-serial-dev \
      libtesseract-dev \
      libgtk-3-0 \
      libtbb12 \
      libgl1 \
      libatlas-base-dev \
      libopenblas-dev \
      build-essential \
      python3-setuptools \
      make \
      cmake \
      nasm \
      yasm \
      pkg-config \
      git \
      libgnutls28-dev \
      libvpx-dev \
      libopus-dev \
      libvorbis-dev \
      libmp3lame-dev \
      libfreetype-dev \
      libass-dev \
      libaom-dev \
      libdav1d-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

WORKDIR /workspace

COPY deployment_scripts/pyproject.toml .

# Install the three Thor-specific packages from the Jetson CUDA 13 index first.
# Everything else is then resolved strictly from the fast PyPI mirror. Keeping
# the indexes in separate pip invocations prevents pip from selecting a slow
# duplicate candidate from the Jetson index for ordinary PyPI packages.
RUN python3 -m pip install \
      --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
      --ignore-installed --no-deps PyYAML==6.0.2 && \
    python3 -m pip install \
      --index-url https://pypi.jetson-ai-lab.io/sbsa/cu130 \
      --trusted-host pypi.jetson-ai-lab.io \
      --no-deps \
      torchcodec==0.7.0 \
      diffusers==0.36.0.dev0 \
      decord2 && \
    python3 -m pip install \
      --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
      -e '.[thor]' && \
    python3 -m pip install \
      --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
      onnxslim lief
