# Modern-glibc container for SoulX-LiveAct, mirroring the fix found for SoulX-FlashHead:
# a WSL2 Ubuntu 20.04 base (glibc 2.31) forces flash-attn/vllm-adjacent packages to build
# CUDA kernels from source. This image's newer glibc lets prebuilt wheels install directly.
FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel

WORKDIR /workspace

# opencv-python's Python bindings dlopen libxcb/libGL etc even in headless/no-display use.
# git is needed to clone SageAttention/LightX2V at build time (not in the base image).
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libxcb1 libxext6 libsm6 libxrender1 git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- Step 2: SageAttention (fp8 attention kernel) ---
# `docker build` has no GPU access (unlike `docker run --gpus all`), so setup.py can't
# auto-detect compute capability -- must set TORCH_CUDA_ARCH_LIST explicitly. 8.9 = Ada
# Lovelace (this machine's RTX 4060 Ti); adjust/add architectures for other target GPUs.
ENV TORCH_CUDA_ARCH_LIST="8.9"
RUN pip install --no-cache-dir ninja setuptools wheel && \
    git clone --depth 1 --branch v2.2.0 https://github.com/thu-ml/SageAttention.git /tmp/SageAttention && \
    cd /tmp/SageAttention && python setup.py install && \
    rm -rf /tmp/SageAttention

# --- Step 3: vllm (fp8 gemm kernel) ---
RUN pip install --no-cache-dir vllm==0.11.0

# --- Step 4: LightVAE (from ModelTC/LightX2V) ---
# setup_vae.py's own install_requires is tiny (torch/numpy/einops/loguru, already present),
# but pyproject.toml's [project.dependencies] carries LightX2V's full serving-stack deps
# (redis, aioboto3, alibabacloud SDK, asyncpg...) which setuptools uses INSTEAD OF
# setup_vae.py's minimal list once a pyproject.toml with real dependencies exists --
# hence --no-deps and installing the directory directly rather than running setup_vae.py.
# Also patches pyproject.toml's `license = "Apache-2.0"` (PEP 639 SPDX string) to the
# older `license = {text = "Apache-2.0"}` table form some setuptools versions require.
# `av` is a genuine runtime dependency of lightx2v (imported directly, e.g. ltx2_media_io.py)
# that --no-deps skips -- install explicitly. Python 3.11 here (unlike the WSL2 Python 3.10
# venv) satisfies av>=18's python_requires, so no version pin needed.
RUN pip install --no-cache-dir av
RUN git clone --depth 1 https://github.com/ModelTC/LightX2V /tmp/LightX2V && \
    sed -i 's/^license = "Apache-2.0"$/license = {text = "Apache-2.0"}/' /tmp/LightX2V/pyproject.toml && \
    pip install --no-cache-dir --no-deps --no-build-isolation /tmp/LightX2V && \
    rm -rf /tmp/LightX2V

# Code and model weights (~55GB) are mounted at runtime, not baked into the image -- see
# the `docker run -v` example in CLAUDE.md / README, not re-downloaded inside the container.
