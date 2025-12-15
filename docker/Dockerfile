FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04@sha256:2d913b09e6be8387e1a10976933642c73c840c0b735f0bf3c28d97fc9bc422e0


# 基础环境设置
ENV TZ=Asia/Shanghai \
    DEBIAN_FRONTEND=noninteractive \
    PATH=/opt/miniconda3/bin:$PATH

# 安装 uv 工具
COPY --from=ghcr.io/astral-sh/uv:0.5.1 /uv /uvx /bin/

WORKDIR /workspace

# 安装基本工具
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget vim git git-lfs net-tools linux-headers-generic build-essential clang \
 && apt-get install -y curl \
 && apt-get install -y cmake \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

# 安装 Miniconda
RUN wget -O /opt/miniconda.sh "https://mirrors.bfsu.edu.cn/anaconda/miniconda/Miniconda3-py310_25.1.1-1-Linux-x86_64.sh" \
 && chmod +x /opt/miniconda.sh \
 && /bin/bash /opt/miniconda.sh -b -p /opt/miniconda3 \
 && rm /opt/miniconda.sh

# 设置使用 bash 作为 shell 并开启登录模式
SHELL ["/bin/bash", "--login", "-c"]

# # 更新 Conda 并创建新环境
# RUN conda update -n base -c defaults conda -y \
#     && conda create -n model_train_finetune_env python=3.11 -y \
#     && conda init bash


# COPY ../../pyproject.toml /workspace/pyproject.toml
# COPY ../../uv.lock /workspace/uv.lock
# COPY ../../packages/openpi-client/pyproject.toml /workspace/packages/openpi-client/pyproject.toml

# RUN echo "conda activate openpi_finetune" >> ~/.bashrc \
#  && source ~/.bashrc \
#  && source activate \
#  && conda activate openpi_finetune \
#  && GIT_LFS_SKIP_SMUDGE=1 uv sync --index-url https://pypi.tuna.tsinghua.edu.cn/simple

#  此步骤主要是用于安装本地构建的依赖包，如 openpi-client
# RUN conda run -n openpi_finetune GIT_LFS_SKIP_SMUDGE=1 uv pip install -e . --index-url https://pypi.tuna.tsinghua.edu.cn/simple
 
