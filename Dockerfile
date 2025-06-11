# ✅ 基础镜像：Ubuntu 18.04 + CUDA 12.1 + CUDNN8 + NVIDIA Driver
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu18.04

# ✅ 环境变量（提前设置 CUDA 等路径）
ENV DEBIAN_FRONTEND=noninteractive
ENV PATH /opt/conda/bin:$PATH
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8
ENV CUDA_HOME /usr/local/cuda

# ✅ 安装基本工具
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    curl \
    ca-certificates \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    build-essential \
    gcc \
    g++ \
    vim \
    && rm -rf /var/lib/apt/lists/*

# ✅ 安装 Miniconda
WORKDIR /opt
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-py38_4.12.0-Linux-x86_64.sh -O miniconda.sh && \
    bash miniconda.sh -b -p /opt/conda && \
    rm miniconda.sh && \
    /opt/conda/bin/conda clean -ya

# ✅ 拷贝你的 Conda 环境文件
COPY cmnext_backup.yml /opt/cmnext_backup.yml

# ✅ 创建 Conda 环境
RUN /opt/conda/bin/conda env create -f /opt/cmnext_backup.yml

# ✅ 激活环境
SHELL ["conda", "run", "-n", "cmnext", "/bin/bash", "-c"]

# ✅ 安装你手动装的 CUDA 12.1 PyTorch
RUN pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121

# ✅ 拷贝你的项目文件
WORKDIR /workspace
COPY . /workspace

# ✅ 默认启动命令
CMD ["bash"]
