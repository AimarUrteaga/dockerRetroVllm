# syntax=docker/dockerfile:1


# docker run \
# --runtime nvidia \
# --gpus all \
# -v /home/usuario/vLLMDoker/modelos:/mnt/model \
# -p 8000:8000 \
# --ipc=host \
# --env "TRANSFORMERS_OFFLINE=1" \
# --env "HF_DATASET_OFFLINE=1" \
# vllm/vllm-openai:latest \
# --tensor-parallel-size 8 \
# --model /mnt/model/DeepSeek-R1-Distill-Llama-70B-Q4_K_M.gguf \
# #--tokenizer deepseek-ai/DeepSeek-R1-Distill-Llama-70B
# #--quantization gguf \
# #--enforce-eager \

# docker build . -t prueva_vllm-openai-base --target vllm-openai-base


# en /etc/docker/daemon.json:
#     {
#       "runtimes": {
#         "nvidia": {
#           "path": "nvidia-container-runtime",
#           "runtimeArgs": []
#         }
#       }
#     }


ARG CUDA_VERSION=12.4.1
ARG PYTHON_VERSION=3.12
ARG torch_cuda_arch_list='5.0 7.0'
ARG pytorch_Version=2.5.1
ARG pytorch_Vision_Version=0.20.1
ARG pytorch_Audio_Version=2.5.1

FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu22.04 AS pytorch_dependencias_instaladas 
ARG torch_cuda_arch_list
ARG pytorch_Version
ARG PYTHON_VERSION

ENV TORCH_CUDA_ARCH_LIST=${torch_cuda_arch_list}
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHON_VERSION=${PYTHON_VERSION}

WORKDIR /

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        ccache \
        cmake \
        curl \
        git \
        libjpeg-dev \
        libpng-dev \
        ninja-build \
        software-properties-common \
        rust-1.80-all

RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update -y
RUN apt-get install -y python${PYTHON_VERSION} python${PYTHON_VERSION}-dev python${PYTHON_VERSION}-venv
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1
RUN update-alternatives --set python3 /usr/bin/python${PYTHON_VERSION}
RUN ln -sf /usr/bin/python${PYTHON_VERSION}-config /usr/bin/python3-config
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python${PYTHON_VERSION}


RUN git clone https://github.com/pytorch/pytorch.git

WORKDIR /pytorch

RUN git checkout v${pytorch_Version}

RUN git submodule update --init --recursive

#RUN pip install cmake
RUN pip install -r requirements.txt

RUN rm -rf /var/lib/apt/lists/*
RUN /usr/sbin/update-ccache-symlinks
RUN mkdir /opt/ccache && ccache --set-config=cache_dir=/opt/ccache

FROM pytorch_dependencias_instaladas AS pytorch_build
ARG torch_cuda_arch_list

ENV TORCH_CUDA_ARCH_LIST=${torch_cuda_arch_list}

RUN make triton

RUN TORCH_NVCC_FLAGS="-Xfatbin -compress-all" \
    python3 setup.py develop

FROM pytorch_build AS vllm-base

ARG PYTHON_VERSION
ARG torch_cuda_arch_list

ENV TORCH_CUDA_ARCH_LIST=${torch_cuda_arch_list}

WORKDIR /vllm-workspace

ENV DEBIAN_FRONTEND=noninteractive

ARG TARGETPLATFORM

RUN PYTHON_VERSION_STR=$(echo ${PYTHON_VERSION} | sed 's/\.//g') && \
    echo "export PYTHON_VERSION_STR=${PYTHON_VERSION_STR}" >> /etc/environment

RUN echo 'tzdata tzdata/Areas select America' | debconf-set-selections \
    && echo 'tzdata tzdata/Zones/America select Los_Angeles' | debconf-set-selections \
    && apt-get update -y \
    && apt-get install -y ccache software-properties-common git curl wget sudo vim python3-pip \
    && apt-get install -y ffmpeg libsm6 libxext6 libgl1 \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update -y \
    && apt-get install -y python${PYTHON_VERSION} python${PYTHON_VERSION}-dev python${PYTHON_VERSION}-venv libibverbs-dev \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1 \
    && update-alternatives --set python3 /usr/bin/python${PYTHON_VERSION} \
    && ln -sf /usr/bin/python${PYTHON_VERSION}-config /usr/bin/python3-config \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python${PYTHON_VERSION} \
    && python3 --version && python3 -m pip --version

RUN ldconfig /usr/local/cuda-$(echo $CUDA_VERSION | cut -d. -f1,2)/compat/

RUN --mount=type=cache,target=/root/.cache/pip

RUN --mount=type=bind,from=build,src=/workspace/dist,target=/vllm-workspace/dist \
    --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install dist/*.whl --verbose

RUN --mount=type=cache,target=/root/.cache/pip \
. /etc/environment && \
python3 -m pip install https://github.com/flashinfer-ai/flashinfer/releases/download/v0.2.1.post1/flashinfer_python-0.2.1.post1+cu124torch2.5-cp38-abi3-linux_x86_64.whl ;
COPY examples examples

COPY requirements-build.txt requirements-build.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install -r requirements-build.txt


FROM vllm-base AS vllm-openai-base

# install additional dependencies for openai api server
RUN --mount=type=cache,target=/root/.cache/pip \
    if [ "$TARGETPLATFORM" = "linux/arm64" ]; then \
        pip install accelerate hf_transfer 'modelscope!=1.15.0' 'bitsandbytes>=0.42.0' 'timm==0.9.10' boto3 runai-model-streamer runai-model-streamer[s3]; \
    else \
        pip install accelerate hf_transfer 'modelscope!=1.15.0' 'bitsandbytes>=0.45.0' 'timm==0.9.10' boto3 runai-model-streamer runai-model-streamer[s3]; \
    fi

ENV VLLM_USAGE_SOURCE production-docker-image

FROM vllm-openai-base AS vllm-openai

ENTRYPOINT ["python3", "-m", "vllm.entrypoints.openai.api_server"]