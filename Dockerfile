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

# docker build . -t prueva_vllm-openai --target vllm-openai


# en /etc/docker/daemon.json:
#     {
#       "runtimes": {
#         "nvidia": {
#           "path": "nvidia-container-runtime",
#           "runtimeArgs": []
#         }
#       }
#     }











# ARG CUDA_VERSION=12.4.1
# ARG PYTHON_VERSION=3.12
# ARG torch_cuda_arch_list='5.0 7.0'
# ARG pytorch_Version=2.5.1
# ARG pytorch_Vision_Version=0.20.1
# ARG pytorch_Audio_Version=2.5.1

# FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu22.04 AS pytorch_dependencias_instaladas 
# ARG torch_cuda_arch_list
# ARG pytorch_Version
# ARG PYTHON_VERSION

# ENV TORCH_CUDA_ARCH_LIST=${torch_cuda_arch_list}
# ENV DEBIAN_FRONTEND=noninteractive
# ENV PYTHON_VERSION=${PYTHON_VERSION}

# WORKDIR /

# RUN apt-get update && apt-get install -y --no-install-recommends \
#         build-essential \
#         ca-certificates \
#         ccache \
#         cmake \
#         curl \
#         git \
#         libjpeg-dev \
#         libpng-dev \
#         ninja-build \
#         software-properties-common \
#         rust-1.80-all

# RUN add-apt-repository ppa:deadsnakes/ppa
# RUN apt-get update -y
# RUN apt-get install -y python${PYTHON_VERSION} python${PYTHON_VERSION}-dev python${PYTHON_VERSION}-venv
# RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1
# RUN update-alternatives --set python3 /usr/bin/python${PYTHON_VERSION}
# RUN ln -sf /usr/bin/python${PYTHON_VERSION}-config /usr/bin/python3-config
# RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python${PYTHON_VERSION}


# RUN git clone https://github.com/pytorch/pytorch.git

# WORKDIR /pytorch

# RUN git checkout v${pytorch_Version}

# RUN git submodule update --init --recursive

# RUN pip install -r requirements.txt

# RUN rm -rf /var/lib/apt/lists/*
# RUN /usr/sbin/update-ccache-symlinks
# RUN mkdir /opt/ccache && ccache --set-config=cache_dir=/opt/ccache

# FROM pytorch_dependencias_instaladas AS pytorch_build
# ARG torch_cuda_arch_list

# ENV TORCH_CUDA_ARCH_LIST=${torch_cuda_arch_list}

# RUN make triton

# RUN TORCH_NVCC_FLAGS="-Xfatbin -compress-all" \
#     python3 setup.py develop

# FROM pytorch_build AS vllm-base

# ARG PYTHON_VERSION
# ARG torch_cuda_arch_list
# ARG PYTHON_VERSION

# ENV TORCH_CUDA_ARCH_LIST=${torch_cuda_arch_list}

# WORKDIR /vllm-workspace

# ENV DEBIAN_FRONTEND=noninteractive

# RUN PYTHON_VERSION_STR=$(echo ${PYTHON_VERSION} | sed 's/\.//g') && \
#     echo "export PYTHON_VERSION_STR=${PYTHON_VERSION_STR}" >> /etc/environment

# RUN echo 'tzdata tzdata/Areas select America' | debconf-set-selections
# RUN echo 'tzdata tzdata/Zones/America select Los_Angeles' | debconf-set-selections
# RUN apt-get update -y
# RUN apt-get install -y ccache software-properties-common git curl wget sudo vim python3-pip
# RUN apt-get install -y ffmpeg libsm6 libxext6 libgl1
# RUN apt-get install -y software-properties-common python3-apt
# #RUN add-apt-repository ppa:deadsnakes/ppa
# RUN apt-get update -y
# RUN apt-get install -y python${PYTHON_VERSION} python${PYTHON_VERSION}-dev python${PYTHON_VERSION}-venv libibverbs-dev
# RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1
# RUN update-alternatives --set python3 /usr/bin/python${PYTHON_VERSION}
# RUN ln -sf /usr/bin/python${PYTHON_VERSION}-config /usr/bin/python3-config
# RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python${PYTHON_VERSION}
# RUN python3 --version && python3 -m pip --version

# RUN ldconfig /usr/local/cuda-$(echo $CUDA_VERSION | cut -d. -f1,2)/compat/

# RUN --mount=type=bind,from=build,src=/workspace/dist,target=/vllm-workspace/dist \
#     python3 -m pip install dist/*.whl --verbose

# RUN . /etc/environment && \
# python3 -m pip install https://github.com/flashinfer-ai/flashinfer/releases/download/v0.2.1.post1/flashinfer_python-0.2.1.post1+cu124torch2.5-cp38-abi3-linux_x86_64.whl ;
# COPY examples examples

# COPY requirements-build.txt requirements-build.txt
# RUN python3 -m pip install -r requirements-build.txt


# FROM vllm-base AS vllm-openai-base

# # install additional dependencies for openai api server
# RUN pip install accelerate hf_transfer 'modelscope!=1.15.0' 'bitsandbytes>=0.45.0' 'timm==0.9.10' boto3 runai-model-streamer runai-model-streamer[s3];

# ENV VLLM_USAGE_SOURCE production-docker-image

# FROM vllm-openai-base AS vllm-openai

# ENTRYPOINT ["python3", "-m", "vllm.entrypoints.openai.api_server"]








































ARG CUDA_VERSION=12.4.1
ARG PYTHON_VERSION=3.12
ARG torch_cuda_arch_list='5.0 7.0'
ARG pytorch_Version=2.5.1
ARG pytorch_Vision_Version=0.20.1
ARG pytorch_Audio_Version=2.5.1






# image with vLLM installed
# TODO: Restore to base image after FlashInfer AOT wheel fixed
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu22.04 AS vllm-base
ARG CUDA_VERSION
ARG PYTHON_VERSION
WORKDIR /vllm-workspace
ENV DEBIAN_FRONTEND=noninteractive
ARG TARGETPLATFORM

RUN PYTHON_VERSION_STR=$(echo ${PYTHON_VERSION} | sed 's/\.//g') && \
    echo "export PYTHON_VERSION_STR=${PYTHON_VERSION_STR}" >> /etc/environment

# Install Python and other dependencies
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

# Workaround for https://github.com/openai/triton/issues/2507 and
# https://github.com/pytorch/pytorch/issues/107960 -- hopefully
# this won't be needed for future versions of this docker image
# or future versions of triton.
RUN ldconfig /usr/local/cuda-$(echo $CUDA_VERSION | cut -d. -f1,2)/compat/

# arm64 (GH200) build follows the practice of "use existing pytorch" build,
# we need to install torch and torchvision from the nightly builds first,
# pytorch will not appear as a vLLM dependency in all of the following steps
# after this step
RUN --mount=type=cache,target=/root/.cache/pip \
    if [ "$TARGETPLATFORM" = "linux/arm64" ]; then \
        python3 -m pip install --index-url https://download.pytorch.org/whl/nightly/cu124 "torch==2.6.0.dev20241210+cu124" "torchvision==0.22.0.dev20241215";  \
    fi

# Install vllm wheel first, so that torch etc will be installed.
RUN --mount=type=bind,from=build,src=/workspace/dist,target=/vllm-workspace/dist \
    --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install dist/*.whl --verbose

# If we need to build FlashInfer wheel before its release:
# $ export FLASHINFER_ENABLE_AOT=1
# $ # Note we remove 7.0 from the arch list compared to the list below, since FlashInfer only supports sm75+
# $ export TORCH_CUDA_ARCH_LIST='7.5 8.0 8.6 8.9 9.0+PTX'
# $ git clone https://github.com/flashinfer-ai/flashinfer.git --recursive
# $ cd flashinfer
# $ git checkout 524304395bd1d8cd7d07db083859523fcaa246a4
# $ rm -rf build
# $ python3 setup.py bdist_wheel --dist-dir=dist --verbose
# $ ls dist
# $ # upload the wheel to a public location, e.g. https://wheels.vllm.ai/flashinfer/524304395bd1d8cd7d07db083859523fcaa246a4/flashinfer_python-0.2.1.post1+cu124torch2.5-cp38-abi3-linux_x86_64.whl

RUN --mount=type=cache,target=/root/.cache/pip \
. /etc/environment && \
if [ "$TARGETPLATFORM" != "linux/arm64" ]; then \
    python3 -m pip install https://github.com/flashinfer-ai/flashinfer/releases/download/v0.2.1.post1/flashinfer_python-0.2.1.post1+cu124torch2.5-cp38-abi3-linux_x86_64.whl ; \
fi
COPY examples examples

# Although we build Flashinfer with AOT mode, there's still
# some issues w.r.t. JIT compilation. Therefore we need to
# install build dependencies for JIT compilation.
# TODO: Remove this once FlashInfer AOT wheel is fixed
COPY requirements-build.txt requirements-build.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install -r requirements-build.txt


# base openai image with additional requirements, for any subsequent openai-style images
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
#################### OPENAI API SERVER ####################