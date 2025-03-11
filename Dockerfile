#TODO torchvision==0.20.1 torchaudio==2.5.1

ARG CUDA_VERSION=12.4.1
ARG PYTHON_VERSION=3.12
ARG torch_cuda_arch_list='5.0 7.0'
ARG vllm_fa_cmake_gpu_arches='50-real;70-real;90-real'
ARG pytorch_Version=2.5.1
ARG pytorch_Vision_Version=0.20.1
ARG pytorch_Audio_Version=2.5.1
ARG max_jobs=24
ARG nvcc_threads=24


FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu20.04 AS base
ARG CUDA_VERSION
ARG PYTHON_VERSION
ARG torch_cuda_arch_list

ENV CUDA_VERSION=${CUDA_VERSION}
ENV PYTHON_VERSION=${PYTHON_VERSION}
ENV TORCH_CUDA_ARCH_LIST=${torch_cuda_arch_list}

ENV DEBIAN_FRONTEND=noninteractive

# Install Python and other dependencies
RUN echo 'tzdata tzdata/Areas select America' | debconf-set-selections \
    && echo 'tzdata tzdata/Zones/America select Los_Angeles' | debconf-set-selections \
    && apt-get update -y \
    && apt-get install -y ccache software-properties-common git curl sudo \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update -y \
    && apt-get install -y python${PYTHON_VERSION} python${PYTHON_VERSION}-dev python${PYTHON_VERSION}-venv \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1 \
    && update-alternatives --set python3 /usr/bin/python${PYTHON_VERSION} \
    && ln -sf /usr/bin/python${PYTHON_VERSION}-config /usr/bin/python3-config \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python${PYTHON_VERSION} \
    && python3 --version && python3 -m pip --version

# Upgrade to GCC 10 to avoid https://gcc.gnu.org/bugzilla/show_bug.cgi?id=92519
# as it was causing spam when compiling the CUTLASS kernels
RUN apt-get install -y gcc-10 g++-10
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-10 110 --slave /usr/bin/g++ g++ /usr/bin/g++-10
RUN gcc --version

# Workaround for https://github.com/openai/triton/issues/2507 and
# https://github.com/pytorch/pytorch/issues/107960 -- hopefully
# this won't be needed for future versions of this docker image
# or future versions of triton.
RUN ldconfig /usr/local/cuda-$(echo $CUDA_VERSION | cut -d. -f1,2)/compat/

WORKDIR /workspace

# install build and runtime dependencies

# arm64 (GH200) build follows the practice of "use existing pytorch" build,
# we need to install torch and torchvision from the nightly builds first,
# pytorch will not appear as a vLLM dependency in all of the following steps
# after this step
RUN --mount=type=cache,target=/root/.cache/pip

COPY requirements-common.txt requirements-common.txt
COPY requirements-cuda.txt requirements-cuda.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install -r requirements-cuda.txt


ENV TORCH_CUDA_ARCH_LIST=${torch_cuda_arch_list}
ENV VLLM_FA_CMAKE_GPU_ARCHES=${vllm_fa_cmake_gpu_arches}

FROM base AS base-torchvision-torchaudio

ARG pytorch_Vision_Version
ARG pytorch_Audio_Version

ENV pytorch_Vision_Version=${pytorch_Vision_Version}
ENV pytorch_Audio_Version=${pytorch_Audio_Version}

RUN python3 -m pip install torchvision==${pytorch_Vision_Version} torchaudio==${pytorch_Audio_Version}

FROM base-torchvision-torchaudio AS base-pytorch

ARG pytorch_Version

ENV pytorch_Version=${pytorch_Version}

WORKDIR /

RUN apt-get update && apt-get install -y --no-install-recommends \
#    cmake \
    ninja-build \
    rust-1.80-all

RUN python3 -m pip install cmake

RUN git clone https://github.com/pytorch/pytorch.git

WORKDIR /pytorch

RUN git checkout v${pytorch_Version}
RUN git submodule update --init --recursive

RUN pip install -r requirements.txt
RUN pip install mkl-static mkl-include

FROM base-pytorch AS base-build-pytorch
ARG torch_cuda_arch_list

ENV TORCH_CUDA_ARCH_LIST=${torch_cuda_arch_list}

WORKDIR /pytorch

RUN make triton
RUN python3 setup.py develop

FROM base-build-pytorch AS base-pytorch-adapter

WORKDIR /workspace


FROM base-pytorch-adapter AS build
ARG max_jobs
ARG nvcc_threads

ENV MAX_JOBS=${max_jobs}
ENV NVCC_THREADS=${nvcc_threads}

# install build dependencies
COPY requirements-build.txt requirements-build.txt

RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install -r requirements-build.txt

COPY . .
ARG GIT_REPO_CHECK=0
RUN --mount=type=bind,source=.git,target=.git \
    if [ "$GIT_REPO_CHECK" != 0 ]; then bash tools/check_repo.sh ; fi

ARG USE_SCCACHE
ARG SCCACHE_BUCKET_NAME=vllm-build-sccache
ARG SCCACHE_REGION_NAME=us-west-2
ARG SCCACHE_S3_NO_CREDENTIALS=0
# if USE_SCCACHE is set, use sccache to speed up compilation
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=bind,source=.git,target=.git \
    if [ "$USE_SCCACHE" = "1" ]; then \
        echo "Installing sccache..." \
        && curl -L -o sccache.tar.gz https://github.com/mozilla/sccache/releases/download/v0.8.1/sccache-v0.8.1-x86_64-unknown-linux-musl.tar.gz \
        && tar -xzf sccache.tar.gz \
        && sudo mv sccache-v0.8.1-x86_64-unknown-linux-musl/sccache /usr/bin/sccache \
        && rm -rf sccache.tar.gz sccache-v0.8.1-x86_64-unknown-linux-musl \
        && export SCCACHE_BUCKET=${SCCACHE_BUCKET_NAME} \
        && export SCCACHE_REGION=${SCCACHE_REGION_NAME} \
        && export SCCACHE_S3_NO_CREDENTIALS=${SCCACHE_S3_NO_CREDENTIALS} \
        && export SCCACHE_IDLE_TIMEOUT=0 \
        && export CMAKE_BUILD_TYPE=Release \
        && sccache --show-stats \
        && python3 setup.py bdist_wheel --dist-dir=dist --py-limited-api=cp38 \
        && sccache --show-stats; \
    fi

ENV CCACHE_DIR=/root/.cache/ccache
RUN --mount=type=cache,target=/root/.cache/ccache \
    --mount=type=cache,target=/root/.cache/pip \
    --mount=type=bind,source=.git,target=.git  \
    if [ "$USE_SCCACHE" != "1" ]; then \
        python3 setup.py bdist_wheel --dist-dir=dist --py-limited-api=cp38; \
    fi

# Check the size of the wheel if RUN_WHEEL_CHECK is true
COPY .buildkite/check-wheel-size.py check-wheel-size.py
# sync the default value with .buildkite/check-wheel-size.py
ARG VLLM_MAX_SIZE_MB=400
ENV VLLM_MAX_SIZE_MB=$VLLM_MAX_SIZE_MB
ARG RUN_WHEEL_CHECK=false
RUN if [ "$RUN_WHEEL_CHECK" = "true" ]; then \
        python3 check-wheel-size.py dist; \
    else \
        echo "Skipping wheel size check."; \
    fi

# image with vLLM installed
# TODO: Restore to base image after FlashInfer AOT wheel fixed
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu22.04 AS vllm-base
ARG CUDA_VERSION
ARG PYTHON_VERSION

ENV CUDA_VERSION=${CUDA_VERSION}
ENV PYTHON_VERSION=${PYTHON_VERSION}

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /vllm-workspace


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
RUN --mount=type=cache,target=/root/.cache/pip

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
. /etc/environment
COPY examples examples

# Although we build Flashinfer with AOT mode, there's still
# some issues w.r.t. JIT compilation. Therefore we need to
# install build dependencies for JIT compilation.
# TODO: Remove this once FlashInfer AOT wheel is fixed
COPY requirements-build.txt requirements-build.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install -r requirements-build.txt

FROM vllm-base AS vllm-base-torchvision-torchaudio

ARG pytorch_Vision_Version
ARG pytorch_Audio_Version

ENV pytorch_Vision_Version=${pytorch_Vision_Version}
ENV pytorch_Audio_Version=${pytorch_Audio_Version}

RUN python3 -m pip install torchvision==${pytorch_Vision_Version} torchaudio==${pytorch_Audio_Version}


FROM vllm-base-torchvision-torchaudio AS vllm-base-pytorch

ARG pytorch_Version

ENV pytorch_Version=${pytorch_Version}

WORKDIR /

RUN apt-get update && apt-get install -y --no-install-recommends \
#    cmake \
    ninja-build \
    rust-1.80-all

RUN python3 -m pip install cmake

RUN git clone https://github.com/pytorch/pytorch.git

WORKDIR /pytorch

RUN git checkout v${pytorch_Version}
RUN git submodule update --init --recursive

RUN pip install -r requirements.txt
RUN pip install mkl-static mkl-include

FROM vllm-base-pytorch AS vllm-base-build-pytorch
ARG torch_cuda_arch_list

ENV TORCH_CUDA_ARCH_LIST=${torch_cuda_arch_list}

WORKDIR /pytorch

RUN make triton
RUN python3 setup.py develop

FROM vllm-base-build-pytorch AS vllm-base-pytorch-adapter

WORKDIR /workspace



# base openai image with additional requirements, for any subsequent openai-style images
FROM vllm-base-pytorch-adapter AS vllm-openai-base

ARG torch_cuda_arch_list

ENV TORCH_CUDA_ARCH_LIST=${torch_cuda_arch_list}

# install additional dependencies for openai api server
RUN --mount=type=cache,target=/root/.cache/pip \
        pip install accelerate hf_transfer 'modelscope!=1.15.0' 'bitsandbytes>=0.45.0' 'timm==0.9.10' boto3 runai-model-streamer runai-model-streamer[s3]

ENV VLLM_USAGE_SOURCE production-docker-image


FROM vllm-openai-base AS vllm-openai

ENTRYPOINT ["python3", "-m", "vllm.entrypoints.openai.api_server"]
#################### OPENAI API SERVER ####################