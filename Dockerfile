# Build arguments 
# debian:buster-slim is much smaller than ubuntu 22
ARG BASE_IMAGE=debian:bullseye-slim

FROM ${BASE_IMAGE} AS base

ARG POETRY_VERSION=1.8.2
ARG PYTHON_VERSION=3.11.6
ARG POETRY_HOME="/opt/poetry"
ARG ROOT_DIR="/root"
ARG FLATNAV_PATH="${ROOT_DIR}/flatnavlib"
ARG INCLUDE_HNSWLIB=false

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update -y \
    && apt-get install -y --no-install-recommends \
        # Need for python installation: 
        # https://github.com/pyenv/pyenv/wiki#suggested-build-environment
        make \
        build-essential \
        ca-certificates \
        libssl-dev \
        zlib1g-dev \
        libbz2-dev \
        libreadline-dev \
        libsqlite3-dev \
        wget \
        curl \
        llvm \
        libncursesw5-dev \
        xz-utils \
        tk-dev \
        libxml2-dev \
        libxmlsec1-dev \
        libffi-dev \
        liblzma-dev \
        # Install the rest
        git \
        gcc \
        g++ \
        apt-utils \
        ninja-build \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /tmp/*

# Install perf separately (kernel-version specific)
# Debian packages perf as linux-perf-<version>, find and symlink it
RUN apt-get update && \
    KERNEL_VERSION=$(uname -r) && \
    apt-get install -y linux-perf-${KERNEL_VERSION} || \
    apt-get install -y linux-perf || \
    echo "Warning: Could not install perf" && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    # Create symlink from versioned perf to /usr/bin/perf
    PERF_BIN=$(find /usr/bin -name 'perf_*' -type f 2>/dev/null | head -1) && \
    if [ -n "$PERF_BIN" ]; then \
        ln -sf "$PERF_BIN" /usr/bin/perf; \
        echo "Installed perf: $(perf --version 2>&1 || echo 'perf binary found')"; \
    fi

# Install CMake 3.21+ (required by python-bindings)
RUN wget -q https://github.com/Kitware/CMake/releases/download/v3.27.7/cmake-3.27.7-linux-x86_64.sh \
    && mkdir -p /opt/cmake \
    && sh cmake-3.27.7-linux-x86_64.sh --skip-license --prefix=/opt/cmake \
    && rm cmake-3.27.7-linux-x86_64.sh
ENV PATH="/opt/cmake/bin:${PATH}"

# Install python 
# We use pyenv to manage python versions 
ENV PYENV_ROOT=${ROOT_DIR}/.pyenv

# Shims are small proxy executables that intercept calls to Python commands. 
# Putting $PYENV_ROOT/shims at the beginning of PATH ensures that the shimmed 
# Python commands are found and used before any system-wide Python installations.
ENV PATH=${PYENV_ROOT}/shims:${PYENV_ROOT}/bin:${PATH}

ENV PYTHON_VERSION=${PYTHON_VERSION}


RUN set -ex \
    && curl -L https://pyenv.run | /bin/sh \
    && pyenv update \
    && pyenv install $PYTHON_VERSION \
    && pyenv global $PYTHON_VERSION \
    && pyenv rehash 

# PYTHONDONTWRITEBYTECODE: Prevents Python from writing pyc files to disc
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    POETRY_HOME=${POETRY_HOME} \
    POETRY_VERSION=${POETRY_VERSION} 

# Install poetry
RUN curl -sSL https://install.python-poetry.org | python3 - 

# Add poetry to PATH
ENV PATH="${POETRY_HOME}/bin:${PATH}"

WORKDIR ${FLATNAV_PATH}

# Copy source code
COPY include/ ./include/
COPY python-bindings/ ./python-bindings/
COPY experiments/ ./experiments/
COPY README.md ./README.md

# Copy external dependencies (for now only cereal)
COPY external/ ./external/

# Build flatnav wheel from source (with scikit-build for CMake integration)
WORKDIR ${FLATNAV_PATH}/python-bindings
RUN pip install --upgrade pip wheel setuptools scikit-build && \
    rm -rf build dist *.egg-info && \
    python setup.py bdist_wheel

ENV FLATNAV_WHEEL=${FLATNAV_PATH}/python-bindings/dist/*.whl

# Install needed dependencies including flatnav. 
# Install hnwlib (from a forked repo that has extensions we need)
WORKDIR ${FLATNAV_PATH}
# Install hnswlib (from a forked repo that has extensions we need)
RUN if [ "$INCLUDE_HNSWLIB" = true ] ; then \
        git clone https://github.com/BlaiseMuhirwa/hnswlib-original.git \
        && cd hnswlib-original/python_bindings \
        && poetry install --no-root \
        && poetry run python setup.py bdist_wheel; \
    fi 

# Get the wheel as an environment variable 
ENV HNSWLIB_WHEEL=${FLATNAV_PATH}/hnswlib-original/python_bindings/dist/*.whl

# Add flatnav, faiss-cpu, and hnswlib to the experiment runner 
WORKDIR ${FLATNAV_PATH}/experiments
RUN poetry add ${FLATNAV_WHEEL} faiss-cpu ${HNSWLIB_WHEEL} && poetry install --no-root