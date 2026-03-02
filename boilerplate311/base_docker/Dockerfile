# Base Image
FROM ubuntu:20.04
# FROM python:3.11-slim

LABEL maintainer="Amazon AI"

# Define build-time variables
ARG PYTHON=python3
ARG PYTHON_PIP=python3-pip
ARG PIP=pip3
ARG PYTHON_VERSION=3.11.11

ENV DEBIAN_FRONTEND=noninteractive

# Install essential libraries and tools
RUN sed -i 's|http://archive.ubuntu.com/ubuntu/|http://ftp.kaist.ac.kr/ubuntu/|g' /etc/apt/sources.list \
        && apt-get update -o Acquire::ForceIPv4=true \
        && apt-get install -y --no-install-recommends \
        software-properties-common \
        build-essential \
        ca-certificates \
        curl \
        wget \
        git \
        libopencv-dev \
        openssh-client \
        openssh-server \
        vim \
        libffi-dev \
        libssl-dev \
        sqlite3 \
        libsqlite3-dev \
        libreadline-dev \
        libncursesw5-dev \
        tk-dev \
        libgdbm-dev \
        libc6-dev \
        libbz2-dev \
        zlib1g-dev && \
    add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Python 3.11.11 from source (PGO + LTO 최적화)
RUN wget https://www.python.org/ftp/python/$PYTHON_VERSION/Python-$PYTHON_VERSION.tgz && \
    tar -xvf Python-$PYTHON_VERSION.tgz && cd Python-$PYTHON_VERSION && \
    ./configure \
        --enable-optimizations \
        --with-lto \
        --enable-loadable-sqlite-extensions && \
    make -j"$(nproc)" && make install && \
    cd .. && rm -rf Python-$PYTHON_VERSION* && \
    ln -s /usr/local/bin/pip3 /usr/bin/pip && \
    ln -s /usr/local/bin/python3 /usr/bin/python && \
    rm -rf /root/.cache

# Install fonts
RUN apt-get update && \
    apt-get install -y --no-install-recommends fontconfig && \
    wget https://hangeul.pstatic.net/hangeul_static/webfont/NanumGothic/NanumGothic.ttf && \
    mkdir -p /usr/share/fonts/nanum/ && \
    mv NanumGothic.ttf /usr/share/fonts/nanum/ && \
    fc-cache -fv && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# uv 설치 (pip 대비 10~100x 빠른 패키지 설치)
RUN pip install --no-cache-dir --upgrade pip setuptools && \
    pip install --no-cache-dir uv

# Copy and install dependencies
WORKDIR /
COPY requirements.txt /requirements.txt

# uv pip install: 빠른 resolver + 병렬 다운로드
# --no-cache: 디스크 절약, --python: 시스템 Python 명시 지정
RUN uv pip install --no-cache --system --python python3 -r /requirements.txt \
    && rm -f /requirements.txt

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/lib:/var/lang/lib:/lib64:/usr/lib64:/var/runtime:/var/runtime/lib:/var/task:/var/task/lib:/opt/lib" \
    PYTHONIOENCODING=UTF-8 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8