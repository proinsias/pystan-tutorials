FROM python:3.9.5

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# hadolint ignore=DL3008
RUN \
    DEBIAN_FRONTEND="noninteractive" && \
    export DEBIAN_FRONTEND && \
    apt-get update --yes && \
    apt-get install --auto-remove --no-install-recommends --yes \
        ccache  `# Speed up gcc/g++ functionality.` \
    && \
    /usr/sbin/update-ccache-symlinks && \
    apt-get clean && \
    rm --recursive --force \
        /var/cache/* \
        /var/lib/apt/lists/* \
        /var/log/*

ENV APP_DIR="/root/pystan" \
    PATH="/usr/lib/ccache${PATH:+:${PATH}}" \
    TZ="UTC"

RUN mkdir --parents "${APP_DIR}"/

RUN ccache --version

# hadolint ignore=DL3008
RUN \
    DEBIAN_FRONTEND="noninteractive" && \
    export DEBIAN_FRONTEND && \
    apt-get update --yes && \
    apt-get install --auto-remove --no-install-recommends --yes \
        build-essential \
        curl  `# Tool to transfer data using various protocols` \
        git  `# Version control system` \
        pkg-config \
        python3.7 \
        python3.7-dev \
        python3-pip \
        python3-setuptools \
        wget \
      && \
    apt-get clean && \
    rm --recursive --force \
        /var/cache/* \
        /var/lib/apt/lists/* \
        /var/log/*

RUN \
    curl --version && \
    git --version && \
    python3.7 --version

# hadolint ignore=DL3008
RUN \
    DEBIAN_FRONTEND="noninteractive" && \
    export DEBIAN_FRONTEND && \
    apt-get update && \
    apt-get install --auto-remove --no-install-recommends --yes \
        shellcheck `# Shell script linter` \
    && \
    `# Upgrade gcc to gcc-8 for fbprophet.` \
    apt-get remove --yes gcc && \
    apt-get install --auto-remove --no-install-recommends --yes \
        gcc-8 \
        g++-8 \
    && \
    ln --symbolic /usr/bin/gcc-8 /usr/bin/x86_64-linux-gnu-gcc && \
    apt-get clean && \
    rm --recursive --force \
        /var/cache/* \
        /var/lib/apt/lists/* \
        /var/log/* \
    && \
    curl --fail --location --show-error --silent \
        "https://github.com/hadolint/hadolint/releases/download/v1.17.4/hadolint-Linux-x86_64" \
        --output /usr/local/bin/hadolint && \
    chmod +x /usr/local/bin/hadolint

RUN \
    gcc-8 --version && \
    g++-8 --version && \
    hadolint --version && \
    shellcheck --version

### Install main packages ###

# hadolint ignore=DL3005,DL3008,DL3009
RUN \
    DEBIAN_FRONTEND="noninteractive" && \
    export DEBIAN_FRONTEND && \
    apt-get update --yes && \
    apt-get install --auto-remove --no-install-recommends --yes \
        emacs  `# Text editor` \
        sudo \
    && \
    apt-get clean && \
    rm --recursive --force \
        /var/cache/* \
        /var/lib/apt/lists/* \
        /var/log/*

RUN emacs --version

RUN \
    `# https://pyup.io/posts/patched-vulnerability/` \
    curl --fail --location --show-error --silent \
        "https://github.com/pyupio/safety/releases/download/1.8.7/safety-linux-x86_64" \
        --output /usr/local/bin/safety && \
    chmod +x /usr/local/bin/safety

RUN safety --version

### Test script files.

COPY ./bin/* "${APP_DIR}"/bin/

RUN shellcheck "${APP_DIR}"/bin/*

### Python Setup ###

ENV JUPYTER_CONFIG_DIR "${APP_DIR}"/conf

ENV CC=gcc-8 \
    CXX=g++-8 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN python3.7 -m pip install \
        pip==19.3.1  `# Make sure this is compatible with pip-tools in requirements.txt` \
        setuptools==45.1.0 \
        wheel==0.33.6 && \
    rm --recursive --force ~/.cache/pip

RUN python3.7 -m pip --version && \
    wheel version

# Add requirements.txt separately to better use cached builds
# (i.e. editing other files in the dir won't trigger a full pip-reinstall)
COPY ./*requirements.txt "${APP_DIR}"/

# hadolint ignore=DL3013
RUN \
    python3.7 -m pip install --requirement "${APP_DIR}"/requirements.txt && \
    rm --recursive --force ~/.cache/pip

RUN \
    ipython --version && \
    jupyter --version && \
    jupyter contrib nbextension install --symlink && \
    jupyter nbextension enable execute_time/ExecuteTime && \
    jupyter nbextension enable toc2/main && \
    safety check --file "${APP_DIR}"/requirements.txt --full-report

RUN "${APP_DIR}"/bin/docker-entrypoint exec bash --version

# Test Dockerfile.
COPY ./Dockerfile "${APP_DIR}"/

RUN hadolint "${APP_DIR}"/Dockerfile

# SHELL: Set the terminal for Jupyter Notebook.
ENV SHELL=/bin/bash

ENTRYPOINT ["sh", "/root/pystan/bin/docker-entrypoint"]

# HEALTHCHECK --interval=5m --start-period=30s --timeout=10s \
#     CMD curl --fail --silent http://host.docker.internal:8888 || exit 1

# hadolint ignore=DL3000
WORKDIR "${APP_DIR}"
