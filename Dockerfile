ARG BASE_IMAGE=pytorch/pytorch:2.11.0-cuda12.8-cudnn9-devel

FROM ${BASE_IMAGE} AS runtime

ARG PYTHON_VERSION=3.12
ARG APP_USER=appuser
ARG APP_UID=1000
ARG APP_GID=100
ARG DEBIAN_FRONTEND=noninteractive

ENV APP_USER=${APP_USER}
ENV APP_GID=${APP_GID}

USER root

# ---- 1. System packages (Simplified) ----
RUN apt-get update && apt-get install -y --no-install-recommends \
    openssh-server gosu curl wget git vim htop rsync build-essential \
    libopenblas-dev software-properties-common libgflags-dev \
    && rm -rf /var/lib/apt/lists/*

RUN add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        python${PYTHON_VERSION} \
        python${PYTHON_VERSION}-venv \
        python${PYTHON_VERSION}-dev \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv --system-site-packages /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN set -eux; \
    # Make sure target group exists (it should, on Debian/Ubuntu, but be safe)
    if ! getent group ${APP_GID} >/dev/null; then \
        groupadd -g ${APP_GID} ${APP_USER}; \
    fi; \
    # Rename the existing UID 1000 user, or create one if it doesn't exist
    if getent passwd ${APP_UID} >/dev/null; then \
        existing=$(getent passwd ${APP_UID} | cut -d: -f1); \
        if [ "$existing" != "${APP_USER}" ]; then \
            usermod -l ${APP_USER} -d /home/${APP_USER} -m -g ${APP_GID} "$existing"; \
        else \
            usermod -g ${APP_GID} ${APP_USER}; \
        fi; \
    else \
        useradd -m -u ${APP_UID} -g ${APP_GID} -s /bin/bash ${APP_USER}; \
    fi; \
    echo "${APP_USER} ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers

# ---- 3. SSH Configuration ----
RUN mkdir -p /run/sshd \
    && sed -i 's/#Port 22/Port 2222/' /etc/ssh/sshd_config \
    && echo "AllowUsers ${APP_USER}" >> /etc/ssh/sshd_config

# ---- 4. Install Dependencies Directly ----
# Using --break-system-packages because modern Ubuntu/Debian requires it 
# to install via pip outside of a venv.
COPY requirements.txt /tmp/requirements.txt
RUN pip install torch-scatter -f https://data.pyg.org/whl/torch-2.11.0+cu128.html --break-system-packages
RUN pip install torch-sparse -f https://data.pyg.org/whl/torch-2.11.0+cu128.html --break-system-packages
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --break-system-packages --no-build-isolation -r /tmp/requirements.txt

RUN apt-get update && apt-get install -y --no-install-recommends \
        git build-essential cmake ninja-build swig \
        libopenblas-dev libomp-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir numpy

# ---- 5. Entrypoint setup ----
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

RUN chown -R ${APP_USER}:${APP_GID} /opt/venv /home/${APP_USER}
WORKDIR /home/${APP_USER}

EXPOSE 2222
ENTRYPOINT ["/entrypoint.sh"]