ARG BASE_IMAGE=pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel
FROM ${BASE_IMAGE} AS runtime

ARG APP_USER=appuser
ARG APP_UID=1000
ARG APP_GID=1000
ARG DEBIAN_FRONTEND=noninteractive

ENV APP_USER=${APP_USER}

# ---- 1. System packages (Simplified) ----
RUN apt-get update && apt-get install -y --no-install-recommends \
    openssh-server gosu curl wget git vim htop rsync build-essential \
    && rm -rf /var/lib/apt/lists/*

# ---- 2. Non-root user setup ----
RUN groupadd -g ${APP_GID} ${APP_USER} \
    && useradd -m -u ${APP_UID} -g ${APP_GID} -s /bin/bash ${APP_USER} \
    && echo "${APP_USER} ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers

# ---- 3. SSH Configuration ----
RUN mkdir -p /run/sshd \
    && sed -i 's/#Port 22/Port 2222/' /etc/ssh/sshd_config \
    && echo "AllowUsers ${APP_USER}" >> /etc/ssh/sshd_config

# ---- 4. Install Dependencies Directly ----
# Using --break-system-packages because modern Ubuntu/Debian requires it 
# to install via pip outside of a venv.
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --break-system-packages -r /tmp/requirements.txt

# ---- 5. Entrypoint setup ----
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 2222
ENTRYPOINT ["/entrypoint.sh"]