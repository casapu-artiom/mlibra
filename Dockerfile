ARG BASE_IMAGE=pytorch/pytorch:2.11.0-cuda12.8-cudnn9-devel

FROM ${BASE_IMAGE} AS runtime

ARG PYTHON_VERSION=3.12
ARG APP_USER=appuser
ARG APP_UID=1000
ARG APP_GID=100
ARG DEBIAN_FRONTEND=noninteractive
# faiss is compiled from source (step below). 120 = Blackwell (sm_120).
ARG FAISS_VERSION=v1.14.1
ARG CUDA_ARCHS="80;86;89;90;120"

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

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --break-system-packages --no-build-isolation -r /tmp/requirements.txt

RUN apt-get update && apt-get install -y --no-install-recommends \
        git build-essential cmake ninja-build swig \
        libopenblas-dev libomp-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir numpy

# ---- 4b. Build faiss (GPU) from source, once, at image build time ----
# Moved out of entrypoint.sh so it is baked into the image instead of compiled
# on every container boot. Mirrors the previous entrypoint build exactly.
RUN git clone --depth 1 --branch ${FAISS_VERSION} \
        https://github.com/facebookresearch/faiss.git /opt/faiss \
    && cmake -B /opt/faiss/build -S /opt/faiss \
        -DCMAKE_BUILD_TYPE=Release \
        -DFAISS_ENABLE_GPU=ON \
        -DFAISS_ENABLE_PYTHON=ON \
        -DBUILD_SHARED_LIBS=ON \
        -DFAISS_OPT_LEVEL=avx2 \
        -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHS}" \
        -DPython_EXECUTABLE=$(which python) \
    && make -C /opt/faiss/build -j"$(nproc)" faiss faiss_avx2 swigfaiss swigfaiss_avx2 \
    && make -C /opt/faiss/build install \
    && (cd /opt/faiss/build/faiss/python && python setup.py install) \
    # `libfaiss_python_callbacks.so` is built for the python bindings but is NOT
    # installed by `make install`; _swigfaiss.so links it via an RPATH into the
    # build tree, so it must be copied somewhere on the loader path before the
    # build tree is deleted (otherwise: "cannot open shared object file").
    && find /opt/faiss/build -name 'libfaiss_python_callbacks*.so' \
        -exec cp {} /usr/local/lib/ \; \
    && ldconfig \
    && rm -rf /opt/faiss/build

    # PETSc/SLEPc build needs gfortran + an MPI (the base has gcc/g++/make/cmake/curl
# via build-essential, but neither a Fortran compiler nor MPI). openmpi-bin also
# provides the `mpirun` the eigensolver entrypoint shells out to at runtime.
RUN apt-get update && apt-get install -y --no-install-recommends \
        gfortran openmpi-bin libopenmpi-dev \
    && rm -rf /var/lib/apt/lists/*

# Build PETSc + SLEPc + MUMPS from source and install petsc4py/slepc4py into the
# inherited /opt/venv (on PATH from the base image). WITH_MUMPS=1 is what makes
# the 64-bit parallel direct solver available for the shift-invert path.
# PREFIX=/opt installs the libs to /opt/petsc and /opt/slepc.
COPY slepc/build_slepc_petsc.sh /tmp/build_slepc_petsc.sh
RUN PREFIX=/opt WITH_MUMPS=1 bash /tmp/build_slepc_petsc.sh \
    && rm -f /tmp/build_slepc_petsc.sh

# Make the prefix installs discoverable at runtime: PETSC_DIR/SLEPC_DIR for the
# python bindings, LD_LIBRARY_PATH so the loader finds libpetsc/libslepc (+ the
# bundled MUMPS/ScaLAPACK/metis shared objects).
ENV PETSC_DIR=/opt/petsc
ENV SLEPC_DIR=/opt/slepc
ENV LD_LIBRARY_PATH=/opt/petsc/lib:/opt/slepc/lib:${LD_LIBRARY_PATH}

# ---- 5. Entrypoint setup ----
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

RUN chown -R ${APP_USER}:${APP_GID} /opt/venv /home/${APP_USER}
WORKDIR /home/${APP_USER}

EXPOSE 2222
ENTRYPOINT ["/entrypoint.sh"]