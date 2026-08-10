#!/usr/bin/env bash
# Build PETSc + SLEPc from source (optionally with MUMPS) and install the
# petsc4py / slepc4py bindings into the CURRENT Python environment.
#
# This is the reliable MUMPS path: the pip `petsc` wheel ships libpetsc.so but
# NOT the MUMPS static libs, so SLEPc can't link (`cannot find -lsmumps`).
# Building from source puts the MUMPS libs on disk where SLEPc finds them.
#
# Local test (no sudo):   PREFIX=$HOME/opt bash manifold/slepc/build_slepc_petsc.sh
# Docker (as root):       PREFIX=/opt      bash manifold/slepc/build_slepc_petsc.sh
# Without MUMPS (fast):   WITH_MUMPS=0     bash manifold/slepc/build_slepc_petsc.sh
#
# Requires: gcc, g++, gfortran, make, cmake, an MPI (mpicc), BLAS/LAPACK, curl.
#
# NOTE on PETSC_DIR/SLEPC_DIR: during each library's OWN ./configure + make,
# its *_DIR must point at the unpacked SOURCE tree (configure uses it as the
# source root). The INSTALL prefix is passed via --prefix and only becomes the
# value of PETSC_DIR/SLEPC_DIR afterwards (for SLEPc's build against PETSc, and
# for the python bindings). We manage this per-section in subshells.

set -euo pipefail

PETSC_VERSION="${PETSC_VERSION:-3.25.2}"
SLEPC_VERSION="${SLEPC_VERSION:-3.25.1}"
PREFIX="${PREFIX:-$HOME/opt}"
WITH_MUMPS="${WITH_MUMPS:-1}"
JOBS="${JOBS:-$(nproc)}"

PETSC_PREFIX="${PREFIX}/petsc"          # install locations (NOT the build *_DIR)
SLEPC_PREFIX="${PREFIX}/slepc"

SCRATCH="$(mktemp -d)"
trap 'rm -rf "$SCRATCH"' EXIT

echo "=== building PETSc ${PETSC_VERSION} + SLEPc ${SLEPC_VERSION} (MUMPS=${WITH_MUMPS})"
echo "    install -> PETSC=${PETSC_PREFIX}  SLEPC=${SLEPC_PREFIX}  jobs=${JOBS}"

# Don't let any inherited values leak into the source builds.
unset PETSC_DIR PETSC_ARCH SLEPC_DIR || true

for t in gcc g++ gfortran make cmake mpicc curl; do
    command -v "$t" >/dev/null || { echo "MISSING tool: $t" >&2; exit 1; }
done

mumps_opts=()
if [ "$WITH_MUMPS" = "1" ]; then
    mumps_opts=(--download-mumps --download-scalapack
                --download-metis --download-parmetis)
fi

# ---- PETSc (PETSC_DIR = source tree during its build) ----
cd "$SCRATCH"
curl -L "https://web.cels.anl.gov/projects/petsc/download/release-snapshots/petsc-${PETSC_VERSION}.tar.gz" | tar xz
(
    cd "petsc-${PETSC_VERSION}"
    src="$PWD"
    export PETSC_DIR="$src"
    export PETSC_ARCH="arch-build"
    ./configure --prefix="${PETSC_PREFIX}" \
        --with-cc=mpicc --with-cxx=mpicxx --with-fc=mpif90 \
        --with-debugging=0 --with-shared-libraries=1 --with-fortran-bindings=0 \
        --with-make-np="${JOBS}" \
        "${mumps_opts[@]}" \
        COPTFLAGS=-O2 CXXOPTFLAGS=-O2 FOPTFLAGS=-O2
    make PETSC_DIR="$src" PETSC_ARCH="arch-build" all
    make PETSC_DIR="$src" PETSC_ARCH="arch-build" install
)

# ---- SLEPc (SLEPC_DIR = source tree; PETSC_DIR = installed PETSc prefix) ----
cd "$SCRATCH"
curl -L "https://slepc.upv.es/download/distrib/slepc-${SLEPC_VERSION}.tar.gz" | tar xz
(
    cd "slepc-${SLEPC_VERSION}"
    src="$PWD"
    export SLEPC_DIR="$src"
    export PETSC_DIR="${PETSC_PREFIX}"
    unset PETSC_ARCH || true            # prefix install -> empty arch
    ./configure --prefix="${SLEPC_PREFIX}"
    make SLEPC_DIR="$src" PETSC_DIR="${PETSC_PREFIX}"
    make SLEPC_DIR="$src" PETSC_DIR="${PETSC_PREFIX}" install
)

# ---- Python bindings, built against the installed prefixes (which have MUMPS) ----
export PETSC_DIR="${PETSC_PREFIX}"
export SLEPC_DIR="${SLEPC_PREFIX}"
unset PETSC_ARCH || true
# Remove pip petsc/slepc wheels so the bindings use PETSC_DIR/SLEPC_DIR.
pip uninstall -y petsc slepc petsc4py slepc4py >/dev/null 2>&1 || true
# Cython must be <3.1: slepc4py 3.25.x .pyx predates 3.1's stricter nogil
# checks ("Accessing Python global ... without gil").
pip install --no-cache-dir "cython<3.1" numpy
# petsc4py and slepc4py MUST be installed in SEPARATE commands: pip builds all
# wheels before installing any, so a combined install compiles slepc4py while
# petsc4py is only built-not-installed -> slepc4py's setup.py can't import it
# (petsc4py.get_include()), and every cimported symbol (CHKERR, PETSC_SUCCESS,
# PyPetscType_Register) is undefined. Install petsc4py first, then slepc4py.
#
# Match petsc4py to slepc4py's patch version (cimport ABI); petsc4py only needs
# the same MINOR as the PETSc C lib, so this is safe against PETSc ${PETSC_VERSION}.
pip install --no-cache-dir --no-build-isolation "petsc4py==${SLEPC_VERSION}"
pip install --no-cache-dir --no-build-isolation "slepc4py==${SLEPC_VERSION}"
pip install --no-cache-dir mpi4py

echo ""
echo "=== build complete. Add to your shell / Dockerfile ENV:"
echo "    export PETSC_DIR=${PETSC_PREFIX}"
echo "    export SLEPC_DIR=${SLEPC_PREFIX}"
echo ""
echo "=== verifying ..."
python - <<PY
from petsc4py import PETSc
from slepc4py import SLEPc
print("SLEPc import OK, PETSc", PETSc.Sys.getVersion())
if "${WITH_MUMPS}" == "1":
    A = PETSc.Mat().createAIJ([4, 4]); A.setUp()
    for i in range(4): A.setValue(i, i, 2.0)
    A.assemble()
    ksp = PETSc.KSP().create(); ksp.setOperators(A); ksp.setType("preonly")
    pc = ksp.getPC(); pc.setType("lu"); pc.setFactorSolverType("mumps"); ksp.setUp()
    print("MUMPS factor OK")
PY
echo "=== done."