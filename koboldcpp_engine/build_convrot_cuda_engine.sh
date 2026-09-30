#!/bin/bash
# Builds koboldcpp_cublas.so from source on a live worker with a real GPU
# attached, using a modern CUDA toolkit (12.8.0) whose cuBLAS actually ships
# INT8 tensor-core (IMMA) kernels for Blackwell (sm_120, e.g. RTX 5090).
#
# Why this exists: the official prebuilt koboldcpp_cublas.so (what
# ensure_koboldcpp_engine() used to extract from
# https://koboldai.org/cpplinuxcu12) is built against CUDA 12.1.0 — see
# upstream's own environment.yaml, `channels: - nvidia/label/cuda-12.1.0`.
# cuBLAS from that toolkit predates Blackwell hardware entirely (RTX
# 50-series shipped alongside CUDA 12.8). ConvRot's INT8 tensorwise matmul
# (ggml_cuda_mul_mat_i8 in ggml-cuda.cu) calls cublasGemmEx with
# CUBLAS_COMPUTE_32I + CUBLAS_GEMM_DEFAULT_TENSOR_OP on CUDA_R_8I data — the
# compile-time gate for this op (turing_mma_available() in common.cuh)
# treats any compiled arch >= Turing (750) as fine, which incorrectly
# includes Blackwell (1200) too, so ggml happily assigns the op to CUDA —
# but cuBLAS 12.1 has no IMMA kernel for sm_120 at all, so the actual
# cublasGemmEx call fails at runtime with "the requested functionality is
# not supported".
#
# Fix: rebuild just the CUDA backend (koboldcpp_cublas.so) from source,
# against CUDA 12.8.0 instead, on a worker with the real target GPU
# physically attached — this Makefile's own NVCCFLAGS default
# (`-arch=native`) then resolves to exactly the right architecture
# automatically, no arch list to hand-maintain.
#
# Source is pinned to the exact commit already validated by a standalone
# CUDA-backend correctness test for ConvRot, not whatever HEAD happens to
# be, so this reproduces a known-good build.
set -euo pipefail

SCRATCH_REPO_URL="https://github.com/SimoneOCE/koboldcpp-convrot-scratch.git"
# TEMPORARY: pinned to the debug-weightscale branch's tip, not the
# validated main-line commit, while we track down the black-frame-output
# bug via CONVROT_DEBUG_SCALES (see ggml_cuda_mul_mat_i8 in
# ggml-cuda.cu). Revert to 9676017cc69cb8f2c20b2fbf5718bc1163a2b632 (the
# last commit validated by the standalone CUDA-backend correctness test)
# once this is resolved - the debug prints are gated behind an env var
# and inert otherwise, but there's no reason to keep building from an
# unreviewed branch once we don't need to.
SCRATCH_REPO_COMMIT="fbbd6ddc9d10cbe79164a36c47df421b9140a3f7"
CUDA_CHANNEL_LABEL="nvidia/label/cuda-12.8.0"

BUILD_ROOT="${1:?usage: build_convrot_cuda_engine.sh <build-root-dir> <output-dir>}"
OUT_DIR="${2:?usage: build_convrot_cuda_engine.sh <build-root-dir> <output-dir>}"

mkdir -p "$BUILD_ROOT" "$OUT_DIR"
cd "$BUILD_ROOT"

echo "[convrot-build] Checking for a visible NVIDIA GPU..."
if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi -L 2>/dev/null | grep -qE '^GPU [0-9]+:'; then
    echo "[convrot-build] FATAL: no NVIDIA GPU visible to this container - refusing to build (NVCCFLAGS uses -arch=native, which needs one present to resolve to anything)." >&2
    exit 1
fi
nvidia-smi -L

echo "[convrot-build] Setting up micromamba..."
mkdir -p bin
if [ ! -f "bin/micromamba" ]; then
    curl -Ls https://anaconda.org/conda-forge/micromamba/1.5.3/download/linux-64/micromamba-1.5.3-0.tar.bz2 | tar -xvj -C . bin/micromamba
fi

if [ ! -x "conda/envs/build/bin/nvcc" ]; then
    echo "[convrot-build] Creating build environment (CUDA 12.8.0 toolkit)..."
    rm -rf conda/envs/build
    cat > environment_build.yaml <<EOF
name: koboldcpp-build
channels:
  - ${CUDA_CHANNEL_LABEL}
  - conda-forge
dependencies:
  - cuda-nvcc
  - cuda-cuobjdump
  - cuda-libraries-dev
  - cxx-compiler
  - gxx=10
  - git
  - make
  - libcblas
EOF
    bin/micromamba create --no-rc --no-shortcuts -r conda -p conda/envs/build -f environment_build.yaml -y
fi

echo "[convrot-build] Fetching koboldcpp source @ ${SCRATCH_REPO_COMMIT}..."
if [ ! -d "src/.git" ]; then
    rm -rf src
    mkdir -p src
    (cd src && git init -q && git remote add origin "$SCRATCH_REPO_URL")
fi
(cd src && git fetch --depth 1 origin "$SCRATCH_REPO_COMMIT" && git checkout -q FETCH_HEAD)

echo "[convrot-build] Building koboldcpp_cublas.so (this can take a while - full C++/CUDA rebuild)..."
cd src

# The final link step needs -lcuda (the CUDA *driver* library, distinct from
# -lcudart/-lcublas which are runtime libs and link fine on their own).
# There's no real libcuda.so on a build machine's linker search path - only
# the driver's own versioned runtime .so.1, sufficient to run but not to
# link against - so the cuda-driver-dev conda package ships a stub
# libcuda.so specifically for this. It's not on the default search path
# (that's the whole reason it's called a "stub" dir), so find it and hand
# it to the linker explicitly via LIBRARY_PATH, which gcc/g++ read
# automatically for every -l flag without needing Makefile changes.
CUDA_STUB_DIR=$(dirname "$(find "$BUILD_ROOT/conda/envs/build" -name 'libcuda.so' -path '*stubs*' 2>/dev/null | head -1)")
if [ -z "$CUDA_STUB_DIR" ] || [ "$CUDA_STUB_DIR" = "." ]; then
    echo "[convrot-build] FATAL: libcuda.so stub not found anywhere under the build env - cuda-driver-dev may not have installed it where expected." >&2
    exit 1
fi
echo "[convrot-build] Found libcuda.so stub at $CUDA_STUB_DIR"

LIBRARY_PATH="$CUDA_STUB_DIR${LIBRARY_PATH:+:$LIBRARY_PATH}" \
"$BUILD_ROOT/bin/micromamba" run -r "$BUILD_ROOT/conda" -p "$BUILD_ROOT/conda/envs/build" \
    env LIBRARY_PATH="$CUDA_STUB_DIR${LIBRARY_PATH:+:$LIBRARY_PATH}" \
    make -j"$(nproc)" LLAMA_CUBLAS=1 LLAMA_ADD_CONDA_PATHS=1 koboldcpp_cublas

if [ ! -f koboldcpp_cublas.so ]; then
    echo "[convrot-build] FATAL: build finished but koboldcpp_cublas.so was not produced." >&2
    exit 1
fi

echo "[convrot-build] Build OK. Checking which GPU architecture actually got compiled in..."
GPU_CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')
CUOBJDUMP="$BUILD_ROOT/conda/envs/build/bin/cuobjdump"
if [ ! -x "$CUOBJDUMP" ]; then
    echo "[convrot-build] FATAL: cuobjdump not found at $CUOBJDUMP - can't verify -arch=native actually embedded sm_${GPU_CC} code. This check exists precisely because that silently failed before (runtime warning: 'ggml was not compiled with any CUDA arch'), so refusing to ship an unverified build." >&2
    exit 1
fi
EMBEDDED_ARCHS=$("$CUOBJDUMP" --list-elf koboldcpp_cublas.so 2>/dev/null | grep -o "sm_[0-9]*" | sort -u)
echo "[convrot-build] Embedded SASS architectures: ${EMBEDDED_ARCHS:-<none found>}"
if ! echo "$EMBEDDED_ARCHS" | grep -q "^sm_${GPU_CC}\$"; then
    echo "[convrot-build] FATAL: sm_${GPU_CC} (this GPU's compute capability) is NOT among the compiled architectures ($EMBEDDED_ARCHS). -arch=native failed to target the attached GPU - refusing to ship a binary that will silently produce wrong output at runtime." >&2
    exit 1
fi
echo "[convrot-build] Confirmed sm_${GPU_CC} is present in the built binary."

echo "[convrot-build] Collecting output into $OUT_DIR..."
cp koboldcpp_cublas.so "$OUT_DIR/"
chmod 755 "$OUT_DIR/koboldcpp_cublas.so"

# koboldcpp_cublas.so dynamically links against this env's CUDA runtime libs
# (libcublas.so.12, libcudart.so.12, etc) — it needs them alongside it at
# runtime, same reason ensure_koboldcpp_engine() used to copy every .so out
# of the official binary's extraction, not just koboldcpp_cublas.so itself.
COPIED=0
for lib in "$BUILD_ROOT"/conda/envs/build/lib/lib{cublas,cublasLt,cudart,curand}.so.*; do
    [ -e "$lib" ] || continue
    cp -L "$lib" "$OUT_DIR/"
    chmod 755 "$OUT_DIR/$(basename "$lib")"
    COPIED=$((COPIED + 1))
done
echo "[convrot-build] Copied $COPIED CUDA runtime shared library file(s) to $OUT_DIR."

echo "[convrot-build] Cleaning up build workspace (toolkit env + source + object files) to free volume space..."
cd "$BUILD_ROOT"
rm -rf conda src environment_build.yaml

echo "[convrot-build] Done."
