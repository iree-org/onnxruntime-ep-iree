#!/usr/bin/env bash
# build_kernels.sh - Compile HIP test kernels to HSACO (.co) files.
#
# Usage: ./build_kernels.sh [--arch gfx1100] [--rocm-path /opt/rocm]
#
# Prerequisites:
#   - ROCm installed (clang with HIP support)
#
# Output:
#   test/build/kernel_${ARCH}.co

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Defaults.
ARCH="gfx1100"
ROCM_PATH="${ROCM_PATH:-/opt/rocm}"

# Parse arguments.
while [[ $# -gt 0 ]]; do
  case "$1" in
    --arch) ARCH="$2"; shift 2 ;;
    --rocm-path) ROCM_PATH="$2"; shift 2 ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

# Find clang with HIP support.
# ROCm 6+ puts clang under llvm/bin/, older versions used bin/ directly.
CLANG="${ROCM_PATH}/llvm/bin/clang"
if [[ ! -x "$CLANG" ]]; then
  CLANG="${ROCM_PATH}/bin/clang"
fi
if [[ ! -x "$CLANG" ]]; then
  CLANG="$(command -v clang 2>/dev/null || true)"
fi

if [[ -z "$CLANG" ]] || [[ ! -x "$CLANG" ]]; then
  echo "ERROR: clang not found (tried ROCm paths and PATH)"
  echo "  Set ROCM_PATH or ensure clang is on PATH."
  exit 1
fi

BUILD_DIR="$SCRIPT_DIR/build"
mkdir -p "$BUILD_DIR"

echo "=== Building HIP kernel → kernel_${ARCH}.co ==="
echo "  ARCH:      $ARCH"
echo "  ROCM_PATH: $ROCM_PATH"
echo "  CLANG:     $CLANG"

CO_FILE="$BUILD_DIR/kernel_${ARCH}.co"

"$CLANG" \
  -x hip \
  --offload-device-only \
  "--offload-arch=${ARCH}" \
  "--rocm-path=${ROCM_PATH}" \
  -fuse-cuid=none \
  -O3 \
  "$SCRIPT_DIR/kernel.cu" \
  -o "$CO_FILE"

echo "  Output: $CO_FILE"
echo "=== Done ==="
