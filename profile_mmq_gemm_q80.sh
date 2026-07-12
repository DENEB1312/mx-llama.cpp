#!/bin/bash
# Profile mmq_gemm_q8_0_repacked with rocprofv3 hardware counters.
# Usage: bash profile_mmq_gemm_q80.sh [1|0]   (1=repack, 0=native)
set -euo pipefail

# --- Environment (mirrors SCRIPT_llama_bench.sh) ---
export VENV_ROOT="/home/iacopo/Desktop/TheRock/.tmpvenv-vega"
export CORE_PATH="$VENV_ROOT/lib/python3.12/site-packages/_rocm_sdk_core"
export DEVEL_PATH="$VENV_ROOT/lib/python3.12/site-packages/_rocm_sdk_devel"
export LIBS_PATH="$VENV_ROOT/lib/python3.12/site-packages/_rocm_sdk_libraries_gfx906/lib"
export LD_LIBRARY_PATH="$CORE_PATH/lib:$DEVEL_PATH/lib/llvm/lib:$DEVEL_PATH/lib:$LIBS_PATH:${LD_LIBRARY_PATH:-}"
export HSA_OVERRIDE_GFX_VERSION=9.0.6
export HIP_VISIBLE_DEVICES=0
export GGML_CUDA_P2P=1
export GGML_ENABLE_CUSTOM_AR=1
export HSA_FORCE_FINE_GRAIN_PCIE=1
export GPU_MAX_HW_QUEUES=8
export GGML_LOG_LEVEL=1
export TURBOPREFILL=1
export GGML_CUDA_REPACK=1
export GGML_CUDA_REPACK_Q8_0=${1:-1}
export GGML_MMVQ_KSHARD_MAXROWS=1
# NOTE: do NOT set GGML_CUDA_DISABLE_GRAPHS=1 — it bypasses the repack dispatch

MODEL="/media/iacopo/LLMs/llms/Qwen3-4B-Instruct-2507-Q8_0.gguf"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PMC_FILE="$SCRIPT_DIR/profile_mmq_gemm_q80.pmc.txt"
OUTDIR="$SCRIPT_DIR/rocprofv3-mmq-q80-$(date +%Y%m%d-%H%M%S)"

echo "=== rocprofv3 PMC profiling ==="
echo "Model: $(basename "$MODEL")"
echo "REPACK_Q8_0=$GGML_CUDA_REPACK_Q8_0"
echo "Counters: $(tr '\n' ' ' < "$PMC_FILE")"
echo "Output: $OUTDIR/"
echo ""

# Install RCCL tuner if needed (copied from bench script)
TUNER_DIR="$DEVEL_PATH/share/rccl/tuner"
TUNER_FILE="$TUNER_DIR/rccl_tuner_gfx906.csv"
if [ ! -f "$TUNER_FILE" ]; then
    mkdir -p "$TUNER_DIR"
    cp "$SCRIPT_DIR/rccl_tuner_gfx906.csv" "$TUNER_FILE" 2>/dev/null || true
fi

cd "$SCRIPT_DIR" || exit 1
[ ! -f "./build/bin/llama-bench" ] && echo "Error: llama-bench not found" && exit 1

rocprofv3 \
    -i "$PMC_FILE" \
    -o "$OUTDIR/pmc" \
    --output-format csv \
    --kernel-trace \
    --summary \
    --summary-output-file "$OUTDIR/summary.txt" \
    --log-level warning \
    -- \
    ./build/bin/llama-bench -m "$MODEL" \
    -ngl 99 -t "$(nproc)" -fa 1 -ctk f16 -ctv f16 \
    -b 2048 -ub 2048 -mmp 0 -p 2048 -n 0 -r 1 2>&1

echo ""
echo "=== Results ==="

# Find the PMC CSV
PMC_CSV=$(find "$OUTDIR" -name "*.csv" -path "*pmc*" | head -1)
if [ -z "$PMC_CSV" ]; then
    # Try any CSV
    PMC_CSV=$(find "$OUTDIR" -name "*.csv" | head -1)
fi

if [ -z "$PMC_CSV" ]; then
    echo "ERROR: No CSV found in $OUTDIR/"
    ls -R "$OUTDIR/"
    exit 1
fi

echo "Raw CSV: $PMC_CSV"
echo ""

# Show all columns
echo "--- CSV columns ---"
head -1 "$PMC_CSV"
echo ""

# Filter to the mmq_gemm kernel
echo "--- mmq_gemm_q8_0_repacked rows ---"
grep "mmq_gemm_q8_0" "$PMC_CSV" || echo "(no mmq_gemm_q8_0 rows — printing all kernels)"
echo ""

# Show all kernel rows sorted by duration
echo "--- All kernels (top 5 by duration) ---"
head -1 "$PMC_CSV"
grep -v "^Kernel" "$PMC_CSV" 2>/dev/null | sort -t',' -k5 -nr 2>/dev/null | head -5
echo ""

# Summary
if [ -f "$OUTDIR/summary.txt" ]; then
    echo "--- Summary ---"
    grep -A5 "mmq_gemm_q8_0" "$OUTDIR/summary.txt" || cat "$OUTDIR/summary.txt" | head -40
fi

echo ""
echo "Done. Full data: $PMC_CSV"
echo "Analyze: less $PMC_CSV"
