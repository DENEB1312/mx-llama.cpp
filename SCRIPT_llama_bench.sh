#!/bin/bash
cat << 'EOF'

   ██╗     ██╗      █████╗ ███╗   ███╗ █████╗    ██████╗██████╗ ██████╗
   ██║     ██║     ██╔══██╗████╗ ████║██╔══██╗  ██╔════╝██╔══██╗██╔══██╗
   ██║     ██║     ███████║██╔████╔██║███████║  ██║     ██████╔╝██████╔╝
   ██║     ██║     ██╔══██║██║╚██╔╝██║██╔══██║  ██║     ██╔═══╝ ██╔═══╝
   ███████╗███████╗██║  ██║██║ ╚═╝ ██║██║  ██║  ╚██████╗██║     ██║
   ╚══════╝╚══════╝╚═╝  ╚═╝╚═╝     ╚═╝╚═╝  ╚═╝   ╚═════╝╚═╝     ╚═╝
            ██████╗ ███████╗██╗  ██╗ █████╗  ██████╗  ██████╗
           ██╔════╝ ██╔════╝╚██╗██╔╝██╔══██╗██╔═████╗██╔════╝
           ██║  ███╗█████╗   ╚███╔╝ ╚██████║██║██╔██║███████╗
           ██║   ██║██╔══╝   ██╔██╗  ╚═══██║████╔╝██║██╔═══██╗
           ╚██████╔╝██║     ██╔╝ ██╗ █████╔╝╚██████╔╝╚██████╔╝
            ╚═════╝ ╚═╝     ╚═╝  ╚═╝ ╚════╝  ╚═════╝  ╚═════╝


EOF
export VENV_ROOT="/home/iacopo/Desktop/TheRock/.tmpvenv-vega"
export CORE_PATH="$VENV_ROOT/lib/python3.12/site-packages/_rocm_sdk_core"
export DEVEL_PATH="$VENV_ROOT/lib/python3.12/site-packages/_rocm_sdk_devel"
export LIBS_PATH="$VENV_ROOT/lib/python3.12/site-packages/_rocm_sdk_libraries_gfx906/lib"
export LD_LIBRARY_PATH="$CORE_PATH/lib:$DEVEL_PATH/lib/llvm/lib:$DEVEL_PATH/lib:$LIBS_PATH:$LD_LIBRARY_PATH"
export HSA_OVERRIDE_GFX_VERSION=9.0.6
export HIP_VISIBLE_DEVICES=0
export GGML_CUDA_P2P=1
export GGML_ENABLE_CUSTOM_AR=1
export HSA_FORCE_FINE_GRAIN_PCIE=1
export GPU_MAX_HW_QUEUES=8
export GGML_LOG_LEVEL=1
export TURBOPREFILL=1
# A/B repack toggle:
#   arg "repack" -> run only WITH repack (GGML_CUDA_REPACK=1, GGML_CUDA_REPACK_Q8_0=1)
#   arg "native" -> run only WITHOUT repack (GGML_CUDA_REPACK=0, GGML_CUDA_REPACK_Q8_0=0)
#   no arg       -> run A/B: both native then repack
if [ "$1" = "repack" ]; then MODES=(1); shift
elif [ "$1" = "native" ]; then MODES=(0); shift
else MODES=(0 1); fi
# K-sharding (GGML_MMVQ_KSHARD) regresses decode on gfx906 for this MoE workload; off by default.
# Opt back in experimentally with: GGML_MMVQ_KSHARD_MAXROWS=<row ceiling>
export GGML_MMVQ_KSHARD_MAXROWS=1
# Install RCCL tuner config for gfx906 (if not already present)
TUNER_DIR="$DEVEL_PATH/share/rccl/tuner"
TUNER_FILE="$TUNER_DIR/rccl_tuner_gfx906.csv"
if [ -f "$TUNER_FILE" ]; then
  echo "[tuner] Using existing: $TUNER_FILE"
else
  echo "[tuner] Installing tuner config to $TUNER_DIR"
  mkdir -p "$TUNER_DIR"
  cp rccl_tuner_gfx906.csv "$TUNER_FILE"
fi

#MODEL_PATH="/path/..."
LOG_FILE="bench_results.md"
MODEL_PATH="/media/iacopo/LLMs/llms/Qwen3.6-27B-Q8_0.gguf"
MODEL_PATH="/media/iacopo/LLMs/llms/Qwen3-4B-Instruct-2507-Q8_0.gguf"
#MODEL_PATH="/media/iacopo/LLMs/llms/Qwen_Qwen3.5-4B-Q8_0.gguf"
BENCH_PARAMS=(
    -m "$MODEL_PATH"       # Model path
    -ngl 99                # Number of GPU layers (all on GPU)
    -t "$(nproc)"          # Number of CPU threads
    -fa 1                  # Flash attention (1=on, 0=off)
    -ctk f16              # KV cache key type (q8_0 quantization)
    -ctv f16               # KV cache value type (f16 precision)
    --progress             # Show progress during benchmark
    -r 1                   # Number of repetitions
    -ub "16-2048*2"               # Micro-batch size
    -mmp 0 
    #--spec-type draft-mtp --spec-draft-n-max 2 
    #-d 8192               # Context size
    #-v                     # Verbose output (show llama.cpp internal logs)
)


#BENCH_TESTS="-p 0 -n 2048"
#BENCH_TESTS="-p 2048 -n 0"
BENCH_TESTS="-p 128,2048 -n 0"
#BENCH_TESTS="-p 512 -n 128"

echo "=== Benchmark ==="
echo "Model: $(basename "$MODEL_PATH")"
echo ""

cd "$(dirname "$0")" || exit
[ ! -f "./build/bin/llama-bench" ] && echo "Error: llama-bench not found" && exit 1

SCRIPT_ARGS=("$@")

run_one() {
    local repack="$1"
    if [ "$repack" = "1" ]; then
        export GGML_CUDA_REPACK=1
        export GGML_CUDA_REPACK_Q8_0=1
        tag="repack(on)"
    else
        export GGML_CUDA_REPACK=0
        export GGML_CUDA_REPACK_Q8_0=0
        tag="native(off)"
    fi
    echo "" >> "$LOG_FILE"
    echo "### $(date +%H:%M:%S) [$tag] pp2048/tg128" >> "$LOG_FILE"
    echo ">>> [$tag]"
    ./build/bin/llama-bench "${BENCH_PARAMS[@]}" $BENCH_TESTS "${SCRIPT_ARGS[@]}" 2>&1 | tee -a "$LOG_FILE"
}

for mode in "${MODES[@]}"; do
    run_one "$mode"
done

echo ""
echo "Output saved to: $LOG_FILE"
