#!/bin/bash
# shellcheck disable=SC1143,SC2215
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

# =================================================================
# 1. THE ROCK (ROCm 7.13) RUNTIME HOOKS
# =================================================================
export VENV_ROOT="/home/iacopo/Desktop/TheRock/.tmpvenv-vega"
export CORE_PATH="$VENV_ROOT/lib/python3.12/site-packages/_rocm_sdk_core"
export DEVEL_PATH="$VENV_ROOT/lib/python3.12/site-packages/_rocm_sdk_devel"
export LIBS_PATH="$VENV_ROOT/lib/python3.12/site-packages/_rocm_sdk_libraries_gfx906/lib"
export LD_LIBRARY_PATH="$CORE_PATH/lib:$DEVEL_PATH/lib/llvm/lib:$DEVEL_PATH/lib:$LIBS_PATH:$LD_LIBRARY_PATH"
#export NCCL_DEBUG=INFO
export HSA_OVERRIDE_GFX_VERSION=9.0.6
export HIP_VISIBLE_DEVICES=0,1
export GGML_CUDA_P2P=1 
export TURBOPREFILL=1 
export GGML_ENABLE_CUSTOM_AR=1
export HSA_FORCE_FINE_GRAIN_PCIE=1
export GPU_MAX_HW_QUEUES=8
export GGML_TP_AR_TWOSHOT=0
export GGML_TP_AR_BLOCKS=60
export GGML_TP_AR_FORCE=1
export GGML_TP_AR_WIRE=bf16
export GGML_CUDA_REPACK=1
export GGML_CUDA_REPACK_Q8_0=0
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

# Model path
#MODEL_PATH="/media/iacopo/LLMs/llms/Qwen3.6-35B-A3B-Q8_0.gguf"
#MODEL_PATH="/media/iacopo/LLMs/llms/Qwen3-4B-Instruct-2507-Q8_0.gguf"
#MODEL_PATH="/home/iacopo/Downloads/deepreinforce-ai_Ornith-1.0-35B-Q8_0.gguf"
MODEL_PATH="/media/iacopo/LLMs/llms/Qwen3.6-27B-Q8_0.gguf"
#MODEL_PATH="/media/iacopo/LLMs/llms/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-Q8_K_P.gguf"
# Vision projector path (uncomment for multimodal models)
#MMPROJ_PATH="/media/iacopo/LLMs/llms/mmproj-F16_35B.gguf"
MMPROJ_PATH="/media/iacopo/LLMs/llms/mmproj-F16_27B.gguf"

#ngram settings:
#        --spec-type ngram-mod \
#        --spec-ngram-size-n 24 \
#        --draft-min 48 \
#        --draft-max 64 \
#    --spec-type draft-mtp --spec-draft-n-max 3 \
#GGML_CUDA_P2P=1 
#    --mmproj "$MMPROJ_PATH" \
#CUDA_SCALE_LAUNCH_QUEUES=4x \
#    --mmproj "$MMPROJ_PATH" \
./build/bin/llama-server \
    -m "$MODEL_PATH" \
    -ngl 99 \
    -fa on \
    -ctk f16 \
    -ctv f16 \
    --host 0.0.0.0 \
    --port 8080 \
    -c 262144 \
    --jinja \
    --no-mmap \
    -sm tensor \
    -tps 2 \
    -fit off \
    --models-dir /media/iacopo/LLMs/llms/ \
    --chat-template-kwargs '{"preserve_thinking": true}' \
    --temp 0.6 \
    --top-p 0.95 \
    --top-k 20 \
    --min-p 0.00 \
    -b 2048 -ub 2048 --image-min-tokens 2048 --image-max-tokens 32000 \
    -lv 5 --metrics --mmproj "$MMPROJ_PATH" > server.log 2>&1 \




    
