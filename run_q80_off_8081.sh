#!/bin/bash
export VENV_ROOT="/home/iacopo/Desktop/TheRock/.tmpvenv-vega"
export CORE_PATH="$VENV_ROOT/lib/python3.12/site-packages/_rocm_sdk_core"
export DEVEL_PATH="$VENV_ROOT/lib/python3.12/site-packages/_rocm_sdk_devel"
export LIBS_PATH="$VENV_ROOT/lib/python3.12/site-packages/_rocm_sdk_libraries_gfx906/lib"
export LD_LIBRARY_PATH="$CORE_PATH/lib:$DEVEL_PATH/lib/llvm/lib:$DEVEL_PATH/lib:$LIBS_PATH:$LD_LIBRARY_PATH"
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
echo "LAUNCH_ENV: GGML_CUDA_REPACK_Q8_0=$GGML_CUDA_REPACK_Q8_0" > /tmp/launch_env.log
exec ./build/bin/llama-server \
    -m "/media/iacopo/LLMs/llms/Qwen3.6-27B-Q8_0.gguf" \
    -ngl 99 -fa on -ctk f16 -ctv f16 \
    --host 0.0.0.0 --port 8081 -c 4096 \
    --jinja --no-mmap -sm tensor -tps 2 -fit off \
    -b 256 -ub 256 -lv 5 --metrics
