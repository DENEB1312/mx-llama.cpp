# 1. Unset the bad paths from our previous attempts
unset ROCM_PATH
unset HIP_PATH
unset CC
unset CXX

# 2. Set the true paths based on your previous output
export VENV_ROOT="/home/iacopo/Desktop/TheRock/.tmpvenv-vega"
export CORE_PATH="$VENV_ROOT/lib/python3.12/site-packages/_rocm_sdk_core"
export DEVEL_PATH="$VENV_ROOT/lib/python3.12/site-packages/_rocm_sdk_devel"

export ROCM_PATH="$DEVEL_PATH"
export HIP_PATH="$DEVEL_PATH"

export PATH="$DEVEL_PATH/lib/llvm/bin:$CORE_PATH/bin:$PATH"
export LD_LIBRARY_PATH="$CORE_PATH/lib:$DEVEL_PATH/lib/llvm/lib:$DEVEL_PATH/lib:$LD_LIBRARY_PATH"

# 3. Nuke the old build entirely
rm -rf build

# 4. Configure CMake (with the explicit CMAKE_HIP_COMPILER_ROCM_ROOT flag for CMake 3.28+)
cmake -B build \
  -DGGML_HIP=ON \
  -DGGML_HIP_RCCL=ON \
  -DAMDGPU_TARGETS="gfx906,gfx1030" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER="$DEVEL_PATH/lib/llvm/bin/clang" \
  -DCMAKE_CXX_COMPILER="$DEVEL_PATH/lib/llvm/bin/clang++" \
  -DCMAKE_HIP_COMPILER_ROCM_ROOT="$DEVEL_PATH" \
  -DCMAKE_PREFIX_PATH="$DEVEL_PATH;$CORE_PATH" \
  -DCMAKE_IGNORE_PATH="/opt/rocm;/opt/rocm-6.3.3;/usr/local/rocm" \
  -DLLAMA_SERVER_METRICS=ON

# 5. Build
cmake --build build --config Release -j$(nproc)
