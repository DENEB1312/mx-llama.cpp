# Extracting HIP kernel ISA assembly (gfx906)

This folder builds a HIP/ROCm project via CMake (target `ggml-hip`). To inspect
the compiled GPU ISA of a `.cu` kernel (e.g. `ggml/src/ggml-cuda/repack-gcn.cu`),
follow these steps.

## 1. Capture the exact compile command

Force recompilation so CMake re-emits the command, and run verbose:

```bash
touch /home/iacopo/Desktop/mx-llama.cpp-gfx906/ggml/src/ggml-cuda/repack-gcn.cu \
  && cmake --build build --target ggml-hip -j1 -- VERBOSE=1 2>&1 \
  | grep -A1 "repack-gcn.cu" | head -10
```

This prints the full `clang++ ... -x hip -c .../repack-gcn.cu` line. The compiler
is the ROCm SDK clang from the venv:

`TheRock/.tmpvenv-vega/lib/python3.12/site-packages/_rocm_sdk_devel/lib/llvm/bin/clang++`

## 2. Compile emitting assembly with `-save-temps`

Take that command, change `--offload-arch=gfx906,gfx1030` -> `--offload-arch=gfx906`
(single target keeps the `.s` small), add `-save-temps`, and send the object to /tmp:

```bash
cd /home/iacopo/Desktop/mx-llama.cpp-gfx906/build/ggml/src/ggml-hip && \
/home/iacopo/Desktop/TheRock/.tmpvenv-vega/lib/python3.12/site-packages/_rocm_sdk_devel/lib/llvm/bin/clang++ \
  -DGGML_BACKEND_BUILD -DGGML_BACKEND_SHARED -DGGML_HIP_GRAPHS -DGGML_HIP_NO_VMM \
  -DGGML_SCHED_MAX_COPIES=5 -DGGML_SHARED -DGGML_USE_HIP -DGGML_USE_NCCL -DUSE_PROF_API=1 \
  -D_GNU_SOURCE -D_XOPEN_SOURCE=600 -D__HIP_PLATFORM_AMD__=1 -D__HIP_ROCclr__=1 -Dggml_hip_EXPORTS \
  -I/home/iacopo/Desktop/mx-llama.cpp-gfx906/ggml/src/ggml-hip/.. \
  -I/home/iacopo/Desktop/mx-llama.cpp-gfx906/ggml/src/../include \
  -O3 -DNDEBUG -std=gnu++17 --offload-arch=gfx906 -fPIC -ffast-math \
  -Wno-nan-infinity-disabled -save-temps -o /tmp/repack_test.o -x hip -c \
  /home/iacopo/Desktop/mx-llama.cpp-gfx906/ggml/src/ggml-cuda/repack-gcn.cu
```

## 3. Locate the device ISA file

`-save-temps` produces two `.s` files. The **device (GPU) ISA** is:

```
build/ggml/src/ggml-hip/repack-gcn-hip-amdgcn-amd-amdhsa-gfx906.s
```

The other `repack-gcn-host-x86_64-...s` is the host stub and is NOT what you want.

## 4. Extract the kernel body

The device `.s` is large. Slice out the compute portion to a small file for analysis:

```bash
sed -n '540,1900p' \
  /home/iacopo/Desktop/mx-llama.cpp-gfx906/build/ggml/src/ggml-hip/repack-gcn-hip-amdgcn-amd-amdhsa-gfx906.s \
  > /tmp/kernel_main.s
wc -l /tmp/kernel_main.s
```

Adjust the line range to cover the target kernel (`mmq_gemm_q8_0_repacked` /
`mmq_gemm_q8_0_repacked_w32`). Search for the mangled kernel symbol or `.text`
section to find the right range.

## 5. Analyze the ISA

Write a Python script to parse `/tmp/kernel_main.s` and count the instruction mix:
dp4a (`v_dot4_i32_i8`) vs FMA (`v_fma_f32` / `v_fmac_f32_e32` / `v_fma_mix_f32`),
LDS ops (`ds_*`), global memory (`buffer_*` / `global_*`), scalar (`s_*`),
`s_waitcnt` stalls (lgkmcnt = LDS/scalar, vmcnt = global loads), register usage
from `.amdhsa_kernel` metadata (vgpr/sgpr counts, spills), and loop structure
(branch/barrier positions).

Example metrics to extract:

- Instruction mix percentages (vector / LDS / global / scalar)
- `v_dot4_i32_i8` count and dp4a/FMA ratio
- Number and type of `s_waitcnt` stalls (latency hiding)
- `v_mov_b32` count (register-pressure shuffling overhead)
- `ds_read`/`ds_write` vs `buffer_load`/`buffer_store` balance
- Hot-loop body density between `s_barrier` calls

## Quick reference

```bash
# 1.
touch ggml/src/ggml-cuda/repack-gcn.cu
cmake --build build --target ggml-hip -j1 -- VERBOSE=1 2>&1 | grep -A1 "repack-gcn.cu"
# 2. rerun that clang++ line with: --offload-arch=gfx906 -save-temps -o /tmp/repack_test.o
# 3. device ISA: build/ggml/src/ggml-hip/repack-gcn-hip-amdgcn-amd-amdhsa-gfx906.s
# 4.
sed -n '540,1900p' build/ggml/src/ggml-hip/repack-gcn-hip-amdgcn-amd-amdhsa-gfx906.s > /tmp/kernel_main.s
# 5. analyze /tmp/kernel_main.s with a python script
```
