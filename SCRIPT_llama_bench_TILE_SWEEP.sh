#!/bin/bash
cat << 'EOF'

   ████████╗██╗██╗     ███████╗    ███████╗██╗   ╗███████╗███████╗██████╗
   ╚══██╔══╝██║██║     ██╔════╝    ██╔════╝██║    ║██╔════╝██╔════╝██╔══██╗
      ██║   ██║██║     █████╗      ███████╗██║ █╗ ║█████╗  █████╗  ██████╔╝
      ██║   ██║██║     ██╔══╝      ╚════██║██║███╗║██╔══╝  ██╔══╝  ██╔═══╝
      ██║   ██║███████╗███████╗    ███████║╚███╔██║███████╗███████╗██║
      ╚═╝   ╚═╝╚══════╝╚══════╝    ╚══════╝ ╚══╝╚═╝╚══════╝╚══════╝╚═╝

      Sweeping ALL tile parameters (BM x BK x TN x NROW_LANES) x PROMPT
EOF

# ── Environment (copied from SCRIPT_llama_bench.sh) ──────────────────────
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
export GGML_CUDA_REPACK=1
export GGML_CUDA_REPACK_Q8_0=1
export GGML_MMVQ_KSHARD_MAXROWS=1

TUNER_DIR="$DEVEL_PATH/share/rccl/tuner"
TUNER_FILE="$TUNER_DIR/rccl_tuner_gfx906.csv"
if [ ! -f "$TUNER_FILE" ]; then
  mkdir -p "$TUNER_DIR"
  cp rccl_tuner_gfx906.csv "$TUNER_FILE"
fi

MODEL_PATH="/media/iacopo/LLMs/llms/Qwen_Qwen3.5-4B-Q8_0.gguf"
LOG_FILE="bench_tile_sweep.md"
RAW_FILE="bench_tile_sweep_raw.md"
SRC_FILE="ggml/src/ggml-cuda/repack-gcn.cu"
BUILD_CMD="cmake --build build --target llama-bench -j24"

# ubatch sweep: 16,32,64,128,256,512,1024,2048
UBATCH_RANGE="16-2048*2"
UBATCH_LIST=(16 32 64 128 256 512 1024 2048)

BENCH_PARAMS=(
    -m "$MODEL_PATH"
    -ngl 99
    -t "$(nproc)"
    -fa 1
    -ctk f16
    -ctv f16
    --progress
    -r 1
    -ub "16-2048*2"               # Micro-batch size
    -mmp 0
)

# Prompt sizes to sweep (where the repack regression is observed at pp128/pp2048)
PROMPT_VALUES=(2048)

cd "$(dirname "$0")" || exit 1
[ ! -f "./build/bin/llama-bench" ] && echo "Error: llama-bench not found" && exit 1

# ── Save original source ─────────────────────────────────────────────────
cp "$SRC_FILE" "${SRC_FILE}.orig"

# ── Python helper: patch macros + launch_bounds + dispatch dim3 ───────────
# All patches are STRING-ANCHORED (regex on the #define / launch_bounds /
# dispatch tokens) so the script survives future line-number shifts in the
# source. Patches 3 locations in repack-gcn.cu:
#   #define MMQ_RP_Q8_{BK,TN,BM,BN,NROW_LANES}
#   __launch_bounds__(512, 1) mmq_gemm_q8_0_repacked(...)
#   mmq_gemm_q8_0_repacked<false, ...><<<grid, dim3(64, 8), ...>>>
patch_config() {
    local BK=$1 TN=$2 BM=$3 NROW_LANES=$4
    python3 - "$SRC_FILE" "$BK" "$TN" "$BM" "$NROW_LANES" << 'PYEOF'
import sys, re
src, bk, tn, bm, nrl = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
threads = 64 * nrl
with open(src) as f:
    text = f.read()

def set_define(text, name, value):
    pat = re.compile(r'^#define\s+' + re.escape(name) + r'\b.*$', re.M)
    repl = '#define %s %s' % (name, value)
    if pat.search(text):
        return pat.sub(repl, text, count=1)
    return text + '\n' + repl + '\n'

text = set_define(text, 'MMQ_RP_Q8_BK', bk)
text = set_define(text, 'MMQ_RP_Q8_TN', tn)
text = set_define(text, 'MMQ_RP_Q8_BM', bm)
text = set_define(text, 'MMQ_RP_Q8_BN', '(64 * MMQ_RP_Q8_TN)')
text = set_define(text, 'MMQ_RP_Q8_NROW_LANES', nrl)

# launch_bounds — anchored to the mmq_gemm_q8_0_repacked declaration only
lb = re.compile(r'__launch_bounds__\(\s*512\s*,\s*1\s*\)\s*mmq_gemm_q8_0_repacked')
if not lb.search(text):
    sys.stderr.write('WARN: launch_bounds anchor not found\n')
text = lb.sub('__launch_bounds__(%d, 1) mmq_gemm_q8_0_repacked' % threads, text, count=1)

# dispatch dim3(64, 8) — anchored to the <false, ...> instantiation only
disp = re.compile(r'(mmq_gemm_q8_0_repacked<false[^\n]*<<<grid,\s*)dim3\(\s*64\s*,\s*8\s*\)')
if not disp.search(text):
    sys.stderr.write('WARN: dispatch dim3 anchor not found\n')
text = disp.sub(lambda m: m.group(1) + 'dim3(64, %d)' % nrl, text, count=1)

with open(src, 'w') as f:
    f.write(text)
PYEOF
}

restore_source() {
    cp "${SRC_FILE}.orig" "$SRC_FILE"
}

# ── Python helper: extract per-ubatch pp t/s from llama-bench markdown ────
# Reads the llama-bench stdout FILE passed as $1, finds the header row
# containing 't/s', then pulls that column from each subsequent data row in
# order. The order of data rows matches the -ub sweep (16,32,...,2048), so we
# map them positionally. The bench output is passed via a file (not a pipe)
# because the heredoc below supplies the python SOURCE on stdin.
extract_ts() {
    python3 - "$1" << 'PYEOF'
import sys, re
out = open(sys.argv[1]).read()
lines = out.splitlines()
header_idx = None
cols = None
for i, l in enumerate(lines):
    if l.strip().startswith('|') and 't/s' in l:
        header_idx = i
        cols = [c.strip() for c in l.strip().strip('|').split('|')]
        break
if header_idx is None or cols is None:
    print(' '.join(['?'] * 8))
    sys.exit(0)
ti = None
for j, c in enumerate(cols):
    if c == 't/s':
        ti = j
        break
if ti is None:
    ti = len(cols) - 1
vals = []
for l in lines[header_idx + 1:]:
    s = l.strip()
    if not s.startswith('|'):
        continue
    if set(s) <= set('|-: '):
        continue  # separator
    cells = [c.strip() for c in s.strip('|').split('|')]
    if ti < len(cells):
        # Keep ONLY the leading numeric token (llama-bench prints "216.10 ± 0.00");
        # otherwise the caller's `read -ra` would split it into 3 array elements.
        m = re.search(r'-?\d+(?:\.\d+)?', cells[ti])
        if m:
            vals.append(m.group(0))
print(' '.join(vals))
PYEOF
}

# ── Parameter ranges ──────────────────────────────────────────────────────
BM_VALUES=(64 128 256)
BK_VALUES=(2 4 8)
TN_VALUES=(1 2)
NROW_LANES_VALUES=(4 8)

# ── Build log header ──────────────────────────────────────────────────────
{
    echo "# Tile Parameter Sweep — $(date '+%Y-%m-%d %H:%M')"
    echo ""
    echo "Model: $(basename "$MODEL_PATH")"
    echo "Bench: pp sweep, repack on (GGML_CUDA_REPACK_Q8_0=1)"
    echo "ubatch: $UBATCH_RANGE"
    echo "prompts: ${PROMPT_VALUES[*]}"
    echo ""
} > "$LOG_FILE"

{
    echo "# Raw llama-bench tables — Tile Parameter Sweep — $(date '+%Y-%m-%d %H:%M')"
    echo ""
} > "$RAW_FILE"

COUNT=0
PASS=0
FAIL=0
SKIP=0

for P in "${PROMPT_VALUES[@]}"; do
    MATFILE="bench_tile_sweep_P${P}.mat"
    rm -f "$MATFILE"

    echo ""
    echo "##################################################"
    echo "# PROMPT = $P"
    echo "##################################################"

    {
        echo ""
        echo "## Prompt = $P  (ubatch: ${UBATCH_LIST[*]})"
        echo ""
        echo "| # | BM | BK | TN | BN | NROW_LANES | Threads | NROW | Accs/T | LDS(KiB) | ubatch t/s (pp) | Status |"
        echo "|---|---:|---:|---:|---:|-----------:|--------:|-----:|-------:|---------:|------------------:|--------|"
    } >> "$LOG_FILE"

    for BM in "${BM_VALUES[@]}"; do
      for BK in "${BK_VALUES[@]}"; do
        for TN in "${TN_VALUES[@]}"; do
          for NROW_LANES in "${NROW_LANES_VALUES[@]}"; do
            COUNT=$((COUNT + 1))
            BN=$((64 * TN))
            NROW=$((BM / NROW_LANES))
            THREADS=$((64 * NROW_LANES))
            ACCS=$((NROW * TN))

            # ── Constraint: BM divisible by NROW_LANES ────────────────────
            if [ $((BM % NROW_LANES)) -ne 0 ]; then
              echo "[$COUNT] SKIP BM=$BM BK=$BK TN=$TN NL=$NROW_LANES (BM%NLR!=0)"
              echo "| $COUNT | $BM | $BK | $TN | $BN | $NROW_LANES | $THREADS | $NROW | $ACCS | — | — | skip:div |" >> "$LOG_FILE"
              SKIP=$((SKIP + 1))
              continue
            fi

            # ── Constraint: LDS < 64 KiB ──────────────────────────────────
            # sW_lo = BM*BK*16 (uint4, qs)  sW_hi/d = BM*BK*4 (float, d)  -> BM*BK*36
            # sX    = BN*(BK+1)*34  (block_q8_0 = 34 bytes, verified)
            LDS_TOTAL=$(( BM * BK * 36 + BN * (BK + 1) * 34 ))
            LDS_KIB=$(( (LDS_TOTAL + 1023) / 1024 ))
            if [ $LDS_TOTAL -gt 65536 ]; then
              echo "[$COUNT] SKIP BM=$BM BK=$BK TN=$TN NL=$NROW_LANES (LDS=${LDS_KIB}KiB > 64)"
              echo "| $COUNT | $BM | $BK | $TN | $BN | $NROW_LANES | $THREADS | $NROW | $ACCS | $LDS_KIB | — | skip:LDS |" >> "$LOG_FILE"
              SKIP=$((SKIP + 1))
              continue
            fi

            TAG="BM${BM}_BK${BK}_TN${TN}_NL${NROW_LANES}"
            echo ""
            echo "================================================================"
            echo "[$COUNT] $TAG  (BN=$BN, threads=$THREADS, NROW=$NROW, accs=$ACCS, LDS≈${LDS_KIB}KiB)  P=$P"
            echo "================================================================"

            # ── Patch source (string-anchored) ────────────────────────────
            patch_config "$BK" "$TN" "$BM" "$NROW_LANES"

            echo "--- patched #defines ---"
            grep -nE 'define MMQ_RP_Q8_(BK|TN|BM|BN|NROW_LANES)' "$SRC_FILE"
            echo "--- launch_bounds ---"
            grep '__launch_bounds__.*mmq_gemm_q8_0_repacked' "$SRC_FILE"
            echo "--- dispatch dim3 ---"
            grep 'mmq_gemm_q8_0_repacked<false' "$SRC_FILE" | head -1
            echo "------------------------------"

            # ── Build ──────────────────────────────────────────────────────
            echo ">>> Building $TAG (P=$P)..."
            if ! eval "$BUILD_CMD" > /tmp/build_${TAG}_P${P}.log 2>&1; then
              echo "[$COUNT] BUILD FAILED for $TAG (see /tmp/build_${TAG}_P${P}.log)"
              echo "| $COUNT | $BM | $BK | $TN | $BN | $NROW_LANES | $THREADS | $NROW | $ACCS | $LDS_KIB | — | FAIL:build |" >> "$LOG_FILE"
              FAIL=$((FAIL + 1))
              restore_source
              continue
            fi

            # ── Bench ──────────────────────────────────────────────────────
            echo ">>> Benchmarking $TAG (P=$P)..."
            OUTPUT=$(./build/bin/llama-bench "${BENCH_PARAMS[@]}" -p "$P" -n 0 2>&1)
            echo "$OUTPUT"

            # Save raw table for this config
            {
              echo "#### P=$P $TAG"
              echo '```'
              echo "$OUTPUT" | sed -n '/^|/p'
              echo '```'
              echo ""
            } >> "$RAW_FILE"

            # Extract per-ubatch pp t/s (positional, order matches -ub sweep)
            TS_TMP=$(mktemp /tmp/lb_ts.XXXXXX)
            echo "$OUTPUT" > "$TS_TMP"
            TS=$(extract_ts "$TS_TMP")
            rm -f "$TS_TMP"
            read -ra TSARR <<< "$TS"
            if [ "${#TSARR[@]}" -ne "${#UBATCH_LIST[@]}" ]; then
              STATUS="? (rows=${#TSARR[@]})"
            else
              STATUS="ok"
              PASS=$((PASS + 1))
            fi

            COMPACT=""
            for i in "${!UBATCH_LIST[@]}"; do
              v="${TSARR[$i]:-?}"
              COMPACT="${COMPACT} ${UBATCH_LIST[$i]}:${v}"
            done

            echo "| $COUNT | $BM | $BK | $TN | $BN | $NROW_LANES | $THREADS | $NROW | $ACCS | $LDS_KIB |$COMPACT | $STATUS |" >> "$LOG_FILE"
            echo "$TAG;$BM;$BK;$TN;$BN;$NROW_LANES;$TS" >> "$MATFILE"

            # Restore for next iteration (safety)
            restore_source
          done
        done
      done
    done

    # ── Per-prompt matrix: rows = configs, cols = ubatch ──────────────────
    {
        echo ""
        echo "### Matrix — config x ubatch pp t/s (Prompt = $P)"
        echo ""
        hdr="| config (BM/BK/TN/NL) |"
        for u in "${UBATCH_LIST[@]}"; do hdr="$hdr ub$u |"; done
        echo "$hdr"
        sep="|---|"
        for u in "${UBATCH_LIST[@]}"; do sep="$sep---|"; done
        echo "$sep"
        while IFS=';' read -r tag bm bk tn bn nrl ts; do
          [ -z "$tag" ] && continue
          row="| $tag ($bm/$bk/$tn/$nrl) |"
          read -ra A <<< "$ts"
          for v in "${A[@]}"; do row="$row $v |"; done
          echo "$row"
        done < "$MATFILE"
    } >> "$LOG_FILE"
done

# ── Final restore ─────────────────────────────────────────────────────────
restore_source
rm -f "${SRC_FILE}.orig" bench_tile_sweep_P*.mat

# ── Summary ───────────────────────────────────────────────────────────────
{
    echo ""
    echo "---"
    echo "## Summary"
    echo "- Tested: $COUNT combinations"
    echo "- Passed: $PASS"
    echo "- Failed (build): $FAIL"
    echo "- Skipped (constraints): $SKIP"
    echo ""
    echo "### Original config"
    echo "BM=128, BK=4, TN=2, BN=128, NROW_LANES=8, Threads=512"
    echo ""
    echo "### Constraint rules"
    echo "- BM must be divisible by NROW_LANES"
    echo "- LDS = BM*BK*36 (qs+d) + BN*(BK+1)*34 (block_q8_0) must fit < 64 KiB"
    echo ""
    echo "### Parameter meanings"
    echo "- **BM**: weight rows per block (output tile height)"
    echo "- **BK**: K-sub-blocks per LDS fill (inner loop unroll)"
    echo "- **TN**: activation columns per thread (BN = 64 x TN)"
    echo "- **NROW_LANES**: row lanes per block (blockDim.y; threads = 64 x NROW_LANES)"
    echo "- **NROW**: rows per lane = BM / NROW_LANES (weight row loop iters)"
    echo "- **Accs/T**: accumulators per thread = NROW x TN"
    echo "- **LDS(KiB)**: shared memory per workgroup"
    echo ""
    echo "### Output files"
    echo "- $LOG_FILE  : summary tables + per-prompt matrices"
    echo "- $RAW_FILE  : full raw llama-bench markdown tables per config"
} >> "$LOG_FILE"

echo ""
echo "================================================================"
echo "Sweep complete. Results: $LOG_FILE"
echo "  Passed: $PASS / $COUNT  (Failed: $FAIL, Skipped: $SKIP)"
echo "================================================================"
