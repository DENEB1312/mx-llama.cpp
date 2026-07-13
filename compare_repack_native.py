#!/usr/bin/env python3
"""
compare_repack_native.py — isolate the Q8_0 MMQ kernel cost at small M.

Runs llama-bench (via SCRIPT_llama_bench.sh, repack ON vs OFF) under
rocprofv3 at a pinned M (= prompt == ubatch), then compares the two Q8_0
GEMM kernels directly:

  repack  -> mmq_gemm_q8_0_repacked      (repack-gcn.cu:1781)
  native  -> mul_mat_q<Q8_0>             (mmq.cuh mul_mat_q_case)

It reports per-call GPU time (the real "kernel efficiency" signal, free of
the one-time repack transform which only happens at weight upload) and, with
--counters, hardware counters (occupancy / LDS / instruction mix) so we can
tell *why* repack loses at small M.

The transform is NOT measured here on purpose: repack_q*_host runs only in
set_tensor (model load), so it is amortized and cannot explain a per-token
small-batch gap. See repack-gcn.cu:2411-2500.
"""
import argparse, subprocess, re, csv, os, sys
from pathlib import Path

REPACK_SCRIPT = "./SCRIPT_llama_bench.sh"
MODEL_DEFAULT = "/media/iacopo/LLMs/llms/Qwen_Qwen3.5-4B-Q8_0.gguf"

# gfx906: 64 CUs * 4 SIMDs * 16 waves = 4096 max concurrent waves.
MAX_WAVES = 64 * 4 * 16

GFX906_ENV = {
    'HSA_OVERRIDE_GFX_VERSION': '9.0.6', 'HIP_VISIBLE_DEVICES': '0',
    'ROCR_VISIBLE_DEVICES': '0',
    'GGML_BACKEND_HIP': '1', 'HCC_AMDGPU_TARGET': 'gfx906',
    'GGML_CUDA_DISABLE_GRAPHS': '1',  # clean per-kernel traces
}

# Curated gfx906 PMC set (occupancy + LDS + instruction mix). If any counter
# is unavailable rocprofv3 fails the job; the caller guards that.
COUNTERS = ",".join([
    "SQ_WAVES", "SQ_INSTS_VALU", "SQ_INSTS_SALU", "SQ_INSTS_VMEM",
    "SQ_INSTS_LDS", "SQ_INSTS_FLAT", "SQ_INSTS_EXPORT",
    "LDS_READS", "LDS_WRITES", "GRBM_GUI_ACTIVE",
])


def run_rocprofv3(mode, m, out_dir, name, counters=False,
                  model=None, timeout=1200):
    env = os.environ.copy()
    env.update(GFX906_ENV)
    env['GGML_CUDA_REPACK'] = '1' if mode == 'repack' else '0'
    env['GGML_CUDA_REPACK_Q8_0'] = '1' if mode == 'repack' else '0'
    env['BENCH_TESTS_OVERRIDE'] = f"-p {m} -n 0"
    env['UBATCH_OVERRIDE'] = f"-ub {m}"
    if model:
        env['MODEL_OVERRIDE'] = model
    cmd = ["rocprofv3", "--kernel-trace", "--stats",
           "--output-format", "csv", "-d", str(out_dir), "-o", name]
    if counters:
        cmd += ["--pmc", COUNTERS]
    cmd += ["--", REPACK_SCRIPT, mode]
    print(f"\n### rocprofv3 [{mode}] M={m} counters={counters}\n  {' '.join(cmd)}")
    rc = subprocess.run(cmd, env=env, timeout=timeout)
    return rc.returncode


def read_stats_csv(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            if not r.get('Name'):
                continue
            try:
                calls = int(r.get('Calls', 1))
                total = int(r.get('TotalDurationNs', 0))
            except ValueError:
                continue
            rows.append({'name': r['Name'], 'calls': calls, 'total_ns': total})
    return rows


def pick_q8_kernels(rows, mode):
    """Return the Q8_0 MMQ kernels for a given mode."""
    out = []
    for k in rows:
        n = k['name']
        if mode == 'repack':
            if 'mmq_gemm_q8_0_repacked' in n:
                out.append(k)
        else:  # native
            if 'mul_mat_q<' in n and 'vec' not in n and 'ggml_type)8' in n:
                out.append(k)
    return out


def summarize(kernels):
    if not kernels:
        return None
    calls = sum(k['calls'] for k in kernels)
    total = sum(k['total_ns'] for k in kernels)
    per = (total / calls) if calls else 0
    names = " | ".join(sorted({k['name'].split('<')[0].replace('void ', '')
                               for k in kernels}))
    return {'calls': calls, 'total_ns': total, 'per_ns': per,
            'names': names}


def load_counter_csvs(out_dir):
    """Return list of (path, rows) for every csv that has a Name column."""
    res = []
    for p in out_dir.glob("*.csv"):
        try:
            with open(p) as f:
                r = csv.DictReader(f)
                if 'Name' in (r.fieldnames or []):
                    res.append((p, list(r)))
        except Exception:
            continue
    return res


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--ms', default="16",
                   help="comma list of M values (prompt=ubatch), default 16")
    p.add_argument('--counters', action='store_true',
                   help="also collect gfx906 PMC counters (occupancy/LDS/inst mix)")
    p.add_argument('-o', '--out', default="./cmp_repack_native")
    p.add_argument('--model', default=MODEL_DEFAULT)
    p.add_argument('--timeout', type=int, default=1200)
    args = p.parse_args()

    ms = [int(x) for x in args.ms.split(',') if x.strip()]
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}  # m -> {repack: summary, native: summary}
    counter_files = []

    for m in ms:
        results[m] = {}
        for mode in ('repack', 'native'):
            tag = f"{mode}_M{m}"
            rc = run_rocprofv3(mode, m, out_dir, tag,
                               counters=False, model=args.model,
                               timeout=args.timeout)
            if rc != 0:
                print(f"!! rocprofv3 failed (rc={rc}) for [{mode}] M={m}")
                results[m][mode] = None
                continue
            stats = out_dir.glob(f"{tag}_kernel_stats.csv")
            try:
                path = next(stats)
            except StopIteration:
                # rocprofv3 may name it differently; search broadly
                path = next(out_dir.glob("*kernel_stats.csv"))
            rows = read_stats_csv(path)
            k = pick_q8_kernels(rows, mode)
            results[m][mode] = summarize(k)
            if args.counters:
                # Separate -o name so the PMC pass does not overwrite the
                # timing kernel_stats.csv produced above.
                run_rocprofv3(mode, m, out_dir, f"{tag}_pmc",
                              counters=True, model=args.model,
                              timeout=args.timeout)
                counter_files.append((m, mode, path))

    # ── Print comparison ──────────────────────────────────────────────────
    print("\n" + "=" * 92)
    print("Q8_0 MMQ KERNEL COMPARISON  (repack vs native, per-call GPU time)")
    print("=" * 92)
    for m in ms:
        rp, nv = results[m].get('repack'), results[m].get('native')
        print(f"\n--- M = {m}  (prompt=ubatch={m}) ---")
        if not rp or not nv:
            print("   (missing data — see rocprofv3 errors above)")
            continue
        print(f"   {'kernel':<34} {'mode':<8} {'calls':>7} {'total ms':>10} {'per-call µs':>12}")
        print(f"   {rp['names'][:33]:<34} {'repack':<8} {rp['calls']:>7} "
              f"{rp['total_ns']/1e6:>10.2f} {rp['per_ns']/1e3:>12.2f}")
        print(f"   {nv['names'][:33]:<34} {'native':<8} {nv['calls']:>7} "
              f"{nv['total_ns']/1e6:>10.2f} {nv['per_ns']/1e3:>12.2f}")
        if nv['per_ns'] > 0:
            speedup = rp['per_ns'] / nv['per_ns']
            verdict = "repack SLOWER" if speedup > 1 else "repack faster"
            print(f"   => repack is {speedup:.2f}x {verdict} per call "
                  f"({'BAD' if speedup>1 else 'good'})")

    # ── Counters ──────────────────────────────────────────────────────────
    if args.counters:
        print("\n" + "=" * 92)
        print("HARDWARE COUNTERS (occupancy / LDS / instruction mix)")
        print("=" * 92)
        for m, mode, path in counter_files:
            rp = results[m].get(mode)
            if not rp:
                continue
            name_sub = 'mmq_gemm_q8_0_repacked' if mode == 'repack' else 'mul_mat_q<'
            print(f"\n--- M={m} [{mode}]  target contains '{name_sub}' ---")
            found = False
            for cp, rows in load_counter_csvs(out_dir):
                for r in rows:
                    if name_sub in r.get('Name', ''):
                        found = True
                        cols = [c for c in r.keys() if c != 'Name']
                        line = "  " + "  ".join(f"{c}={r[c]}" for c in cols)
                        print(f"  {r['Name'].split('<')[0][:28]:<29}{line}")
                        # occupancy estimate from SQ_WAVES if present
                        if 'SQ_WAVES' in r and rp['calls']:
                            waves = float(r['SQ_WAVES']) / rp['calls']
                            occ = waves / MAX_WAVES * 100
                            print(f"     -> waves/launch≈{waves:.1f}  "
                                  f"occupancy_est≈{occ:.2f}% (of {MAX_WAVES} max waves)")
            if not found:
                print("   (no counter rows matched — raw csv saved in output dir)")

    print(f"\nRaw rocprofv3 output: {out_dir}")


if __name__ == '__main__':
    try:
        main()
    except subprocess.TimeoutExpired:
        print("TIMEOUT — increase --timeout", file=sys.stderr)
        sys.exit(1)
