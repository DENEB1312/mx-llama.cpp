#!/usr/bin/env python3
"""
reparse_sweep.py — regenerate bench_tile_sweep.md from bench_tile_sweep_raw.md.

The original sweep stored clean per-config llama-bench tables in
bench_tile_sweep_raw.md, but the live extraction split each "216.10 +- 0.00"
cell into 3 tokens (bash `read -ra`), corrupting the summary. The raw data
is intact, so this re-parses it correctly and rebuilds the summary tables +
per-prompt matrices. No GPU work required.
"""
import re, sys
from pathlib import Path

RAW = Path("bench_tile_sweep_raw.md")
OUT = Path("bench_tile_sweep.md")
UBATCH = [16, 32, 64, 128, 256, 512, 1024, 2048]
TAG_RE = re.compile(r'BM(\d+)_BK(\d+)_TN(\d+)_NL(\d+)')


def lds_kib(bm, bk, tn, nl):
    bn = 64 * tn
    lds = bm * bk * 36 + bn * (bk + 1) * 34
    return (lds + 1023) // 1024


def parse_block(table_lines):
    """Return list of 8 numeric t/s values in ubatch order, or None."""
    header_idx = None
    cols = None
    for i, l in enumerate(table_lines):
        if l.strip().startswith('|') and 't/s' in l:
            header_idx = i
            cols = [c.strip() for c in l.strip().strip('|').split('|')]
            break
    if header_idx is None or cols is None:
        return None
    ti = next((j for j, c in enumerate(cols) if c == 't/s'), len(cols) - 1)
    vals = []
    for l in table_lines[header_idx + 1:]:
        s = l.strip()
        if not s.startswith('|'):
            continue
        if set(s) <= set('|-: '):
            continue
        cells = [c.strip() for c in s.strip('|').split('|')]
        if ti < len(cells):
            m = re.search(r'-?\d+(?:\.\d+)?', cells[ti])
            if m:
                vals.append(float(m.group(0)))
    return vals if len(vals) == len(UBATCH) else None


def main():
    text = RAW.read_text()
    # Split into blocks: "#### P=VAL TAG" ... up to next "####" or EOF
    blocks = re.split(r'(?=^#### )', text, flags=re.M)
    # collect: prompts[P].append((tag, bm,bk,tn,nl, vals))
    prompts = {}
    for b in blocks:
        h = re.search(r'^####\s+P=(\d+)\s+(\S+)', b, re.M)
        if not h:
            continue
        P = int(h.group(1))
        tag = h.group(2)
        m = TAG_RE.search(tag)
        if not m:
            continue
        bm, bk, tn, nl = (int(x) for x in m.groups())
        # table is between ``` fences
        tbl = re.search(r'```\n(.*?)\n```', b, re.S)
        if not tbl:
            continue
        vals = parse_block(tbl.group(1).splitlines())
        if vals is None:
            continue
        prompts.setdefault(P, []).append((tag, bm, bk, tn, nl, vals))

    if not prompts:
        print("No valid blocks parsed from", RAW)
        sys.exit(1)

    out = []
    out.append("# Tile Parameter Sweep — (regenerated from raw by reparse_sweep.py)\n")
    out.append("")
    out.append("Model: Qwen3.5-4B-Q8_0.gguf   Bench: pp sweep, repack on (GGML_CUDA_REPACK_Q8_0=1)")
    out.append("ubatch: 16-2048*2   prompts: %s" % ", ".join(str(p) for p in sorted(prompts)))
    out.append("")

    for P in sorted(prompts):
        rows = prompts[P]
        out.append("")
        out.append("## Prompt = %d  (ubatch: %s)" % (P, " ".join(str(u) for u in UBATCH)))
        out.append("")
        out.append("| # | BM | BK | TN | BN | NROW_LANES | Threads | NROW | Accs/T | LDS(KiB) | ubatch t/s (pp) |")
        out.append("|---|---:|---:|---:|---:|-----------:|--------:|-----:|-------:|---------:|------------------:|")
        for i, (tag, bm, bk, tn, nl, vals) in enumerate(rows, 1):
            bn = 64 * tn
            nrow = bm // nl
            threads = 64 * nl
            accs = nrow * tn
            lk = lds_kib(bm, bk, tn, nl)
            compact = " ".join("%d:%.2f" % (UBATCH[j], vals[j]) for j in range(len(UBATCH)))
            out.append("| %d | %d | %d | %d | %d | %d | %d | %d | %d | %d |%s |" %
                       (i, bm, bk, tn, bn, nl, threads, nrow, accs, lk, compact))
        # matrix
        out.append("")
        out.append("### Matrix — config x ubatch pp t/s (Prompt = %d)" % P)
        out.append("")
        hdr = "| config (BM/BK/TN/NL) |" + "".join(" ub%d |" % u for u in UBATCH)
        out.append(hdr)
        out.append("|---|" + "---|" * len(UBATCH))
        for tag, bm, bk, tn, nl, vals in rows:
            row = "| %s (%d/%d/%d/%d) |" % (tag, bm, bk, tn, nl)
            row += "".join(" %.2f |" % vals[j] for j in range(len(UBATCH)))
            out.append(row)

    out.append("")
    out.append("---")
    out.append("## Summary")
    out.append("- Configs parsed: %d (per prompt: %s)" %
               (sum(len(v) for v in prompts.values()),
                ", ".join("%d:%d" % (P, len(prompts[P])) for P in sorted(prompts))))
    out.append("- Note: regenerated from bench_tile_sweep_raw.md (live extraction bug fixed).")
    out.append("- Original sweep ran only the prompts present in the raw file; re-run")
    out.append("  SCRIPT_llama_bench_TILE_SWEEP.sh to add missing prompts.")

    OUT.write_text("\n".join(out) + "\n")
    print("Wrote", OUT, "with prompts", sorted(prompts))


if __name__ == '__main__':
    main()
