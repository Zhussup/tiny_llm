#!/usr/bin/env python3
"""build_subset.py — one set of programs, two input formats, one token budget.

dataset/train.jsonl (gcc listings) and dataset/train_disasm.jsonl (objdump -d)
describe the same programs in different forms. Comparing adapters trained on
them only means something on programs whose every sample fits the same
max_seq_len in both forms: a disassembly costs about twice the tokens of the
listing, so at max_seq_len=2048 the disassembly adapter would simply see less
data, and the comparison would mix format with volume.

This script keeps the programs that appear in both corpora with the same
number of samples, every sample within --cap tokens measured exactly the way
train.py measures it (the chat template without a generation prompt). It then
writes one file per view plus a mixed file, so no option has to be added to
train.py:

  dataset/train_common4096_listing.jsonl   listings only
  dataset/train_common4096_disasm.jsonl    disassemblies only
  dataset/train_common4096_mixed.jsonl     both
  dataset/val_common4096_*.jsonl           the same three for validation

Every row keeps its original id (a1-… or d1-…) and its split, so a run's
checkpoints stay traceable back to the corpus.

Usage:
  python3 build_subset.py --cap 4096
  python3 build_subset.py --cap 2048 --dry-run
"""

import argparse
import json
import os
import sys
from collections import defaultdict

from transformers import AutoTokenizer

from train import apply_template, ids_of, BASE_DIR

DATASET_DIR = os.path.join(BASE_DIR, "dataset")

LISTING = "listing"
DISASM = "disasm"


def load_rows(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def chat_length(tokenizer, sample):
    """The same count train.py uses for --max-seq-len."""
    return len(ids_of(apply_template(tokenizer, sample["messages"], False)))


def group_by_source(rows, tokenizer, cap):
    """source_id -> (number of samples, all within cap)."""
    groups = defaultdict(lambda: [0, True])
    for row in rows:
        g = groups[row["source_id"]]
        g[0] += 1
        if chat_length(tokenizer, row) > cap:
            g[1] = False
    return groups


def common_sources(listing_rows, disasm_rows, tokenizer, cap):
    """Programs present in both views, equally represented, within the cap."""
    lg = group_by_source(listing_rows, tokenizer, cap)
    dg = group_by_source(disasm_rows, tokenizer, cap)
    keep = set()
    dropped = {"listing_only": 0, "disasm_only": 0, "count_mismatch": 0,
               "too_long": 0}
    for sid, (n, fits) in lg.items():
        if sid not in dg:
            dropped["listing_only"] += 1
            continue
        dn, dfits = dg[sid]
        if n != dn:
            dropped["count_mismatch"] += 1
            continue
        if not fits or not dfits:
            dropped["too_long"] += 1
            continue
        keep.add(sid)
    dropped["disasm_only"] = sum(1 for sid in dg if sid not in lg)
    return keep, dropped


def filter_rows(rows, keep):
    return [r for r in rows if r["source_id"] in keep]


def main():
    ap = argparse.ArgumentParser(
        description="Keep the programs that fit one token budget in both views")
    ap.add_argument("--cap", type=int, default=4096,
                    help="max_seq_len the subset is built for")
    ap.add_argument("--tokenizer", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--listing-train", default=os.path.join(DATASET_DIR, "train.jsonl"))
    ap.add_argument("--listing-val", default=os.path.join(DATASET_DIR, "val.jsonl"))
    ap.add_argument("--disasm-train", default=os.path.join(DATASET_DIR, "train_disasm.jsonl"))
    ap.add_argument("--disasm-val", default=os.path.join(DATASET_DIR, "val_disasm.jsonl"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    stats = {"cap": args.cap, "tokenizer": args.tokenizer, "splits": {}}
    for split, lp, dp in (("train", args.listing_train, args.disasm_train),
                          ("val", args.listing_val, args.disasm_val)):
        listing = load_rows(lp)
        disasm = load_rows(dp)
        print(f"{split}: {len(listing)} listings, {len(disasm)} disassemblies",
              file=sys.stderr)
        keep, dropped = common_sources(listing, disasm, tokenizer, args.cap)
        lrows = filter_rows(listing, keep)
        drows = filter_rows(disasm, keep)
        stats["splits"][split] = {
            "sources": len(keep), "listing_rows": len(lrows),
            "disasm_rows": len(drows), "dropped": dropped,
        }
        print(f"{split}: {len(keep)} common programs, "
              f"{len(lrows)} rows per view", file=sys.stderr)
        if args.dry_run:
            continue

        stem = os.path.join(DATASET_DIR, f"{split}_common{args.cap}")
        for name, rows in ((LISTING, lrows), (DISASM, drows),
                           ("mixed", lrows + drows)):
            path = f"{stem}_{name}.jsonl"
            tmp = path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                for row in rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)
            print(f"written: {path} ({len(rows)})", file=sys.stderr)

    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
