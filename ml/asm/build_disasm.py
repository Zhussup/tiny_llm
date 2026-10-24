#!/usr/bin/env python3
"""build_disasm.py — the same A1 programs rendered as objdump -d output.

The listing corpus that build_dataset.py produces comes from `gcc -S`: symbolic
labels, no addresses, no instruction bytes. A company has a binary, so its
assembly comes from a disassembler. This script builds that second view of the
same programs: every record in raw_c/ already stores the C source, so the
corpus is a re-derivation, not a new collection.

Input:  ml/asm/raw_c/shard_*.jsonl
Output: ml/asm/dataset/train_disasm.jsonl + val_disasm.jsonl, same chat format,
        with objdump -d output as the user message.

The output is left exactly as `objdump -d` prints it: address column, hex
bytes, `<symbol@plt>` targets. Normalizing any of that away would test a
format nobody actually reads.

One thing is filtered: only functions defined in the program's own source are
kept. A linked binary also contains the C runtime (`_start`, `frame_dummy`,
`register_tm_clones`), which has nothing to do with the program and would
teach the model to reconstruct libc startup code.

The train/val split is the same hash of source_id that build_dataset.py uses,
so a program lands on the same side of both corpora and the two adapters can
be compared on identical programs.

Usage:
  python3 build_disasm.py --limit 20 --jobs 4   smoke test, writes dataset/
  python3 build_disasm.py --dry-run             statistics only
  python3 build_disasm.py --jobs 8              full build
"""

import argparse
import glob
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor

from build_dataset import is_val, write_jsonl

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(BASE_DIR, "raw_c")
OUT_DIR = os.path.join(BASE_DIR, "dataset")

DISASM_SYSTEM_PROMPT = (
    "You are an expert reverse engineer. You are given the x86-64 disassembly "
    "of a single-file C program, produced by objdump -d in AT&T syntax: an "
    "address column, the instruction bytes, and the instruction with resolved "
    "symbol names. Reconstruct a C program with the same observable behavior: "
    "it reads from stdin and writes its result to stdout. "
    "Reply with exactly one fenced code block that starts with ```c and "
    "nothing else."
)

# "0000000000401136 <main>:" — a function header in objdump output
FUNC_RE = re.compile(r"^([0-9a-f]+) <([^>]+)>:$")

GCC_TIMEOUT = 30
MAX_C_BYTES = 200_000   # a runaway answer should not be compiled


def disasm_user_message(text, opt):
    return (
        f"Optimization level: -{opt}\n\n"
        "Disassembly (objdump -d, AT&T syntax):\n"
        f"```asm\n{text}\n```"
    )


def assistant_message(c_text):
    return f"```c\n{c_text}\n```"


def run(cmd, cwd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True,
                              timeout=GCC_TIMEOUT, cwd=cwd)
    except subprocess.TimeoutExpired:
        return None
    except OSError:
        return None


def user_symbols(obj_path, cwd):
    """Function names defined by the program itself (nm type T/t)."""
    p = run(["nm", "--defined-only", obj_path], cwd)
    if p is None or p.returncode != 0:
        return set()
    names = set()
    for line in p.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 3 and parts[1] in ("T", "t"):
            names.add(parts[2])
    return names


def keep_user_functions(text, keep):
    """Drops every function block whose name the program did not define."""
    out, current = [], None
    for line in text.splitlines():
        m = FUNC_RE.match(line.strip())
        if m:
            current = m.group(2)
            if current in keep:
                out.append(line)
            continue
        if current in keep and line.strip():
            out.append(line)
    return "\n".join(out)


def disassemble(c_text, opt):
    """C text -> (ok, disassembly or diagnostic)."""
    with tempfile.TemporaryDirectory(prefix="gendis_") as workdir:
        with open(os.path.join(workdir, "prog.c"), "w", encoding="utf-8") as f:
            f.write(c_text)

        src = ["gcc", "-std=c11", "-fno-asynchronous-unwind-tables",
               f"-{opt}", "-c", "prog.c", "-o", "prog.o"]
        p = run(src, workdir)
        if p is None or p.returncode != 0:
            return False, (p.stderr.strip()[:2000] if p else "gcc timed out")

        keep = user_symbols("prog.o", workdir)
        if not keep:
            return False, "no user-defined functions in the object file"

        # -no-pie keeps addresses in the familiar 0x401000 range instead of
        # the position-independent 0x1139 style; -lm because some answers use
        # math.h, which compiles fine with -S but fails to link without it
        link = ["gcc", "-no-pie", "-o", "prog.x", "prog.o", "-lm"]
        p = run(link, workdir)
        if p is None or p.returncode != 0:
            return False, (p.stderr.strip()[:2000] if p else "link timed out")

        p = run(["objdump", "-d", "-j", ".text", "prog.x"], workdir)
        if p is None or p.returncode != 0:
            return False, (p.stderr.strip()[:2000] if p else "objdump timed out")

        text = keep_user_functions(p.stdout, keep)
        if not text.strip():
            return False, "empty disassembly after filtering"
        return True, text


def light_records(limit=None):
    """Only the fields needed here: the asm listings can be tens of MB."""
    out = []
    for path in sorted(glob.glob(os.path.join(RAW_DIR, "shard_*.jsonl"))):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                out.append({
                    "id": rec.get("id"),
                    "model": rec.get("model") or "unknown",
                    "compile_o0_ok": bool(rec.get("compile_o0_ok")),
                    "compile_o2_ok": bool(rec.get("compile_o2_ok")),
                    "c_source": rec.get("c_source"),
                })
                if limit and len(out) >= limit:
                    return out
    return out


def build_one(rec):
    """One collected record -> up to two samples (O0 and O2)."""
    stats = {"samples": 0, "skipped_not_compiled": 0, "skipped_link": 0,
             "skipped_dup": 0, "skipped_c": 0, "by_opt": {"O0": 0, "O2": 0},
             "by_model": {}}
    c_text = (rec.get("c_source") or "").strip()
    source_id = rec.get("id")
    samples = []
    if not c_text:
        stats["skipped_c"] += 1
        return samples, stats
    if len(c_text) > MAX_C_BYTES:
        stats["skipped_c"] += 1
        return samples, stats

    for opt in ("O0", "O2"):
        if not rec.get(f"compile_{opt.lower()}_ok"):
            stats["skipped_not_compiled"] += 1
            continue
        ok, text = disassemble(c_text, opt)
        if not ok:
            stats["skipped_link"] += 1
            continue
        model = rec["model"]
        stats["by_model"][model] = stats["by_model"].get(model, 0) + 1
        stats["by_opt"][opt] += 1
        stats["samples"] += 1
        samples.append({
            "id": f"d1-{source_id}-{opt}",
            "source_id": source_id,
            "opt": opt,
            "model": model,
            "messages": [
                {"role": "system", "content": DISASM_SYSTEM_PROMPT},
                {"role": "user", "content": disasm_user_message(text, opt)},
                {"role": "assistant", "content": assistant_message(c_text)},
            ],
        })
    return samples, stats


def merge(total, part):
    for k, v in part.items():
        if isinstance(v, dict):
            for kk, vv in v.items():
                total[k][kk] = total[k].get(kk, 0) + vv
        else:
            total[k] += v
    return total


def main():
    ap = argparse.ArgumentParser(
        description="Build the objdump -d view of the A1 corpus")
    ap.add_argument("--val-ratio", type=float, default=0.02)
    ap.add_argument("--limit", type=int, default=0,
                    help="only the first N records (smoke test)")
    ap.add_argument("--jobs", type=int, default=8,
                    help="parallel gcc/objdump workers")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    records = light_records(args.limit or None)
    print(f"records: {len(records)}", file=sys.stderr)

    stats = {"samples": 0, "skipped_not_compiled": 0, "skipped_link": 0,
             "skipped_dup": 0, "skipped_dup_conflict": 0, "skipped_c": 0,
             "by_opt": {"O0": 0, "O2": 0},
             "by_model": {}}
    results = [None] * len(records)

    def work(i):
        return build_one(records[i])

    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        for i, (samples, part) in enumerate(pool.map(work, range(len(records)))):
            results[i] = samples
            merge(stats, part)
            if (i + 1) % 250 == 0:
                print(f"  {i + 1}/{len(records)} records, "
                      f"{stats['samples']} samples", file=sys.stderr)

    seen, targets, train, val = set(), {}, [], []
    for samples in results:
        for sample in samples:
            key = hashlib.sha1(sample["messages"][1]["content"].encode()).hexdigest()
            tgt = hashlib.sha1(sample["messages"][2]["content"].encode()).hexdigest()
            if key in seen:
                stats["skipped_dup"] += 1
                # A disassembly carries no local names, so two different C
                # programs can disassemble to the same text. The first target
                # is kept, like build_dataset.py does; the count says how
                # often the input is ambiguous.
                if targets[key] != tgt:
                    stats["skipped_dup_conflict"] += 1
                continue
            seen.add(key)
            targets[key] = tgt
            if is_val(sample["source_id"], args.val_ratio):
                val.append(sample)
            else:
                train.append(sample)
    stats["train"], stats["val"] = len(train), len(val)

    print(json.dumps(stats, ensure_ascii=False, indent=2))
    if args.dry_run or not stats["samples"]:
        return

    os.makedirs(OUT_DIR, exist_ok=True)
    write_jsonl(os.path.join(OUT_DIR, "train_disasm.jsonl"), train)
    write_jsonl(os.path.join(OUT_DIR, "val_disasm.jsonl"), val)
    with open(os.path.join(OUT_DIR, "stats_disasm.json"), "w",
              encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"written: {OUT_DIR}/train_disasm.jsonl ({stats['train']}), "
          f"val_disasm.jsonl ({stats['val']})")


if __name__ == "__main__":
    main()
