# ml/asm — x86-64 asm → C dataset pipeline

Narrow-domain dataset for a small programming model working with x86-64
assembly in AT&T syntax (the syntax `gcc -S` emits).

The current goal is **asm → C reconstruction**: the model is given a compiler
listing and must reconstruct C code with the same observable behavior.
Correctness is checked mechanically by **differential testing**, not by string
comparison — any C program that behaves the same as the original is a correct
answer.

## Two data lines, kept separate

| Line | What it produces | How correctness is established | File |
|------|------------------|-------------------------------|------|
| A1 (current) | task → C program → two gcc listings (`-O0`, `-O2`) | ground truth by construction: the asm was made by the compiler | `gen_c.py` |
| B (planned) | task → freestanding asm program (`_start`, syscalls only) | test run via `verifier.py` | — |

They are not mixed into one dataset: A1 is hosted code (`main`, libc, PLT),
B is freestanding code (`_start`, no libc). The same question would have two
contradictory correct answers, which turns into label noise for a small model.

## Pipeline

```
gen_c.py ──► raw_c/shard_*.jsonl ──┬─► build_dataset.py ──► dataset/train.jsonl
                                   │                        dataset/val.jsonl
                                   └─► build_disasm.py ───► dataset/train_disasm.jsonl
                                                            dataset/val_disasm.jsonl
                                                                    │
                    build_subset.py ◄───────────────────────────────┘
                    (both views, one token budget)
                          │
                          └─► dataset/*_common<N>_{listing,disasm,mixed}.jsonl

model answer (C) ──► difftest.py: original C vs predicted C ──► verdict
```

## Files

- `gen_c.py` — collector for line A1: asks the model named in `config.json`
  (`beam:cloud` by default) for a C program, compiles it with gcc at `-O0` and
  `-O2`, stores both listings.
- `config.json` — settings for the collector: `model` (the teacher) and
  `parallel` (requests in flight at once, 1 to 3, default 3). A value out of
  range stops the collector with a message instead of being quietly clamped.
- `collector.sh` — supervisor for `gen_c.py`: restarts it whenever it exits, for
  any reason (uncaught exception, killed process, Ollama gone after a shutdown),
  with a `flock` guard so a second copy cannot start.
- `build_dataset.py` — turns raw shards into chat-format JSONL pairs
  (listing → original C), with leak stripping, deduplication and a
  train/val split.
- `build_disasm.py` — the same records as `objdump -d` sees them: compiles the
  stored C, links it, disassembles `.text` and keeps only the functions the
  program itself defines (`nm --defined-only`). Writes
  `dataset/train_disasm.jsonl` / `val_disasm.jsonl` with the same ids, split
  and chat format, `d1-…` instead of `a1-…`.
- `build_subset.py` — keeps the programs whose every sample fits one token
  budget in both views, so adapters trained on listings and on disassembly can
  be compared on identical programs. Writes `dataset/{train,val}_common<N>_
  {listing,disasm,mixed}.jsonl`.
- `gen_cases.py` — builds per-program differential probes: tries a pool of 46
  stdin inputs against each reference program, keeps up to six that it actually
  reacts to, writes them to `dataset/cases.jsonl`.
- `difftest.py` — differential tester: compiles the reference C and a
  predicted C, runs both on the same stdin inputs, compares stdout and exit
  codes.
- `eval.py` — evaluates a local model (base or LoRA-tuned) on `val.jsonl`:
  greedy generation, C extraction, `difftest` check, `compile_rate` and
  `pass_rate`; appends predictions and resumes an interrupted run.
- `train.py` — LoRA SFT of Qwen3-0.6B on `train.jsonl`. The loss is computed
  only on the assistant's C answer: prompt and listing tokens are masked with
  `-100`, so the model learns reconstruction instead of copying the prompt.
  Pairs longer than `--max-seq-len` are skipped, not truncated.
- `plot_training.py` — renders the training-loss figure in `docs/` (light and
  dark PNG) from `train_state.json` or directly from a training log.
- `plot_eval.py` — renders the evaluation figures (`eval_full118`,
  `eval_epochs`, `eval_probes`) from `eval_results/`.
- `verifier.py` — runner/checker for line B asm programs: x86-64, AT&T syntax,
  entry point `_start`, no libc, I/O only through `read(0)` / `write(1)`
  syscalls. Exit codes: `0` all tests pass, `1` assembled but tests failed,
  `2` assembly/link failure, `3` bad input.
- `tests/` — self-tests for the verifier (`hello.s`, `echo.s`, `crash.s`,
  `infinite.s`).
- `raw_c/` — output shards of `gen_c.py` (runtime data).
- `dataset/` — built `train.jsonl`, `val.jsonl`, `stats.json`, their
  disassembly counterparts `train_disasm.jsonl`, `val_disasm.jsonl`,
  `stats_disasm.json`, the token-matched `*_common4096_*` subsets, the
  per-program probes `cases.jsonl` / `cases_val.jsonl`, and the frozen
  validation snapshot `val118.jsonl` that every number in this README was
  measured on (runtime data).
- `checkpoints/` — LoRA adapters and `train_state.json` (runtime data).
- `eval_results/` — `predictions_*.jsonl` and `summary_*.json` per evaluation
  run: one row per validation pair, every answer with its compile and
  differential-test verdict.
- `docs/` — figures used by this README.
- `state_gen_c.json`, `gen_c.log`, `gen_c.out`, `gen_c.pid` — checkpoint, log,
  wrapper output and pid file of the running collector (runtime data).

## One sample of `gen_c.py`, step by step

1. `index → (task, wording variation)` is deterministic, same scheme as
   `dset/collector.py`: 406 tasks × 32 variations = **12,992 combinations**.
   Once the space is exhausted the prompts repeat, but the answers differ
   (the temperature is random on every request).
2. A request goes to `/api/chat` with the model from `config.json`. The system
   prompt demands exactly one ` ```c ` block.
3. The C code is extracted from the response: first a block tagged `c` is
   looked for; if there is none, any block is accepted; if there are several,
   the longest one wins.
4. gcc compiles the code **twice** — with `-O0` and `-O2` — producing two
   listings of the same program. The difference between them (how the
   optimizer rewrites the code) is a valuable part of the dataset in itself.
5. The JSONL record is appended with flush + fsync (it survives a laptop
   shutdown), and the checkpoint `state_gen_c.json` is written atomically
   (temp file + `os.replace` + directory fsync).

### Record fields

`id, ts, model, gcc, task, variation, temperature, prompt, extract_ok,
c_source, prompt_tokens, compile_o0_ok, asm_o0, diag_o0, compile_o2_ok,
asm_o2, diag_o2`

`diag_*` holds gcc warnings/errors: they are data about the quality of the
generated C, not a filter. Records are kept even when gcc fails (the `diag_*`
fields explain why); `compile_*_ok` is the only quality label in line A1.

### Listing cleanup

Compiler service lines (`.file`, `.ident`), header comments (`# GNU C11 …`,
`# options passed …`) and empty linker note sections (`.section .note…`) are
stripped for readability. Everything else is kept, including `# prog.c:7: …`
comments that tie each instruction back to its source line
(`-fverbose-asm`). The gcc version is recorded in the `gcc` field because the
header is removed.

## Dataset format (`build_dataset.py`)

Each sample is a chat record:

```json
{
  "id": "a1-17-O2",
  "source_id": 17,
  "opt": "O2",
  "model": "beam:cloud",
  "messages": [
    {"role": "system", "content": "You are an expert reverse engineer. …"},
    {"role": "user", "content": "Optimization level: -O2\n\nAssembly listing:\n```asm\n…\n```"},
    {"role": "assistant", "content": "```c\n…\n```"}
  ]
}
```

- The reference record produces **two samples** (`-O0` and `-O2`). The
  disassembly corpus follows the same format with `d1-` ids and
  `dataset/train_disasm.jsonl`; only the user message and the system prompt
  differ.
- **Teacher provenance:** every sample carries the `model` that wrote its C,
  copied from the record in `raw_c/` (`"unknown"` for shards collected before
  the field existed). The collector reads its settings from `config.json`:
  `model` is the active teacher, `parallel` is how many requests may be in
  flight at once (1 to 3, default 3). Switching teachers or widening the
  pipeline needs no code change. Two teachers can
  answer the same task at different lengths and in different styles, and a
  run that trains on them needs to see that split to balance or filter by it
  without joining back to `raw_c/`. `stats.json` reports the counts as
  `by_model`.
- **Leak stripping:** `-fverbose-asm` embeds the original source text in the
  listing as `# prog.c:7: …` comments. If kept, reconstruction would be pure
  copying. `build_dataset.py` removes exactly those lines
  (`SRC_LINE_RE`); other gcc comments (operand names such as `#, f`) stay.
  Raw shards keep the full listing, so "explain this listing" tasks can still
  use them.
- **Split by source record:** both listings of one C program carry the same
  `source_id`; the train/val decision is hashed from that id, so the two
  variants always land in the same split. Default val ratio is 2%.
- Identical listings are deduplicated by SHA-1 of the cleaned asm.
- Output files are written atomically (`.tmp` + `os.replace`), and a
  half-written last line in a shard being appended to right now is skipped,
  so rebuilding while collection is running is safe.

## Disassembly view (`build_disasm.py`, `build_subset.py`)

The listing is what a compiler prints; a real target is a binary, and its
assembly comes out of `objdump -d`. `build_disasm.py` builds that second view
of the same programs without collecting anything new: every record keeps its
`c_source`, which is compiled again — this time into an executable — and
disassembled.

What the pipeline does per record, per optimization level:

1. `gcc -std=c11 -fno-asynchronous-unwind-tables -O{0,2} -c` to get the
   object file;
2. `nm --defined-only prog.o` to collect the function names the program
   defines (symbol types `T`/`t`);
3. `gcc -no-pie -o prog.x prog.o -lm` to link — `-no-pie` keeps addresses in
   the `0x401000` range that objdump output usually shows, `-lm` because some
   answers use `math.h`, which compiles fine but fails to link without it;
4. `objdump -d -j .text prog.x`, then every function block whose name is not
   in the `nm` set is dropped.

Step 4 matters: a linked binary also contains the C runtime — `_start`,
`_init`, `frame_dummy`, `register_tm_clones`, `_dl_relocate_static_pie` — and
those functions have nothing to do with the program. A hardcoded blocklist
misses version-specific names (`_dl_relocate_static_pie` slipped through in the
pilot), so the filter is derived from the source's own symbols.

Nothing else is normalized. The output is exactly what objdump prints: address
column, hex bytes, `<symbol@plt>` targets, RIP-relative comments. A normalized
view would test a format nobody actually reads.

Two consequences worth knowing:

- **The view is lossy.** Names, indentation and `puts` versus `printf` do not
  survive compilation, so two different (behaviorally identical) C programs can
  produce the same disassembly. 1,530 of 25,758 pairs were such collisions;
  the first target is kept and the count is `skipped_dup_conflict` in
  `stats_disasm.json`.
- **The view is twice as expensive in tokens.** Hex bytes and addresses
  tokenize badly: the median pair is 4,475 tokens against 2,259 for the
  listing, and 13.8% of disassembly pairs fit 2,048 tokens against 44.7% of
  listings (45.0% / 79.5% at 4,096).

Because of the second point, a comparison at a fixed `--max-seq-len` would
measure format *and* volume at once. `build_subset.py` removes that
confound: a program enters the subset only if it is present in both corpora
with the same number of samples and every sample fits the cap, measured with
the same chat template `train.py` uses. At `--cap 4096`: 3,876 programs, 7,680
rows per view, 172 validation pairs. Three files per split — `listing`,
`disasm`, `mixed` — so `train.py --train …` needs no change.

```bash
python3 build_disasm.py --jobs 8          # ~5 min, writes dataset/*_disasm*
python3 build_disasm.py --limit 20 --jobs 4   # smoke test
python3 build_subset.py --cap 4096        # writes dataset/*_common4096_*
python3 build_subset.py --cap 2048 --dry-run
```

## Differential testing (`difftest.py`)

The reference C and the candidate C are compiled with the same compiler
(`gcc -std=c11 -O0 -w`), then run on the same stdin inputs. For every input:

- stdout must match after ignoring trailing newlines;
- exit codes must match (negative values are signals, compared exactly).

Rules for edge cases:

- both sides time out → treated as a match (identical behavior);
- only one side times out → mismatch;
- a signal (SIGSEGV, SIGKILL, …) is compared as is;
- if the **reference** itself hits our resource limits (timeout, SIGKILL from
  `RLIMIT_CPU`, any signal), the input is marked inconclusive and excluded
  from the score: reproducing "the reference was killed by a limit" is not a
  meaningful target;
- the reference failing to compile is an input error (exit `3`), not a model
  failure.

Run isolation (wall timeout, `RLIMIT_AS`/`RLIMIT_CPU`/`RLIMIT_CORE`/
`RLIMIT_NPROC`, empty environment) is reused from `verifier.py`.

Default inputs cover typical task shapes: empty input, one integer, two
integers, `n` followed by an array, a line of text, multiple lines. Custom
inputs can be supplied as `[{"name": "...", "stdin": "..."}, …]`; expected
outputs are always taken from the reference, never from the file.

CLI exit codes: `0` all inputs matched, `1` behavior mismatch,
`2` candidate failed to compile, `3` bad input.

## Per-program probes (`gen_cases.py`)

One fixed probe list for every program measures the wrong thing. Ten global
inputs cannot tell a correct answer from `int main(void){ return 0; }` when the
program prints nothing on those inputs, and they never feed a matrix program the
shape it expects.

`gen_cases.py` fixes this at the source. For each reference program it runs a
pool of 46 stdin shapes (single integers, pairs, `n` followed by an array,
matrices, text lines, blank lines), and keeps the inputs the reference actually
reacts to: clean exit, non-empty stdout, no timeout, and a stdout different from
every input kept before it. Up to six are kept, longest output first.

```
{"source_id": 46, "informative": true, "n_probes": 4,
 "cases": [{"name": "n + array", "stdin": "5\n3 1 4 1 5\n"}, ...]}
```

Selection is deterministic: same program, same probes. 4,459 of 4,740 collected
programs got at least one probe; the remaining 281 read nothing or print
nothing and are recorded with `"informative": false`, so `eval.py` can exclude
them instead of scoring them as passes.

The selection is not free of bias: a program keeps one to six probes, and a
program with one probe is easier to pass. Reporting the rate only over pairs with
two or more probes is the conservative comparison.

## Evaluation (`eval.py`)

Generation is greedy (`do_sample=False`), so the answer to a given prompt is
deterministic and two runs differ only in the model.

Results are appended to `eval_results/predictions_<tag>__<val>__<probes>.jsonl`,
one line per pair, flushed and fsynced. The name contains no timestamp and a
hash of the probe set, which gives three properties:

- a rerun with the same arguments reads the file, skips the pairs already in it
  and continues the rest — a shutdown costs one pair, not the run;
- a changed `--cases`/`--cases-by-source` lands in a different file instead of
  mixing two measurements;
- a line cut in half by a kill is dropped when the file is read, and the file is
  rewritten without it before appending.

The summary is rebuilt from the file every ten pairs, so even an interrupted run
has correct numbers for the pairs it finished. `compile_rate` and `pass_rate`
cover every record in the file; `pass_rate_informative` excludes pairs whose
program has no probe at all.

## Training

`train.py` fine-tunes Qwen3-0.6B with a LoRA adapter on `dataset/train.jsonl`.
Only the adapter is trained (10.1M of 606M parameters, 1.67%); the base weights
stay frozen in bf16. The loss is computed only on the assistant's C answer:
labels for the system prompt, the user question and the assembly listing are
set to `-100`, which cross-entropy ignores.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/training_loss_full_dark.png">
  <img src="docs/training_loss_full_light.png"
       alt="Full-run training loss on assistant tokens: 0.81 at the first step, 0.06 at the last, over 182 optimizer steps"
       width="840">
</picture>

The figure is the **first full run**: 723 pairs (the pairs that fit 2,048
tokens), 2 epochs, 182 optimizer steps. Loss falls from 0.81 at the first step
to 0.06 at the last (epoch averages: 0.32 and 0.14). `plot_training.py` renders
the figure with **Matplotlib** (3.11.2) from `train_state.json` or a training
log and writes the light and dark PNGs to `docs/`.

The **second full run** (`--tag night-20261022`) used 2,579 pairs, 3 epochs and
969 optimizer steps over 2h15m on the RTX 3050; loss went from 0.60 at the first
step to 0.015 at the last (epoch averages: 0.229, 0.103, 0.057). Its adapter
lives in `checkpoints/lora-night-20261022/` as `epoch-1`, `epoch-2`, `epoch-3`
and `final`; there is no resume support, so a run interrupted before an epoch
boundary must restart.

On the 26-pair validation set the second adapter answers with C code in all 26
cases, compiles 22 (84.6%, up from 19 for the first adapter) and matches the
reference behavior on 6 (23.1%, up from 0%). On the full 118-pair split it
compiles 83 (70.3%) and matches behavior on 27 (22.9%) with the fixed ten
probes, or on 32 (27.1%) with per-program probes — 23 of 96 (24.0%) when only
pairs with two or more probes are counted. An earlier smoke run (60 short
pairs, 3 epochs) validated the training loop and is kept as
`docs/training_loss_light.png` / `docs/training_loss_dark.png`.

Per epoch on the same 118 pairs, behavior match goes 16.1% → 21.2% → 22.9%
while compile rate wanders 72.9% → 63.6% → 70.3%: the second epoch earns most of
the gain, and compile rate is not a proxy for quality.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/eval_epochs_dark.png">
  <img src="docs/eval_epochs_light.png"
       alt="Behavior match after 1, 2 and 3 epochs: 16.1, 21.2 and 22.9 percent, while compile rate goes 72.9, 63.6 and 70.3 percent"
       width="840">
</picture>

The last `build_dataset.py` run produced **24,377 train / 475 validation**
listings from 13,065 collected programs (`-O0` 12,483, `-O2` 12,369; 908
duplicates and 370 compile failures excluded), and `build_disasm.py` turned the
same records into **23,647 / 463** disassembly pairs (1,648 duplicates, 1,530
of them ambiguous, and 372 compile/link failures). Two teachers are in the mix
and they are now close to balanced: `beam:cloud` wrote 12,887 pairs,
`deepseek-v4.1-flash:cloud` 11,965. No adapter has been trained on this
rebuild yet; the adapter above was trained on an earlier 2,579-pair subset.

## Usage

```bash
cd ml/asm

# collection (line A1)
python3 gen_c.py --dry-run 5   # print the next 5 prompts, no network
python3 gen_c.py --status      # show the checkpoint
python3 gen_c.py --once        # one sample, then exit
python3 gen_c.py               # endless collection, Ctrl+C to stop

# dataset
python3 build_dataset.py                     # rebuild train/val from raw_c/
python3 build_dataset.py --dry-run           # statistics only
python3 build_dataset.py --val-ratio 0.02 --min-asm-lines 8

# reconstruction check
python3 difftest.py ref.c cand.c                       # default inputs
python3 difftest.py ref.c cand.c --cases cases.json    # custom inputs
python3 difftest.py ref.c cand.c --json                # machine-readable

# per-program probes (resumable; --split val for the validation programs only)
python3 gen_cases.py --split all --out dataset/cases.jsonl
python3 gen_cases.py --split val --dry-run 5           # preview, write nothing

# model evaluation on val
python3 eval.py --model Qwen/Qwen3-0.6B                # baseline
python3 eval.py --model Qwen/Qwen3-0.6B --adapter checkpoints/lora --tag lora
python3 eval.py --model Qwen/Qwen3-0.6B --limit 5     # quick smoke run
python3 eval.py --model Qwen/Qwen3-0.6B \
    --adapter checkpoints/lora-night-20261022/final \
    --val dataset/val118.jsonl --cases-by-source dataset/cases_val.jsonl \
    --tag night-final-cases                           # same command resumes

# training
python3 train.py --limit 60 --epochs 3 --max-seq-len 1024 \
    --grad-accum 4 --tag smoke                        # smoke overfit
python3 train.py --epochs 3 --max-seq-len 2048 \
    --grad-accum 8 --tag full                         # full run

# figures
python3 plot_training.py                               # training loss
python3 plot_training.py --state checkpoints/lora-night-20261022/train_state.json \
    --title "Second run: loss on assistant tokens" \
    --subtitle "2,579 pairs | 3 epochs | 969 optimizer steps" \
    --stem training_loss_night
python3 plot_eval.py                                   # evaluation figures

# line B asm programs
python3 verifier.py prog.s --tests prog.test.json
```

## Background collection

The collector runs under `collector.sh`, a supervisor that restarts `gen_c.py`
whenever it exits — an uncaught exception, a killed process, or Ollama going
away when the laptop shuts down. A `flock` guard makes a second copy a no-op.

```bash
nohup setsid ./collector.sh >/dev/null 2>&1 &   # detached, survives the shell
tail -f gen_c.log                               # progress
```

Stop it wrapper first, otherwise it restarts the collector:

```bash
kill "$(cat gen_c.pid)"   # stop the wrapper loop
pkill -f "gen_c.py"       # stop the collector itself
```

The checkpoint `state_gen_c.json` is written atomically every minute and after
every sample, so a restart resumes at the last finished sample. The supervisor
does not survive a reboot on its own: start it again, or put it in a systemd
user unit. The collector needs a running Ollama with a logged-in account and the
model named in `config.json` (`beam:cloud` by default) — without it, every
request fails and the process backs off instead of dying.
