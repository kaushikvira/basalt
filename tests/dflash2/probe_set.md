# DFlash2 port — probe set

## v1 (Phase 0): basalt-bench standard fixtures
The v1 probe set is `basalt-bench`'s built-in decode fixtures: **conv, struct, prose**
(conversation / structured / prose). Rationale:
- (a) pretokenized + built into the engine → no tokenization step, fast;
- (b) the repo's standard benchmark → comparable to existing baselines;
- (c) `struct` is a reasonable proxy for code/agentic structure.

The **same fixtures run for MTP and DFlash2**, so the Phase 0 comparison is fair.

Run shape:
```
basalt-bench --model <model> -t conv,struct,prose --seed 42 \
  -- --spec 8 --greedy --max-new 512 --draft-log <file>
```

## v2 (Phase 3): custom agentic/coding probe set
A fixed ~12-prompt set spanning **code / prose / agentic-tool-call** (the workload DFlash2 is
meant to win on), tokenized via the Qwen3.8-Flash-Next HF tokenizer and fed to `basalt generate
--tokens-file`. Added in Phase 3 to *specifically* measure the DFlash2 win on agentic/coding
traffic — the v1 `struct` fixture is a proxy, not a direct measure.
