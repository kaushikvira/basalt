# DFlash2 port — measurement harness

## Files
- `harness.py` — runs `basalt-bench` + parses `--draft-log` → `decode_tps` + draft acceptance (JSON out).
- `probe_set.md` — the probe set (v1 = basalt-bench fixtures; v2 = custom agentic/coding, Phase 3).

## MTP baseline (the Phase 0 reference)
```
# GPU 1 only. Acquire the lease first (see v-llm-gateway/specs/dflash2-port/gpu-lease.md).
~/work/v-llm-gateway/tools/gpu_lease.sh acquire <id> 3600
cd ~/work/v-llm-gateway && ./lane down basalt        # free GPU 1 VRAM

python3 ~/work/basalt-dflash2/tests/dflash2/harness.py \
  --model <MODEL.basalt> --drafter mtp --spec 8 --max-new 512 --seed 42 \
  --out ~/work/v-llm-gateway/specs/dflash2-port/results/mtp-baseline.json

cd ~/work/v-llm-gateway && ./lane up basalt          # restore the lane
~/work/v-llm-gateway/tools/gpu_lease.sh release <id>
```
Record `decode_tps` + `draft.avg_committed_a` + `draft.acceptance_rate` in
`v-llm-gateway/specs/dflash2-port/LOG.md` with this exact command.

## Notes
- The tok/s parse is best-effort: if `decode_tps` is `null`, read `stdout_tail` and fix the
  regex in `harness.py` (one line). The `--draft-log` acceptance parse is solid (`W T a ...`).
- **Losslessness is a separate check**: `--spec 8` vs `--spec 0` output must be bit-identical,
  done via `basalt generate` on a fixed prompt during the GPU step (not in this harness).
- For the DFlash2 spike, point `--bin` at the newly-built basalt binary (with the DFlash2
  drafter) and `--drafter dflash2`; the MTP baseline uses the existing `~/work/basalt/bin/basalt-bench`.
