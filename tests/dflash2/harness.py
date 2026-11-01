#!/usr/bin/env python3
"""DFlash2 port — fast acceptance harness (v1).

Measures, on basalt-bench's standard fixtures (conv,struct,prose), for a given drafter:
  - decode_tps   (tokens/s, from the basalt-bench table)
  - draft acceptance (from --draft-log: avg tokens committed per window 'a', avg window 'T', rate a/T)

No guessed numbers: every metric is parsed from real basalt-bench / --draft-log output. The
tok/s parse is best-effort (the exact table field may need a one-line fix after a real run);
the full stdout tail is always captured so the parse can be corrected. The --draft-log
acceptance parse is solid (per-window lines are "W <T> <a> <from_sfx> <ms...> [probs]").

Losslessness is a SEPARATE check (--spec N vs --spec 0 output must be bit-identical), done via
`basalt generate` on a fixed prompt during the GPU step — not here.

Usage:
  harness.py --model FILE.basalt --drafter mtp [--spec 8] [--max-new 512] [--seed 42] [--out FILE]
  harness.py --model FILE.basalt --drafter none        # spec 0 (pure greedy) — losslessness baseline
  harness.py --model FILE.basalt --drafter dflash2 ... # later (the ported drafter)
"""
import argparse, json, os, re, subprocess, tempfile


def run_bench(binp, model, gpu, spec, seed, max_new, dlog, dflash2=False):
    # --gpu 1 is MANDATORY: basalt-bench defaults to GPU 0 (the 5090), which is FORBIDDEN.
    # --engine: the worktree bench's compile-time stamp is a container path (/src/bin/basalt),
    # so name the engine binary beside the bench explicitly.
    engine = os.path.join(os.path.dirname(os.path.abspath(binp)), "basalt")
    # the worktree bench's fixture stamp is also a container path (/src/data/bench); name the repo's fixtures
    fixtures = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(binp))), "data", "bench")
    passthrough = ["--spec", str(spec), "--greedy", "--max-new", str(max_new), "--draft-log", dlog]
    if dflash2:
        passthrough.append("--dflash2")
    cmd = [binp, "--model", model, "--gpu", gpu, "-t", "conv,struct,prose",
           "--seed", str(seed), "--engine", engine, "--fixtures", fixtures, "--", *passthrough]
    return subprocess.run(cmd, capture_output=True, text=True)


def parse_draft_log(path):
    Ts, As = [], []
    if not os.path.exists(path):
        return None
    with open(path) as f:
        for line in f:
            p = line.split()
            if len(p) >= 3 and p[0] == "W":
                try:
                    Ts.append(int(p[1])); As.append(int(p[2]))
                except ValueError:
                    pass
    if not Ts:
        return None
    at, aa = sum(Ts) / len(Ts), sum(As) / len(As)
    return {"n_windows": len(Ts), "avg_window_T": round(at, 3),
            "avg_committed_a": round(aa, 3),
            "acceptance_rate": round(aa / at, 4) if at else None}


def parse_table(stdout):
    """Parse the basalt-bench table: | test | t/s | accept % | draft len |. Per-fixture metrics."""
    rows = {}
    for line in stdout.splitlines():
        s = line.strip()
        if not s.startswith("|"):
            continue
        cells = [c.strip() for c in s.strip("|").split("|")]
        if len(cells) < 4 or cells[0] in ("test", "") or set(cells[0]) <= set("- "):
            continue
        try:
            rows[cells[0]] = {"tps": float(cells[1]), "accept_pct": float(cells[2]),
                              "draft_len": float(cells[3])}
        except ValueError:
            continue
    return rows or None


def parse_tps(stdout):
    for pat in (r'decode_tps[^\d]*([\d.]+)', r'([\d.]+)\s*tok/s', r'([\d.]+)\s*tps'):
        m = re.search(pat, stdout, re.I)
        if m:
            return float(m.group(1))
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", "--pack", required=True, dest="model")
    ap.add_argument("--drafter", default="mtp", choices=["mtp", "dflash2", "none"])
    ap.add_argument("--spec", type=int, default=8)
    ap.add_argument("--max-new", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--bin", default=os.path.expanduser("~/work/basalt/bin/basalt-bench"))
    ap.add_argument("--gpu", default="1", help="GPU index; MUST be 1 (a5000). GPU 0 is forbidden.")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    spec = 0 if a.drafter == "none" else a.spec
    with tempfile.TemporaryDirectory() as td:
        dlog = os.path.join(td, "draft.log")
        r = run_bench(a.bin, a.model, a.gpu, spec, a.seed, a.max_new, dlog, dflash2=(a.drafter == "dflash2"))
        res = {"model": a.model, "drafter": a.drafter, "spec": spec, "seed": a.seed,
               "max_new": a.max_new, "exit_code": r.returncode,
               "fixtures": parse_table(r.stdout),
               "decode_tps": parse_tps(r.stdout), "draft": parse_draft_log(dlog),
               "stdout_tail": r.stdout[-1200:], "stderr_tail": (r.stderr or "")[-400:]}
    out = json.dumps(res, indent=2)
    if a.out:
        with open(a.out, "w") as f:
            f.write(out + "\n")
    print(out)


if __name__ == "__main__":
    main()
