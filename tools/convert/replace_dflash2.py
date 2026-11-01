"""Replace the DFlash2 drafter section in an existing .basalt pack, leaving every other section byte for byte.

    python3 tools/convert/replace_dflash2.py --from BASE-dflash2.basalt --dflash2-source DIR --out MODEL.basalt [--name NAME] [--force]

The language model (dense, experts, gguf) and the profile/vision/hc8 sections are copied from BASE
unchanged, so the target is exactly BASE's target (e.g. the NVFP4 pack stays an NVFP4 target). The
existing `dflash2/dense` + `dflash2/dense.index` sections are REPLACED (from the DFlash2 drafter's
model.safetensors + config.json) and `meta.dflash2` is updated. This is the counterpart of
`add_dflash2.py`, which refuses a base that already carries a dflash2 section.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import model_file as MF  # noqa: E402
import model_sections as S  # noqa: E402

DFLASH2_SECTIONS = ("dflash2/dense", "dflash2/dense.index")


def die(msg: str) -> None:
    print(f"replace_dflash2: {msg}", file=sys.stderr)
    sys.exit(1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--from", dest="src", required=True, metavar="BASE.basalt",
                    help="the existing pack that already carries a dflash2 section")
    ap.add_argument("--dflash2-source", required=True, metavar="DIR",
                    help="the replacement DFlash2 drafter's model.safetensors + config.json directory")
    ap.add_argument("--out", required=True, metavar="MODEL.basalt")
    ap.add_argument("--name", default=None, help="the model name reported by the server (default: BASE's)")
    ap.add_argument("--force", action="store_true", help="overwrite --out")
    a = ap.parse_args()

    out = Path(a.out)
    if out.exists() and not a.force:
        die(f"{out} exists (use --force to overwrite)")
    if not Path(a.src).is_file():
        die(f"missing input: {a.src}")
    partial = Path(str(out) + ".partial")

    base = MF.ModelFile(a.src)
    missing = [n for n in DFLASH2_SECTIONS if n not in base.sections]
    if missing:
        die(f"{a.src} carries no dflash2 section (missing {missing}); use add_dflash2.py to add one")
    base_meta = json.loads(base.read("meta"))

    d = None
    try:
        d = S.DFlash2(a.dflash2_source)
    except S.ConvertError as e:
        die(str(e))

    if partial.exists():
        partial.unlink()
    w = MF.ModelWriter(str(partial))
    t0 = time.time()
    ok = False
    try:
        kept = [n for n in base.names() if n not in ("meta",) and n not in DFLASH2_SECTIONS]
        print(f"copying {len(kept)} sections from {Path(a.src).name} (replacing {', '.join(DFLASH2_SECTIONS)}) ...",
              flush=True)
        for name in base.names():
            if name == "meta" or name in DFLASH2_SECTIONS:
                continue
            off, size = base.section(name)
            w.add_file(name, a.src, off, size)
            if w.entries[-1][3] != base.sections[name][2]:
                die(f"section {name}: the copy's SHA-256 differs from {a.src}'s")
        print("writing the replacement dflash2 sections ...", flush=True)
        w.add_chunks("dflash2/dense", d.dense_chunks())
        w.add_bytes("dflash2/dense.index", d.dense_index())
        meta = dict(base_meta)
        meta["name"] = a.name or base_meta.get("name")
        meta["dflash2"] = d.provenance()
        meta["dflash2_replaced_from"] = {"path": Path(a.src).name,
                                         "size": Path(a.src).stat().st_size}
        w.add_bytes("meta", json.dumps(meta, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
        w.close()
        ok = True
    finally:
        if not ok:
            try:
                w.f.close()
            except OSError:
                pass
            partial.unlink(missing_ok=True)
    os.replace(partial, out)
    total = out.stat().st_size
    print(f"done: {len(w.entries)} sections, {total / (1 << 30):.3f} GiB, {time.time() - t0:.1f} s -> {out}")
    base = d = w = None
    gc.collect()


if __name__ == "__main__":
    main()
