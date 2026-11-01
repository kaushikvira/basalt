"""Copy a .basalt pack, dropping named section prefixes (here: the mtp/* sections), for a no-drafter baseline.

    python3 tools/copy_nodrafter.py SRC.basalt DST.basalt mtp/
"""
import sys

sys.path.insert(0, "tools")
from model_file import ModelFile, ModelWriter  # noqa: E402

src_path, dst_path, skip = sys.argv[1], sys.argv[2], sys.argv[3]
src = ModelFile(src_path)
w = ModelWriter(dst_path)
for name in src.names():
    if name.startswith(skip):
        print(f"skipping {name}")
        continue
    off, size = src.section(name)
    w.add_file(name, src_path, off, size)
w.close()
print(f"done -> {dst_path}")
