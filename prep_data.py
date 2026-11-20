"""Build the ImageNet splits and the uint8 pixel caches on CPU, so the GPU job skips this step.

Runs the notebook's own config and data cells (so paths, splits and cache names match exactly),
then stops before anything that needs a GPU or model weights.
Usage:  python prep_data.py [aimv2_feature_inversion.ipynb]
"""
import json
import sys
import time
from pathlib import Path

nb_path = Path(sys.argv[1] if len(sys.argv) > 1 else "aimv2_feature_inversion.ipynb")
cells = [c for c in json.loads(nb_path.read_text())["cells"] if c["cell_type"] == "code"]

def source(cell):
    return "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]

def is_only_comments(code):
    return all(not line.strip() or line.strip().startswith("#") for line in code.splitlines())

t0 = time.time()
ns = {"__name__": "__main__"}
for i, cell in enumerate(cells):
    code = source(cell)
    if is_only_comments(code):
        continue
    exec(compile(code, f"{nb_path.name} [code cell {i}]", "exec"), ns)
    if "CACHE_PATHS = {}" in code:        # the data-pipeline cell: splits + pixel caches
        break
else:
    sys.exit("could not find the data-pipeline cell in the notebook")

print(f"\nprep finished in {(time.time() - t0) / 60:.1f} min")
for split, path in ns["CACHE_PATHS"].items():
    print(f"  {split:6s} {path}  ({Path(path).stat().st_size / 1e9:.2f} GB)")
