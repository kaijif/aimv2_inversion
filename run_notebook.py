"""Run the notebook's code cells as a plain Python script (for Slurm batch jobs).

Unlike `jupyter nbconvert --execute`, output goes straight to the Slurm log as it happens,
there is no kernel connection to stall, and a failure shows a normal traceback.
The value of a cell's last bare expression (e.g. a results table) is printed, like in Jupyter.
Usage:  python run_notebook.py aimv2_feature_inversion.ipynb
"""
import ast
import json
import os
import sys
import time

os.environ.setdefault("MPLBACKEND", "Agg")      # figures are saved to files; plt.show() is a no-op

path = sys.argv[1] if len(sys.argv) > 1 else "aimv2_feature_inversion.ipynb"
cells = [c for c in json.load(open(path))["cells"] if c["cell_type"] == "code"]
ns = {"__name__": "__main__"}
t_start = time.time()

for i, cell in enumerate(cells):
    src = "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
    if all(not l.strip() or l.strip().startswith("#") for l in src.splitlines()):
        continue
    first = next(l for l in src.splitlines() if l.strip())
    print(f"\n===== cell {i}: {first[:70]}  [{(time.time() - t_start) / 60:.1f} min elapsed]", flush=True)
    tree = ast.parse(src)
    last = None
    if tree.body and isinstance(tree.body[-1], ast.Expr):
        last = ast.Expression(tree.body.pop().value)
    exec(compile(tree, f"{path} [cell {i}]", "exec"), ns)
    if last is not None:
        value = eval(compile(last, f"{path} [cell {i}]", "eval"), ns)
        if value is not None:
            print(value.to_string() if hasattr(value, "to_string") else value, flush=True)

print(f"\nall cells finished in {(time.time() - t_start) / 3600:.2f} h", flush=True)
