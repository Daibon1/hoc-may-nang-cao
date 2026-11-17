"""Execute all notebook cells and retain their outputs; fail on any cell error."""
from pathlib import Path

import nbformat
from nbclient import NotebookClient

root = Path(__file__).resolve().parents[1]
path = root / "notebooks/01_eda_preprocessing.ipynb"
notebook = nbformat.read(path, as_version=4)
NotebookClient(notebook, timeout=300, kernel_name="python3",
               resources={"metadata": {"path": str(root)}},
               record_timing=False).execute()
nbformat.write(notebook, path)
print(f"PASS: executed {sum(c.cell_type == 'code' for c in notebook.cells)} code cells")
