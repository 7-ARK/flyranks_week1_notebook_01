"""Execute the submitted study notebooks, preserving real outputs."""
from pathlib import Path
import nbformat
from nbclient import NotebookClient

root=Path(__file__).resolve().parent
names=['w02_ml_task_framing.ipynb','w03_data_contract.ipynb','w04_baseline_score.ipynb','capstone.ipynb']
for name in names:
    p=root/'notebooks'/name
    notebook=nbformat.read(p,as_version=4)
    NotebookClient(notebook,timeout=180,resources={'metadata':{'path':str(p.parent)}}).execute()
    assert all(cell.execution_count is not None and not any(o.output_type=='error' for o in cell.outputs)
               for cell in notebook.cells if cell.cell_type=='code')
    nbformat.write(notebook,p)
    print(f'{name}: all code cells executed successfully')
