"""Run the cuPhoton dragon-imaging demo the way its README describes.

The example lives in the cuPhoton checkout at CUPHOTON_DEMO_REPO. prepare.py downloads the
public FITS image, builds the dataset and trains the xScan model; run_dragon_imaging.ipynb then
runs the pipeline on Dragon. Both run as shipped. CUPHOTON_DEMO_BATCH_SIZE replaces the training
batch size of train.example.yaml (32 as shipped, about 22 GiB of GPU memory) for a smaller GPU.
"""

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import nbformat
import pytest
import yaml
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

from test_notebooks import check_gpu_state, extract_cell_outputs

REPO = Path(os.environ['CUPHOTON_DEMO_REPO'])
EXAMPLE = REPO / 'examples' / 'dragon-imaging-pipeline'
NOTEBOOK = EXAMPLE / 'run_dragon_imaging.ipynb'
BATCH_SIZE = os.environ.get('CUPHOTON_DEMO_BATCH_SIZE')


def example_with_batch_size(root, batch_size):
    """Copy the example beside a link to the transient-chain example, with the batch size replaced.

    prepare.py reads train.example.yaml next to itself and the transient-chain example from
    ../transient-chain, so the copy keeps that layout."""
    examples = root / 'examples'
    example = examples / 'dragon-imaging-pipeline'
    shutil.copytree(EXAMPLE, example, ignore=shutil.ignore_patterns('tests', '__pycache__', 'scratch'))
    (examples / 'transient-chain').symlink_to(REPO / 'examples' / 'transient-chain')
    config_path = example / 'train.example.yaml'
    config = yaml.safe_load(config_path.read_text())
    config['batch_size'] = batch_size
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    return example


@pytest.fixture(scope='module')
def prepared_root(tmp_path_factory):
    """The output of prepare.py, or a failure that carries the end of its log."""
    root = tmp_path_factory.mktemp('sc26-dragon')
    example = EXAMPLE if BATCH_SIZE is None else example_with_batch_size(root, int(BATCH_SIZE))
    output = root / 'demo'
    print(f"\n=== prepare.py from {example}, batch size {BATCH_SIZE or 'as shipped'} ===")
    check_gpu_state()
    start = time.time()
    result = subprocess.run(
        [sys.executable, 'prepare.py', '--output', str(output), '--seed', '2026', '--images', '8'],
        cwd=example, capture_output=True, text=True,
    )
    print(f"=== prepare.py exit {result.returncode} in {time.time() - start:.0f}s ===")
    tail = '\n'.join((result.stdout + result.stderr).splitlines()[-60:])
    if result.returncode != 0:
        pytest.fail(f"prepare.py failed:\n{tail}")
    print(tail)
    return output


def test_prepare(prepared_root):
    assert (prepared_root / 'prepared.json').is_file()
    training = yaml.safe_load((prepared_root / 'training' / 'training.yaml').read_text())
    print(f"training batch size {training['batch_size']}")


def test_demo_notebook_executes(prepared_root, monkeypatch):
    """Execute run_dragon_imaging.ipynb cell by cell, with the optional transient chain."""
    monkeypatch.setenv('CUPHOTON_DEMO_ROOT', str(prepared_root))
    monkeypatch.setenv('CUPHOTON_DEMO_TRANSIENT_CHAIN', '1')
    print(f"\n=== Starting notebook: {NOTEBOOK} ===")
    check_gpu_state()
    notebook_start = time.time()
    nb = nbformat.read(NOTEBOOK, as_version=4)
    # The notebook names the kernel of Davin's conda environment; the image has one kernel.
    client = NotebookClient(nb, timeout=1200, kernel_name='python3',
                            resources={'metadata': {'path': str(EXAMPLE)}})
    cell_times = {}
    try:
        with client.setup_kernel():
            for i, cell in enumerate(nb.cells):
                if cell.cell_type != 'code':
                    continue
                cell_start = time.time()
                print(f"  Cell {i}: {cell.source[:60].replace(chr(10), ' ')}...", end='', flush=True)
                client.execute_cell(cell, i)
                cell_times[i] = time.time() - cell_start
                print(f" [{cell_times[i]:.2f}s]", flush=True)
    except (CellExecutionError, Exception) as e:
        all_outputs = extract_cell_outputs(nb, cell_times)
        pytest.fail(f"Notebook execution failed (total time: {time.time() - notebook_start:.2f}s):\n"
                    f"{e}\n\n=== ALL CELL OUTPUTS ===\n{all_outputs}")
    print(f"=== Completed {NOTEBOOK.name} in {time.time() - notebook_start:.2f}s ===")
    print(f"=== ALL CELL OUTPUTS ===\n{extract_cell_outputs(nb, cell_times)}")
