"""Run the cuPhoton dragon-imaging demo the way its README describes.

The example lives in the cuPhoton checkout at CUPHOTON_DEMO_REPO. prepare.py downloads the
public FITS image, builds the dataset and trains the xScan model; run_dragon_imaging.ipynb then
runs the pipeline on Dragon. Both run as shipped and every line they print goes to the test log.
CUPHOTON_DEMO_BATCH_SIZE replaces the training batch size of train.example.yaml (32 as shipped)
for a GPU where that does not fit; the notebook's own inference batch size stays 32.
"""

import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import nbformat
import pytest
import yaml
from nbclient import NotebookClient

from test_notebooks import check_gpu_state, extract_cell_outputs

if 'CUPHOTON_DEMO_REPO' not in os.environ:
    pytest.skip('CUPHOTON_DEMO_REPO is not set; the demo runs in the tutorial image', allow_module_level=True)

REPO = Path(os.environ['CUPHOTON_DEMO_REPO'])
EXAMPLE = REPO / 'examples' / 'dragon-imaging-pipeline'
NOTEBOOK = EXAMPLE / 'run_dragon_imaging.ipynb'
BATCH_SIZE = os.environ.get('CUPHOTON_DEMO_BATCH_SIZE')
OOM_PATTERN = re.compile(r'out of memory|OOM')
OOM_HINT = ('CUDA out of memory: the shipped training batch size needs about 22 GiB of GPU memory; '
            'set CUPHOTON_DEMO_BATCH_SIZE=16 for a smaller GPU.')


def example_with_batch_size(root, batch_size):
    """Copy the example beside a link to the transient-chain example, with the batch size replaced.

    prepare.py reads train.example.yaml next to itself and the transient-chain example from
    ../transient-chain, so the copy keeps that layout."""
    examples = root / 'examples'
    example = examples / 'dragon-imaging-pipeline'
    shutil.copytree(EXAMPLE, example, ignore=shutil.ignore_patterns('tests', '__pycache__'))
    (examples / 'transient-chain').symlink_to(REPO / 'examples' / 'transient-chain')
    config_path = example / 'train.example.yaml'
    config = yaml.safe_load(config_path.read_text())
    config['batch_size'] = batch_size
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    return example


@pytest.fixture(scope='module')
def prepared_root(tmp_path_factory):
    """The output of prepare.py; its whole log is printed, and a failure names the OOM case."""
    root = tmp_path_factory.mktemp('sc26-dragon')
    example = EXAMPLE if BATCH_SIZE is None else example_with_batch_size(root, int(BATCH_SIZE))
    output = root / 'demo'
    print(f"\n=== prepare.py from {example}, training batch size {BATCH_SIZE or 'as shipped'} ===")
    check_gpu_state()
    start = time.time()
    result = subprocess.run([sys.executable, 'prepare.py', '--output', str(output)],
                            cwd=example, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(result.stdout)
    print(f"=== prepare.py exit {result.returncode} in {time.time() - start:.0f}s ===")
    if result.returncode != 0:
        hint = OOM_HINT if OOM_PATTERN.search(result.stdout) else ''
        pytest.fail(f"prepare.py exited {result.returncode}. {hint}")
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
    # The notebook's kernelspec cuphoton-sc26 is not installed in the image.
    client = NotebookClient(nb, timeout=1200, kernel_name='python3',
                            resources={'metadata': {'path': str(EXAMPLE)}})
    cell_times = {}
    try:
        with client.setup_kernel():
            for i, cell in enumerate(nb.cells):
                if cell.cell_type != 'code':
                    continue
                cell_start = time.time()
                source_preview = cell.source[:60].replace('\n', ' ')
                print(f"  Cell {i}: {source_preview}...", end='', flush=True)
                client.execute_cell(cell, i)
                cell_times[i] = time.time() - cell_start
                print(f" [{cell_times[i]:.2f}s]", flush=True)
    except Exception as e:
        all_outputs = extract_cell_outputs(nb, cell_times)
        hint = OOM_HINT if OOM_PATTERN.search(all_outputs) else ''
        pytest.fail(f"Notebook execution failed (total time: {time.time() - notebook_start:.2f}s): {hint}\n"
                    f"{e}\n\n=== ALL CELL OUTPUTS ===\n{all_outputs}")
    print(f"=== Completed {NOTEBOOK.name} in {time.time() - notebook_start:.2f}s ===")
    print(f"=== ALL CELL OUTPUTS ===\n{extract_cell_outputs(nb, cell_times)}")
