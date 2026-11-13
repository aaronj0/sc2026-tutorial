"""Smoke tests of the tutorial image.

Every library the notebooks use is imported and exercised once on the GPU, so a broken image fails
here before the much slower notebook suite runs.
"""

from importlib.metadata import version as distribution_version
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np


def test_dependency_versions():
    """Critical Python CUDA packages match the resolved CUDA 13.2 stack."""
    expected = {
        "cuda-core": "1.2.0",
        "cuda-cccl": "1.1.1",
        "cupy-cuda13x": "14.2.0",
        "numba-cuda": "0.30.4",
        "numba-cuda-mlir": "0.6.0",
        "jax": "0.11.1",
        "scipy": "1.17.1",
    }
    for distribution, expected_version in expected.items():
        assert distribution_version(distribution) == expected_version

    nvcc = subprocess.run(
        ["nvcc", "--version"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "release 13.2" in nvcc.stdout



def test_system_environment():
    """The system Python holds the stack and mpi4py loads OpenMPI."""
    import jax
    import llvmlite
    import numba

    assert sys.prefix == sys.base_prefix == "/usr"
    assert numba.__version__ == "0.63.1"
    assert llvmlite.__version__ == "0.46.0"
    assert jax.__version__ == "0.11.1"

    program = (
        "from mpi4py import MPI\n"
        "assert MPI.get_vendor()[0] == 'Open MPI'\n"
        "assert MPI.COMM_WORLD.Get_size() == 2\n"
    )
    result = subprocess.run(
        [
            "mpirun",
            "--mca",
            "plm",
            "isolated",
            "--oversubscribe",
            "-n",
            "2",
            sys.executable,
            "-c",
            program,
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_cuda_python():
    """Test that cuda-python works by querying device properties using cuda.core."""
    from cuda.core import system, Device

    # Check CUDA driver version
    driver_version = system.get_kernel_mode_driver_version()
    assert driver_version is not None
    assert len(str(driver_version)) > 0

    # Get device count
    assert system.get_num_devices() > 0, "No CUDA devices found"

    # Get device information
    device = Device(0)
    device.set_current()

    # Verify device properties
    assert len(device.name) > 0
    assert device.device_id == 0
    assert device.uuid is not None
    assert device.pci_bus_id is not None


def test_numba_cuda():
    """Test that numba CUDA works by JIT compiling and running a simple kernel."""
    import warnings
    from numba.core.errors import NumbaPerformanceWarning
    from numba import cuda
    import numpy as np

    # Suppress grid size performance warnings for this test
    warnings.simplefilter('ignore', category=NumbaPerformanceWarning)

    assert cuda.is_available(), "CUDA not available for numba"

    # Define a simple kernel that adds two arrays
    @cuda.jit
    def add_kernel(a, b, c):
        i = cuda.grid(1)
        if i < a.size:
            c[i] = a[i] + b[i]

    # Create test data
    n = 100
    a = np.ones(n, dtype=np.float32)
    b = np.ones(n, dtype=np.float32)
    c = np.zeros(n, dtype=np.float32)

    # Copy to device and run kernel
    d_a = cuda.to_device(a)
    d_b = cuda.to_device(b)
    d_c = cuda.to_device(c)

    threads_per_block = 32
    blocks_per_grid = (n + threads_per_block - 1) // threads_per_block
    add_kernel[blocks_per_grid, threads_per_block](d_a, d_b, d_c)

    # Copy result back
    result = d_c.copy_to_host()

    # Verify result
    assert np.allclose(result, 2.0)


def test_cuda_compute():
    """Test that cuda.compute works by running a parallel reduction algorithm."""
    import cuda.compute as compute
    import cupy as cp

    # Test parallel reduce operation using cuda.compute
    # Create test data on GPU
    n = 1000
    d_input = cp.ones(n, dtype=cp.float32)
    d_output = cp.zeros(1, dtype=cp.float32)
    h_init = np.array([0], dtype=np.float32)

    # Use cuda.compute to reduce (sum) the array
    # reduce_into computes a reduction and stores result in d_output
    compute.reduce_into(
        d_in=d_input,
        d_out=d_output,
        op=compute.OpKind.PLUS,
        num_items=n,
        h_init=h_init,
    )

    result = float(d_output.get()[0])

    # Verify result (sum of 1000 ones should be 1000)
    assert np.isclose(result, float(n), rtol=1e-5)


def test_cuda_compute_algorithms():
    """The keyword-only cuda.compute 1.1 APIs execute representative work."""
    import cuda.compute as compute
    import cupy as cp

    d_input = cp.asarray([3, 1, 4, 1], dtype=cp.int32)
    h_init = np.asarray([0], dtype=np.int32)

    d_scan = cp.empty_like(d_input)
    compute.inclusive_scan(
        d_in=d_input,
        d_out=d_scan,
        op=compute.OpKind.PLUS,
        init_value=h_init,
        num_items=d_input.size,
    )
    np.testing.assert_array_equal(d_scan.get(), [3, 4, 8, 9])

    d_merge = cp.empty_like(d_input)
    compute.merge_sort(
        d_in_keys=d_input,
        d_out_keys=d_merge,
        op=compute.OpKind.LESS,
        num_items=d_input.size,
    )
    np.testing.assert_array_equal(d_merge.get(), [1, 1, 3, 4])

    d_radix = cp.empty_like(d_input)
    compute.radix_sort(
        d_in_keys=d_input,
        d_out_keys=d_radix,
        order=compute.SortOrder.DESCENDING,
        num_items=d_input.size,
    )
    np.testing.assert_array_equal(d_radix.get(), [4, 3, 1, 1])

    def double(value):
        return value * 2

    d_unary = cp.empty_like(d_input)
    compute.unary_transform(
        d_in=d_input,
        d_out=d_unary,
        op=double,
        num_items=d_input.size,
    )
    np.testing.assert_array_equal(d_unary.get(), [6, 2, 8, 2])

    d_binary = cp.empty_like(d_input)
    compute.binary_transform(
        d_in1=d_input,
        d_in2=d_input,
        d_out=d_binary,
        op=compute.OpKind.PLUS,
        num_items=d_input.size,
    )
    np.testing.assert_array_equal(d_binary.get(), [6, 2, 8, 2])


def test_cuda_cooperative():
    """Test cuda.coop._experimental with a block-load algorithm."""
    import warnings
    from numba.core.errors import NumbaPerformanceWarning
    from cuda.coop._experimental import block
    from numba import cuda
    import cupy as cp

    # Suppress grid size performance warnings for this test
    warnings.simplefilter('ignore', category=NumbaPerformanceWarning)

    assert cuda.is_available(), "CUDA not available for cuda.coop test"

    # Test cuda.coop by creating a block load algorithm and using it in a kernel
    # This tests that cuda.coop can compile cooperative algorithms
    threads_per_block = 32
    items_per_thread = 4
    items_per_block = threads_per_block * items_per_thread

    # Create a cooperative block load algorithm
    block_load = block.make_load(
        cp.float32, threads_per_block, items_per_thread, "striped"
    )

    # Define a kernel that uses the cooperative block load
    @cuda.jit(link=block_load.files)
    def load_and_sum_kernel(input_data, output):
        # Local storage for items loaded by this thread
        items = cuda.local.array(items_per_thread, dtype=input_data.dtype)

        # Use cooperative block load
        base = cuda.blockIdx.x * items_per_block
        block_load(input_data[base : base + items_per_block], items)

        # Sum the loaded items for this thread
        thread_sum = 0.0
        for i in range(items_per_thread):
            thread_sum += items[i]

        # Write to output (one value per thread)
        tid = cuda.grid(1)
        if tid < len(output):
            output[tid] = thread_sum

    # Create test data
    n = items_per_block
    input_data = cp.ones(n, dtype=cp.float32)
    output = cp.zeros(threads_per_block, dtype=cp.float32)

    # Launch kernel
    load_and_sum_kernel[1, threads_per_block](input_data, output)

    # Verify - each thread should have summed items_per_thread ones
    result = output.get()
    assert np.allclose(result, items_per_thread)


def test_cupy():
    """Test that CuPy works by performing array operations on GPU."""
    import cupy as cp

    # Create CuPy arrays on GPU
    x = cp.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = cp.array([2.0, 2.0, 2.0, 2.0, 2.0])

    # Perform element-wise operations
    z = x * y
    assert cp.allclose(z, cp.array([2.0, 4.0, 6.0, 8.0, 10.0]))

    # Test reduction
    sum_result = cp.sum(x)
    assert cp.isclose(sum_result, 15.0)

    # Test matrix operations
    a = cp.array([[1.0, 2.0], [3.0, 4.0]])
    b = cp.array([[5.0, 6.0], [7.0, 8.0]])
    c = cp.matmul(a, b)
    expected = cp.array([[19.0, 22.0], [43.0, 50.0]])
    assert cp.allclose(c, expected)


def test_jax_gpu():
    """JAX runs a jit-compiled function on a CUDA device.

    In a subprocess: JAX reserves most of the GPU memory on first use and would keep it for the
    rest of the pytest session, starving the notebook tests that follow.
    """
    program = (
        "import jax, jax.numpy as jnp\n"
        "assert any(d.platform == 'gpu' for d in jax.devices())\n"
        "f = jax.jit(lambda x: jnp.sum(x * x))\n"
        "assert float(f(jnp.arange(5.0))) == 30.0\n"
    )
    result = subprocess.run([sys.executable, "-c", program], capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stdout + result.stderr


def test_cppjit():
    """CppJIT compiles C++ in-process with CUDA support enabled."""
    import cppjit

    assert cppjit.CUDA_ENABLED, "CppJIT was built without CUDA support"
    cppjit.cppdef("int cppjit_smoke_add(int a, int b) { return a + b; }")
    assert cppjit.gbl.cppjit_smoke_add(2, 3) == 5


def test_dragon_and_cuphoton():
    """DragonHPC and cuPhoton are installed at the pinned versions, with cuPhoton's extensions."""
    import cuphoton
    import cuphoton.core._imaging_ext
    import torch
    from cuphoton.xdr import nvcomp_batch

    assert cuphoton.__version__ == "0.2.0.dev87+gb17cf8a"
    assert nvcomp_batch.cpp_helper_available(), nvcomp_batch._CPP_EXT_IMPORT_ERROR
    assert torch.cuda.is_available()
    assert Path(os.environ["CUPHOTON_DEMO_REPO"], "examples", "dragon-imaging-pipeline", "demo.py").is_file()
    result = subprocess.run(["dragon", "--version"], capture_output=True, text=True, check=True)
    assert "Dragon Version 0.15.0" in result.stdout, result.stdout + result.stderr


def test_mpi4py():
    """mpi4py ranks launched by mpirun communicate."""
    program = (
        "from mpi4py import MPI\n"
        "comm = MPI.COMM_WORLD\n"
        "total = comm.allreduce(comm.Get_rank(), op=MPI.SUM)\n"
        "assert total == sum(range(comm.Get_size())), total\n"
        "if comm.Get_rank() == 0:\n"
        "    print('mpi4py ranks:', comm.Get_size())\n"
    )
    result = subprocess.run(
        ["mpirun", "--oversubscribe", "-n", "4", sys.executable, "-c", program],
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, f"mpirun failed:\n{result.stdout}\n{result.stderr}"
    assert "mpi4py ranks: 4" in result.stdout


def test_profilers():
    """Nsight Systems and Nsight Compute are on PATH through the capability wrappers."""
    for tool, version in (("nsys", "2026.1.1"), ("ncu", "2026.1.1")):
        assert shutil.which(tool) == f"/sc2026-tutorial/brev/wrappers/{tool}"
        result = subprocess.run([tool, "--version"], capture_output=True, text=True, check=True)
        assert version in result.stdout, result.stdout


def test_nsightful():
    """nsightful imports, and both of its profiler kernels are installed for the system Python."""
    from nsightful.notebook import is_interactive_notebook

    assert not is_interactive_notebook()
    result = subprocess.run(
        [sys.executable, "-m", "jupyter", "kernelspec", "list", "--json"],
        capture_output=True, text=True, check=True,
    )
    kernels = json.loads(result.stdout)["kernelspecs"]
    for name, profiler in (("nsightful-ncu", "ncu"), ("nsightful-nsys", "nsys")):
        assert kernels[name]["spec"]["metadata"]["nsightful_profiler"] == profiler
        assert Path(kernels[name]["spec"]["argv"][0]).resolve() == Path(sys.executable).resolve()


def test_quijote_data():
    """The Quijote particle positions of the mini-app are baked into the image."""
    data = Path(os.environ["COSMIC_WEB_DATA"]) / "quijote_fiducial_0_z0_positions.npy"
    assert data.stat().st_size == 1610612864
