# sc2026-tutorial

Material for the SC26 tutorial [Accelerating and Scaling Python for HPC](https://sc26.conference-program.com/presentation/?id=tut150&sess=sess183),
Monday, 16 November 2026, 8:30am to 5:00pm CST, room W185a.

Python is powering breakthrough exascale scientific discoveries - come learn how to program the world's largest
supercomputers with it! In this interactive tutorial you'll learn how to write, debug, profile, and optimize
high-performance, multi-node GPU applications in Python. You'll learn and master: CuPy for drop-in GPU acceleration of
NumPy workflows, Numba for writing custom kernels that match the performance of C++ and Fortran, mpi4py for scaling
across thousands of nodes, DragonHPC for workflow acceleration, and CppJIT for on-demand Python/C++ interoperability
and automatic binding generation, including compiler-based automatic differentiation of C++ with Clad. Everything is
hands-on: short, interactive lectures by expert instructors are paired with guided Jupyter notebooks that introduce
each concept and have you apply it immediately in bite-sized exercises, among them a mini-app applying several
programming models (CuPy, JAX, cuda.compute, Numba CUDA and CppJIT) to cosmic-web data and a CARTopiaX CAR T-cell
therapy example. We'll use a web-based environment that requires no installs or setup - just a laptop and a browser.
Whether you're a domain scientist seeking faster turnaround or a software engineer evaluating portable acceleration
strategies, you'll leave with a roadmap, skills, and code for bringing scalable Python practices back to your own HPC
facility.

In this tutorial we will cover:
- NumPy fundamentals and the `ndarray` memory model.
- Accelerating array workflows on the GPU with CuPy, including memory spaces and asynchrony.
- Distributed computing with mpi4py for multi-node parallelism.
- Programming models in Python for GPU applications: array expressions, fusion, library iterators, kernel authoring, and interoperability with CUDA C++ kernels.
- Reading an application's profile with Nsight Systems and Nsight Compute.
- Case studies of horizontal Python and C++ integration in domain sciences such as cancer research and high energy physics.
- cuPhoton and DragonHPC (placeholder).

Attendees are expected to have general knowledge of HPC and Python programming. Some familiarity with NumPy and C/C++ is
helpful but not required. The outline and the notebook list are in
[notebooks/syllabi/sc26__8_hours.ipynb](notebooks/syllabi/sc26__8_hours.ipynb).

## Layout

| Path | Content |
|---|---|
| `notebooks/fundamentals/` | NumPy and CuPy |
| `notebooks/distributed/` | mpi4py, DragonHPC and cuPhoton |
| `notebooks/gpu-mini-app/` | Analysis of cosmic-web density fields on GPU |
| `notebooks/science-integration/` | Python/C++ integration examples for domain science |
| `notebooks/syllabi/` | Tutorial outline |
| `slides/` | Slides and material |
| `brev/` | Docker image and compose stack. Build scripts and checks |
| `test/` | Package and notebook tests run during image build |

## Running the tutorial

The tutorial runs in the container image `ghcr.io/pyhpc/sc2026-tutorial`. [brev/docker-compose.yml](brev/docker-compose.yml)
starts JupyterLab on port 8888, opening on the outline, with Nsight Systems on port 8080 and Nsight Compute on
port 8081 as launcher tiles. Brev starts this file. Locally, with Docker and the NVIDIA Container Toolkit installed:

```bash
docker compose -f brev/docker-compose.yml up -d
docker compose -f brev/docker-compose.yml down -v
```

## Development

```bash
docker compose -f brev/docker-compose.yml -f brev/docker-compose.dev.yml up -d   # this checkout mounted in the containers
docker compose -f brev/docker-compose.yml run --rm --no-deps jupyter test       # the test suite
docker compose -f brev/docker-compose.yml run --rm --no-deps jupyter shell      # a shell in the container
brev/build.bash [--chain] [--push]                                              # build and test the image
python3 brev/check-notebooks.py [--fix]                                         # notebook format
python3 brev/check-links.py                                                     # relative links
```

The two checks also run as pre-commit hooks (`pre-commit install`) and in the `Checks` workflow. The `Image`
workflow builds the image on a self-hosted GPU runner set up with `brev/runner-setup.bash`, runs every notebook
in it, starts the Compose stack from it and publishes it: a pull request from a branch of this repository gets
the tag `pr-<number>`, a push to `main` the tags `git-<sha>` and `latest`. The runner is the Brev instance
`sc26-ci`: the workflow starts it with the repository secret `BREV_API_KEY` and stops it after a green build.

## Licences

Written material is licensed under CC BY-NC-SA 4.0 ([LICENSE](LICENSE)) and code under Apache-2.0
([LICENSE-CODE](LICENSE-CODE)).
