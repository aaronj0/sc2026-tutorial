// SPDX-License-Identifier: Apache-2.0
// CUB's device histogram over the births of one cell type, computed as the histogram reads them: q holds the bin index
// of every voxel (int32, n^3, C order with k fastest) and hist receives k_bins int32 counters. The pointers and the
// stream are passed as integers the way CuPy exposes them.

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cub/device/device_histogram.cuh>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        const cudaError_t err_ = (call);                                      \
        if (err_ != cudaSuccess) {                                            \
            std::fprintf(stderr, "CUDA error '%s' at %s:%d\n",                \
                         cudaGetErrorString(err_), __FILE__, __LINE__);       \
            std::abort();                                                     \
        }                                                                     \
    } while (0)

// The eight cell types a voxel owns, in the order of the signed sum: vertex, edges, faces, cube.
enum Cell { V, EX, EY, EZ, FXY, FXZ, FYZ, C };

// The birth of one cell at voxel t: the minimum of q over the cell's corners, read through this functor as CUB's
// histogram sweeps the counting iterator. Each corner is the voxel or its periodic neighbour along one or more axes.
template <Cell CELL>
struct Birth {
    const int* q;
    int n;

    __device__ __forceinline__ int wrap1(int i) const { return i + 1 == n ? 0 : i + 1; }

    __device__ __forceinline__ int operator()(long long t) const
    {
        const long long nn = static_cast<long long>(n) * n;
        const int k = static_cast<int>(t % n), j = static_cast<int>((t / n) % n), i = static_cast<int>(t / nn);
        const int i1 = wrap1(i), j1 = wrap1(j), k1 = wrap1(k);
        const int v000 = q[t];
        if constexpr (CELL == V) return v000;
        const int v100 = q[i1 * nn + j * n + k], v010 = q[i * nn + j1 * n + k], v001 = q[i * nn + j * n + k1];
        if constexpr (CELL == EX) return min(v000, v100);
        if constexpr (CELL == EY) return min(v000, v010);
        if constexpr (CELL == EZ) return min(v000, v001);
        if constexpr (CELL == FXY) return min(min(v000, v100), min(v010, q[i1 * nn + j1 * n + k]));
        if constexpr (CELL == FXZ) return min(min(v000, v100), min(v001, q[i1 * nn + j * n + k1]));
        if constexpr (CELL == FYZ) return min(min(v000, v010), min(v001, q[i * nn + j1 * n + k1]));
        const int v110 = q[i1 * nn + j1 * n + k], v101 = q[i1 * nn + j * n + k1];
        const int v011 = q[i * nn + j1 * n + k1], v111 = q[i1 * nn + j1 * n + k1];
        return min(min(min(v000, v100), min(v010, v110)), min(min(v001, v101), min(v011, v111)));
    }
};

template <Cell CELL>
void histogram_of(const int* q, int* hist, int n, int k_bins, cudaStream_t s)
{
    const long long samples = static_cast<long long>(n) * n * n;
    const auto births = thrust::make_transform_iterator(thrust::counting_iterator<long long>(0), Birth<CELL>{q, n});
    // The scratch is sized by CUB on the first call and kept for the process, so later calls launch only the kernels.
    static void* scratch = nullptr;
    static size_t scratch_bytes = 0;
    size_t needed = 0;
    CUDA_CHECK(cub::DeviceHistogram::HistogramEven(nullptr, needed, births, hist, k_bins + 1, 0, k_bins, samples, s));
    if (needed > scratch_bytes) {
        if (scratch != nullptr) CUDA_CHECK(cudaFree(scratch));
        CUDA_CHECK(cudaMalloc(&scratch, needed));
        scratch_bytes = needed;
    }
    CUDA_CHECK(cub::DeviceHistogram::HistogramEven(scratch, scratch_bytes, births, hist, k_bins + 1, 0, k_bins, samples, s));
}

// The histogram of the births of cell type `cell` (0 to 7 in the order of `Cell`) over the n^3 field at q_ptr.
void cub_birth_histogram(int cell, unsigned long long q_ptr, unsigned long long hist_ptr, int n, int k_bins,
                         unsigned long long stream)
{
    const int* q = reinterpret_cast<const int*>(q_ptr);
    int* hist = reinterpret_cast<int*>(hist_ptr);
    const cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    switch (static_cast<Cell>(cell)) {
    case V:   histogram_of<V>(q, hist, n, k_bins, s); break;
    case EX:  histogram_of<EX>(q, hist, n, k_bins, s); break;
    case EY:  histogram_of<EY>(q, hist, n, k_bins, s); break;
    case EZ:  histogram_of<EZ>(q, hist, n, k_bins, s); break;
    case FXY: histogram_of<FXY>(q, hist, n, k_bins, s); break;
    case FXZ: histogram_of<FXZ>(q, hist, n, k_bins, s); break;
    case FYZ: histogram_of<FYZ>(q, hist, n, k_bins, s); break;
    case C:   histogram_of<C>(q, hist, n, k_bins, s); break;
    default:  std::fprintf(stderr, "cub_birth_histogram: cell %d is not one of the eight\n", cell); std::abort();
    }
}
