// Row-gather kernel: for each selected row, copy its nnz from the
// per-shard intermediate buffer into the contiguous output.
//
// String-substitution placeholders: OUTPUT_DATA_T, OUTPUT_INDICES_T.
//
// Parameters:
//   in_data, in_indices : per-shard intermediate buffers (uint16)
//   out_data, out_indices : output buffers (templated dtype)
//   in_row_offsets : per-row offsets in the input (int64, n_rows+1)
//   out_row_offsets : per-row offsets in the output (int64, n_rows+1)
//   n_rows : number of rows in this shard's contribution
//
// Launch config: grid = (n_rows,), block = (min(256, max_nnz_per_row),)
// Each block handles one row; threads within the block stride over nnz.

#include <cstdint>

extern "C" __global__ void row_gather(
    const uint16_t* __restrict__ in_data,
    const uint16_t* __restrict__ in_indices,
    OUTPUT_DATA_T* __restrict__ out_data,
    OUTPUT_INDICES_T* __restrict__ out_indices,
    const long long* __restrict__ in_row_offsets,
    const long long* __restrict__ out_row_offsets,
    long long n_rows
) {
    long long row = blockIdx.x;
    if (row >= n_rows) return;
    long long in_start  = in_row_offsets[row];
    long long in_end    = in_row_offsets[row + 1];
    long long out_start = out_row_offsets[row];
    long long nnz_row   = in_end - in_start;
    for (long long i = threadIdx.x; i < nnz_row; i += blockDim.x) {
        out_data[out_start + i]    = static_cast<OUTPUT_DATA_T>(in_data[in_start + i]);
        out_indices[out_start + i] = static_cast<OUTPUT_INDICES_T>(in_indices[in_start + i]);
    }
}
