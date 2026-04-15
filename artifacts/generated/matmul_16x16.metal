#include <metal_stdlib>
using namespace metal;

kernel void matmul_kernel(
    device const float* var_A [[buffer(0)]],
    device const float* var_B [[buffer(1)]],
    device float* var_C [[buffer(2)]],
    constant int& var_M [[buffer(3)]],
    constant int& var_N [[buffer(4)]],
    constant int& var_K [[buffer(5)]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 tgid [[threadgroup_position_in_grid]]
) {
    // 2D Threadgroup shared memory tile allocation (BM=16, BN=16, BK=16)
    threadgroup float tileA[16][16];
    threadgroup float tileB[16][16];

    // 2D Thread and Block indexing
    uint tid_x = tid.x; // column local (0..15)
    uint tid_y = tid.y; // row local (0..15)

    uint row = tgid.y * 16 + tid_y;
    uint col = tgid.x * 16 + tid_x;

    float acc = 0.0f;

    // K-tile loop over global K dimension
    for (int k0 = 0; k0 < var_K; k0 += 16) {
        // Cooperative load tileA [BM x BK] with edge masking
        if (row < (uint)var_M && (k0 + tid_x) < (uint)var_K) {
            tileA[tid_y][tid_x] = var_A[row * var_K + (k0 + tid_x)];
        } else {
            tileA[tid_y][tid_x] = 0.0f;
        }

        // Cooperative load tileB [BK x BN] with edge masking
        if ((k0 + tid_y) < (uint)var_K && col < (uint)var_N) {
            tileB[tid_y][tid_x] = var_B[(k0 + tid_y) * var_N + col];
        } else {
            tileB[tid_y][tid_x] = 0.0f;
        }

        // Barrier 1: Wait for threadgroup tile load to complete
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Compute 16x16 tile dot product into local accumulator
        for (int k_inner = 0; k_inner < 16; ++k_inner) {
            acc += tileA[tid_y][k_inner] * tileB[k_inner][tid_x];
        }

        // Barrier 2: Wait for tile compute to finish before overwriting shared memory in next iteration
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write back computed accumulator to global memory with edge masking
    if (row < (uint)var_M && col < (uint)var_N) {
        var_C[row * var_N + col] = acc;
    }
}