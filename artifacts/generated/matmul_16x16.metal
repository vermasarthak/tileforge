#include <metal_stdlib>
using namespace metal;

kernel void matmul_kernel(
    device float* var_A [[buffer(0)]],
    device float* var_B [[buffer(1)]],
    device float* var_C [[buffer(2)]],
    constant int& var_M [[buffer(3)]],
    constant int& var_N [[buffer(4)]],
    constant int& var_K [[buffer(5)]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 tgid [[threadgroup_position_in_grid]]
) {
    int pid4;
    int pid5;
    int c6;
    int v7;
    int tid8;
    int v9;
    int v10;
    int v11;
    float c12 = 0.0f;
    int c13;
    int var_121;
    float var_132 = 0.0f;
    bool cond14;
    float ld15 = 0.0f;
    float ld16 = 0.0f;
    float tiled_dot17 = 0.0f;
    float v18 = 0.0f;
    int c19;
    int v20;
    float var_213 = 0.0f;
    int _state = 0;
    while (true) {
        switch (_state) {
            case 0: {
                pid4 = (int)tgid.x;
                pid5 = (int)tgid.y;
                c6 = 16;
                v7 = pid4 * c6;
                tid8 = (int)tid.x;
                v9 = v7 + tid8;
                v10 = pid5 * c6;
                v11 = v10 + tid8;
                c12 = 0.0f;
                c13 = 0;
                var_121 = c13;
                var_132 = c12;
                _state = 1;
                break;
            }
            case 1: {
                cond14 = (var_121 < var_K);
                if (cond14) {
                    _state = 2;
                } else {
                    var_213 = var_132;
                    _state = 3;
                }
                break;
            }
            case 2: {
                ld15 = var_121 ? var_A[v9] : 0.0f;
                ld16 = v11 ? var_B[var_121] : 0.0f;
                threadgroup float tileA[16][16];
                threadgroup float tileB[16][16];
                uint row = tgid.y * 16 + tid.y;
                uint col = tgid.x * 16 + tid.x;
                float accum_dot = 0.0f;
                for (int k0 = 0; k0 < var_K; k0 += 16) {
                    if (row < (uint)var_M && (k0 + tid.x) < (uint)var_K) {
                        tileA[tid.y][tid.x] = var_A[row * var_K + (k0 + tid.x)];
                    } else {
                        tileA[tid.y][tid.x] = 0.0f;
                    }
                    if ((k0 + tid.y) < (uint)var_K && col < (uint)var_N) {
                        tileB[tid.y][tid.x] = var_B[(k0 + tid.y) * var_N + col];
                    } else {
                        tileB[tid.y][tid.x] = 0.0f;
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                    for (int k_inner = 0; k_inner < 16; ++k_inner) {
                        accum_dot += tileA[tid.y][k_inner] * tileB[k_inner][tid.x];
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                }
                tiled_dot17 = accum_dot;
                v18 = var_132 + tiled_dot17;
                c19 = 1;
                v20 = var_121 + c19;
                var_121 = v20;
                var_132 = v18;
                _state = 1;
                break;
            }
            case 3: {
                uint store_row = tgid.y * 16 + tid.y;
                uint store_col = tgid.x * 16 + tid.x;
                if (store_row < (uint)var_M && store_col < (uint)var_N) {
                    var_C[store_row * var_N + store_col] = tiled_dot17;
                }
                return;
            }
        }
    }
}