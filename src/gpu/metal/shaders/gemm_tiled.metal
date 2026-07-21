#include <metal_stdlib>
using namespace metal;

constant uint TILE_SIZE = 16;

// When compiled on Apple Silicon (macOS, IOS), Apple's Metal compiler translates this directly into GPU hardware instructions.
// It loads matrix patches into high-speed threadgroup (local) memory, outputting high tensor compute performance.
kernel void tiled_gemm_kernel(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C       [[buffer(2)]],
    constant uint& M      [[buffer(3)]],
    constant uint& N      [[buffer(4)]],
    constant uint& K      [[buffer(5)]],
    uint2 thread_pos      [[thread_position_in_grid]],
    uint2 local_pos       [[thread_position_in_threadgroup]],
    uint2 group_pos       [[threadgroup_position_in_grid]])
{
    // Allocate high-speed on-chip shared tile memory
    threadgroup float tileA[TILE_SIZE][TILE_SIZE];
    threadgroup float tileB[TILE_SIZE][TILE_SIZE];

    float accumulator = 0.0f;
    const uint row = thread_pos.y;
    const uint col = thread_pos.x;
    const uint num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (uint t = 0; t < num_tiles; ++t) {
        const uint tiled_col_A = t * TILE_SIZE + local_pos.x;
        const uint tiled_row_B = t * TILE_SIZE + local_pos.y;

        // Load tile from A into threadgroup memory
        if (row < M && tiled_col_A < K) {
            tileA[local_pos.y][local_pos.x] = A[row * K + tiled_col_A];
        } else {
            tileA[local_pos.y][local_pos.x] = 0.0f;
        }

        // Load tile from B into threadgroup memory
        if (tiled_row_B < K && col < N) {
            tileB[local_pos.y][local_pos.x] = B[tiled_row_B * N + col];
        } else {
            tileB[local_pos.y][local_pos.x] = 0.0f;
        }

        // Synchronize all threads in the threadgroup before computation
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Compute inner dot product for this tile
        for (uint k = 0; k < TILE_SIZE; ++k) {
            accumulator += tileA[local_pos.y][k] * tileB[k][local_pos.x];
        }

        // Synchronize before loading the next tile
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < M && col < N) {
        C[row * N + col] = accumulator;
    }
}