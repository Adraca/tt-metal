// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/eltwise_unary/sfpu_math.h"

ALIGNED_PRAGMA_INLINE
void mamba_scan_recurrence(uint32_t cb_in, uint32_t cb_out, uint32_t num_tiles) {
    // Mamba associative scan operator:
    // A_out = A_1 * A_2
    // B_out = A_1 * B_2 + B_1
    //
    // Note: Fable assumed this was a literal 2x2 matrix multiply, but for Mamba,
    // the hidden state D is typically 1024 or 2048. Thus, A_i and B_i are 1D vectors
    // of size D, and the multiplications are element-wise!
    // A 32x32 tile holds exactly 1024 elements (one sequence step of size D).
    // Because it is element-wise, we CAN use the highly parallel Trisc tensor pipeline
    // via mul_tile and add_tile, rather than the scalar SFPU.

    mul_tiles_init();
    add_tiles_init();

    // Local scan (Up-Sweep)
    cb_wait_front(cb_in, num_tiles);
    cb_reserve_back(cb_out, num_tiles);

    // Initialize accumulator tiles in DEST registers
    // Dest 0 = A_accum, Dest 1 = B_accum
    copy_tile_init();
    copy_tile_to_dst_init_short();
    
    // Copy first element to accumulators
    copy_tile(cb_in, 0, 0); // A_0
    copy_tile(cb_in, 1, 1); // B_0

    pack_tile(0, cb_out, 0); // Pack A_0
    pack_tile(1, cb_out, 1); // Pack B_0

    for (uint32_t i = 2; i < num_tiles; i += 2) {
        // Read A_i, B_i
        // Compute B_out = A_accum * B_i + B_accum
        // Compute A_out = A_accum * A_i

        // mul_tile: Dest 2 = A_accum * B_i
        mul_tiles_init_short();
        mul_tile(0, cb_in, i + 1, 2); 

        // add_tile: Dest 1 (B_accum) = Dest 2 + B_accum
        add_tiles_init_short();
        add_tile(2, 1, 1);

        // mul_tile: Dest 0 (A_accum) = A_accum * A_i
        mul_tiles_init_short();
        mul_tile(0, cb_in, i, 0);

        // Pack results to CB out
        pack_tile(0, cb_out, i);
        pack_tile(1, cb_out, i + 1);
    }

    cb_push_back(cb_out, num_tiles);
    cb_pop_front(cb_in, num_tiles);
}

namespace NAMESPACE {
void MAIN {
    // Runtime args
    uint32_t num_cores_x = get_arg_val<uint32_t>(0);
    uint32_t num_cores_y = get_arg_val<uint32_t>(1);
    uint32_t row_sync_sem_addr = get_arg_val<uint32_t>(2);
    uint32_t col_sync_sem_addr = get_arg_val<uint32_t>(3);

    // 2 tiles per step (A and B are interleaved as tiles)
    // In a production kernel, num_tiles would be passed dynamically based on chunk size K.
    constexpr uint32_t num_tiles = 64; 
    
    constexpr uint32_t cb_in0 = tt::CB::c_in0;
    constexpr uint32_t cb_out0 = tt::CB::c_out0;

    // Phase 1: Local Up-Sweep
    mamba_scan_recurrence(cb_in0, cb_out0, num_tiles);

    // Phase 2: Distributed NOC Tree Sync (Mocked for clarity)
    // Real implementation requires NOC atomic inc/wait using the semaphores.
    // row_sync_sem_addr is used for row-level reduction
    // col_sync_sem_addr is used for column-level reduction

    // Phase 3: Local Down-Sweep (Apply base offset)
    // ...
}
}
