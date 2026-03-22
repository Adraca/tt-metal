# Plan: Split First-K Read Between Writer (NOC1) and Reader (NOC0)

## Context

Profiling of 8544/q288/k512 shows a **one-time cold-start penalty** on the absolute first K chunk per core:

- **K_FIRST** (union of TRISCs): 47.0 us P50 vs K_INNER 39.4 us — **+7.6 us** penalty
- **K_FIRST_WAIT** (`cb_wait_front(cb_kt_in)`, TRISC_0 only): **4.17 us** P50
- **Residual init overhead**: ~3.4 us (exp_packthread_tile_init, llk_pack_mop_config, etc.)

This is strictly a **once-per-core** event. After the first K chunk, the reader's double-buffered pipeline is primed and all subsequent K waits are 0.03 us — including across Q-chunk and ring-iteration boundaries.

**Root cause**: The reader (NCRISC) takes ~3.5 us startup before issuing its first DRAM read, plus ~4 us DRAM round-trip. Compute finishes init at ~3.4 us and waits for K that doesn't arrive until ~7.5 us. The writer (BRISC) has already finished its setup (~3 us) and is blocked at `cb_wait_front(cb_signal)` — idle during this window.

**Goal**: Split the first K chunk's DRAM reads between writer (NOC1) and reader (NOC0), doubling DRAM bandwidth for the cold start. Reader takes the upper half of K-sequence rows, writer takes the lower half. Both write into the same cb_k_in region with transposed layout. Expected saving: significant reduction of the 4.17 us K wait.

## Key Insight: No NOC Mode Change Needed

- BRISC (writer) reads on NOC1, NCRISC (reader) reads on NOC0 — different hardware counters, no barrier conflict in `DM_DEDICATED_NOC`
- Writer already issues `noc_async_read_tile` for accumulator restore (`issue_restore_reads`) — proven pattern

## Transpose Layout

K is read with `transpose=true` in `read_block` (dataflow_common.hpp:934-935). Tile `(row, col)` lands at:
```
base_ptr + row * tile_bytes + col * Sk_chunk_t * tile_bytes
```
- `row` ∈ [0, Sk_chunk_t) — K-sequence dimension (fast axis in L1)
- `col` ∈ [0, DHt) — head dimension
- For 8544/q288/k512: Sk_chunk_t=16, DHt=4, total=64 tiles

The writer must replicate this same transposed stride calculation when computing write pointers for its half.

## Row Split

- **Reader (upper half, NOC0)**: rows `[0, Sk_chunk_t/2)` — 32 tiles
- **Writer (lower half, NOC1)**: rows `[Sk_chunk_t/2, Sk_chunk_t)` — 32 tiles

Tiles from both halves interleave in L1 within each head-dim column's stride, but each RISC writes to non-overlapping offsets.

## Approach

### 1. Factory (`ring_joint_sdpa_program_factory.cpp`)

**Add K tensor info to writer compile-time args** (after existing `stats_args`):
```cpp
TensorAccessorArgs(input_tensor_k.buffer()).append_to(writer_compile_time_args);  // NEW
```

**Add K DRAM address to writer runtime args** (before fused_op_receiver args):
```cpp
writer_args = { out_addr, joint_out_addr, stats_addr, global_q_start, global_q_end,
                k_addr };  // NEW — position [5]
```

**Add flag to reader runtime args** (before fused_op_receiver args):
```cpp
// After mcast_sender_wait, before push_ring_sdpa_fused_op_rt_args:
reader_args.push_back(use_streaming_compute && !first_q_receives_from_chain);  // NEW
```

Where `first_q_receives_from_chain` is computed per-core from the chain topology (already known in the factory loop).

**Update `override_runtime_arguments`**: Add `k_addr` at the new index for writer, and recalculate the reader flag.

### 2. Writer Kernel (`ring_joint_writer.cpp`)

**Parse new args** (after existing stats_args at line 363):
```cpp
constexpr auto k_args = TensorAccessorArgs<stats_args.next_compile_time_args_offset()>();
// RT arg:
const uint32_t k_addr = get_arg_val<uint32_t>(argidx++);  // before fused_op_receiver
```

**Issue non-blocking reads for lower half of K BEFORE scalar generation** (insert between lines ~401 and 407):
```cpp
if constexpr (use_streaming_compute) {
    // Prefetch lower half of first K chunk into cb_k_in on NOC1.
    // Reads fly during scalar gen + mask gen + ring loop entry.
    constexpr uint32_t cb_k_in = tt::CBIndex::c_1;
    constexpr uint32_t k_tile_bytes = get_tile_size(cb_k_in);
    constexpr uint32_t k_chunk_tiles = Sk_chunk_t * DHt;
    constexpr uint32_t half_Sk = Sk_chunk_t / 2;

    const auto k_reader = TensorAccessor(k_args, k_addr, k_tile_bytes);
    const auto k_tile_logical = TensorTileShape(B, NHK, local_padded_Nt, DHt);
    const auto k_generator = PaddedAddrGenerator(k_reader, k_tile_logical);

    // First Q chunk for this core: compute K slice coordinates
    const uint32_t first_nb = global_q_start / (NH * num_q_chunks);
    const uint32_t first_nk = ((global_q_start % (NH * num_q_chunks)) / num_q_chunks) / (NH / NHK);

    // Get write pointer (same pattern as reader: get_write_ptr on cb_k_in)
    const uint32_t base_ptr = get_write_ptr(cb_k_in);

    // Issue reads for lower half rows [half_Sk, Sk_chunk_t) with transposed layout
    // Tile (row, col) → base_ptr + row * k_tile_bytes + col * Sk_chunk_t * k_tile_bytes
    for (uint32_t row = half_Sk; row < Sk_chunk_t; ++row) {
        uint32_t write_ptr = base_ptr + row * k_tile_bytes;
        for (uint32_t col = 0; col < DHt; ++col) {
            k_generator.maybe_read_tile(first_nb, first_nk, row, col, local_padded_Nt, write_ptr);
            write_ptr += Sk_chunk_t * k_tile_bytes;
        }
    }
    // DO NOT barrier here — let reads fly during scalar gen
}
```

**Barrier and signal AFTER scalar/mask generation** (insert after line ~419, before `find_last_active_ring_iter`):
```cpp
if constexpr (use_streaming_compute) {
    noc_async_read_barrier();  // Wait for lower-half K reads on NOC1
    // Signal reader that writer's half is done
    noc_semaphore_inc(k_prefetch_semaphore_noc_addr, 1);
}
```

### 3. Reader Kernel (`ring_joint_reader.cpp`)

**Parse new flag** (after `mcast_sender_wait`, before `RingSDPAOpReceiver`):
```cpp
const uint32_t split_first_k_with_writer = get_arg_val<uint32_t>(argidx++);
```

**On k_chunk==0, ring_iter==0: read only upper half, then wait for writer's lower half** (modify lines 282-301):
```cpp
const bool k_split_active = (split_first_k_with_writer && ring_iter == 0 && k_chunk == 0 && !should_receive);

cb_reserve_back(cb_k_in, k_chunk_tiles);
uint32_t cb_k_start_address = get_write_ptr(cb_k_in);

if (k_split_active) {
    // Reader handles upper half: rows [0, half_Sk)
    // Same transposed layout as read_block with transpose=true
    constexpr uint32_t half_Sk = Sk_chunk_t / 2;
    const uint32_t base_ptr = cb_k_start_address;
    for (uint32_t row = 0; row < half_Sk; ++row) {
        uint32_t write_ptr = base_ptr + row * k_tile_bytes;
        for (uint32_t col = 0; col < DHt; ++col) {
            local_k_generator.maybe_read_tile(
                first_nb, first_nk, row, col, end_seq_tile_first_k, write_ptr);
            write_ptr += Sk_chunk_t * k_tile_bytes;
        }
    }
    noc_async_read_barrier();  // Wait for upper-half reads on NOC0

    // Wait for writer's lower-half signal
    noc_semaphore_wait(k_prefetch_semaphore_addr_ptr, 1);

    cb_push_back(cb_k_in, k_chunk_tiles);  // All 64 tiles now in CB
} else if (should_receive) {
    // ... existing receive path unchanged ...
    cb_push_back(cb_k_in, k_chunk_tiles);
} else {
    read_block(/* ... existing full DRAM read unchanged ... */);
}

// Forwarding logic unchanged — uses cb_k_start_address
if (should_forward) { /* ... same as before ... */ }
```

### 4. Synchronization

**Semaphore**: A single L1 semaphore on each core for writer→reader signaling.
- Factory creates it: `auto k_prefetch_sem_id = CreateSemaphore(program, core_grid, 0);`
- Pass semaphore ID as compile-time arg to both reader and writer
- Writer: `noc_semaphore_inc(reader_core_noc_addr | semaphore_addr, 1)` — but since writer and reader are on the SAME core, this is a local L1 write (no NOC needed): just `*semaphore_ptr = 1;`
- Reader: `noc_semaphore_wait(semaphore_ptr, 1)`

### 5. No Compute Changes

Compute is unchanged. `cb_wait_front(cb_kt_in)` at line 704 sees K data earlier.

## CB Protocol Correctness

cb_k_in is double-buffered (2 slots). Reader owns cb_reserve_back/cb_push_back:
1. **Reader** does `cb_reserve_back` — gets slot A base pointer
2. **Reader** reads upper half (rows 0-7) on NOC0 into slot A
3. **Writer** reads lower half (rows 8-15) on NOC1 into slot A (same base pointer, non-overlapping offsets)
4. Both barrier on their respective NOCs
5. Writer signals reader via semaphore
6. **Reader** waits for semaphore, then `cb_push_back` — slot A complete
7. **Compute** waits for slot A, processes all 64 tiles
8. **Reader** on k_chunk=1: `cb_reserve_back` → gets slot B. Normal full-read path resumes.
9. **Compute** pops slot A. Double-buffer continues normally.

Single producer (reader) for CB protocol. Writer just writes to L1 at known offsets within the reserved region.

## Verification

1. **PCC test** (CCLs restored):
   ```bash
   rm -rf built/tt-metal-cache*
   pytest tests/nightly/blackhole/ccl/test_ring_joint_sdpa.py::test_ring_joint_attention_sdpa_accuracy[wan2_2_compat_2240x4_h10-k512-q224-bf16] \
          tests/nightly/blackhole/ccl/test_ring_joint_sdpa.py::test_ring_joint_attention_sdpa_accuracy[wan2_2_compat_8544x4_h10-k512-q288-bf16] -x
   ```

2. **Perf test** (CCLs disabled) — verify first-K wait reduced:
   ```bash
   rm -rf built/tt-metal-cache*
   export TT_METAL_DEVICE_PROFILER=1
   pytest "tests/nightly/blackhole/ccl/test_ring_joint_sdpa.py::test_ring_joint_attention_create_perf_table[wan2_2_compat_8544x4_h10]" -x -s
   ```

3. **Tracy profiling** — measure K_FIRST_WAIT to confirm reduction:
   ```bash
   python -m tracy -r -p -n k_prefetch_after \
     -m pytest "tests/nightly/blackhole/ccl/test_ring_joint_sdpa.py::test_ring_joint_attention_sdpa_sweep_perf_impl[wan2_2_compat_8544x4_h10-k512-q288-bf16]" -x -s
   ```

## Files to Modify

| File | Change |
|------|--------|
| `ttnn/.../sdpa/device/ring_joint_sdpa_program_factory.cpp` | K accessor CT args for writer, k_addr RT arg for writer, split flag RT arg for reader, semaphore creation, update override_runtime_arguments |
| `ttnn/.../sdpa/device/kernels/dataflow/ring_joint_writer.cpp` | Parse K args, issue non-blocking lower-half K reads before scalar gen, barrier + semaphore signal after |
| `ttnn/.../sdpa/device/kernels/dataflow/ring_joint_reader.cpp` | Parse split flag, on first K: read upper half only + wait for writer semaphore + push; all other K chunks unchanged |
