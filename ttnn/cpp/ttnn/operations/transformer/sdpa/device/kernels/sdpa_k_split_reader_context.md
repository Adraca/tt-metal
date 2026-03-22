# SDPA K Split Reader / Dual-NOC Investigation Context

Investigation into whether using both NOC0 and NOC1 for reading (split reader / dual-NOC pattern) can reduce the first-K penalty in the SDPA ring V2 kernel.

---

## 1. Problem Statement

Profiling of 8544/q288/k512 config (branch `dnijemcevic/sdpa_deferred_norm_prefetch_v4`) shows:

| K-chunk type | P50 duration | Delta vs K_INNER |
|---|---|---|
| K_INNER (steady state) | 39.4 us | — |
| K_FIRST | 47.0 us | +7.6 us (19%) |
| K_LAST | 45.4 us | +6.0 us (normalization) |

First-K penalty breakdown:
- **4.17 us**: `cb_wait_front(cb_kt_in)` stall on UNPACK (TRISC_0) — DRAM read latency not yet hidden
- **~3.4 us**: One-time init overhead (`exp_packthread_tile_init`, `mm_block_init_short`, `llk_pack_mop_config`)
- Steady-state K/V/Q waits: all 0.03 us (reader fully ahead of compute)

Source: [Profiling results gist](https://gist.github.com/djordjenTT/092651246eb430628a48ff7e766829d6#file-profiling_results_8544_288_512-md)

---

## 2. Current SDPA Ring NOC Architecture

| RISC | Kernel | NOC | Mode | Role |
|------|--------|-----|------|------|
| NCRISC (RISCV_1) | `ring_joint_reader.cpp` | NOC0 | DM_DEDICATED_NOC | Reads Q, K, V from DRAM; forwards K/V to next ring core |
| BRISC (RISCV_0) | `ring_joint_writer.cpp` | NOC1 | DM_DEDICATED_NOC | Writes final output to DRAM; save/restore accumulators |

### Reader (NCRISC)
- File: `ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/ring_joint_reader.cpp`
- Factory: `ttnn/cpp/ttnn/operations/transformer/sdpa/device/ring_joint_sdpa_program_factory.cpp` (line ~1083)
- Uses `ReaderDataMovementConfig` → RISCV_1, NOC0, DM_DEDICATED_NOC
- K and V fetches are **sequential** — K is fully read + forwarded before V starts
- Double-buffered CBs (k_tiles = `Sk_chunk_t * DHt * 2`)
- No overlap between K and V DRAM reads within a single K-chunk

### Writer (BRISC)
- File: `ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/ring_joint_writer.cpp`
- Factory: same file, line ~1089
- Uses `WriterDataMovementConfig` → RISCV_0, NOC1, DM_DEDICATED_NOC
- Also does **NOC reads** for accumulator restore (`issue_restore_reads` → `noc_async_read_tile`)
- Restore reads go through NOC0 (hardware preferred read NOC), writes through NOC1

---

## 3. NOC0 vs NOC1 for Reading

Both NOCs support `noc_async_read`. No inherent bandwidth/latency difference. Key factors:

- **Routing direction**: NOC0 and NOC1 traverse chip in opposite directions
- **Doubling bandwidth**: Using both NOCs for reads simultaneously doubles aggregate DRAM bandwidth, but prevents overlapping reads + writes on the same NOC
- **Congestion**: One reader per DRAM bank is recommended. Place readers near their bank to minimize hops.

Source: DeepWiki — [NOC0 vs NOC1 query](https://deepwiki.com/search/can-both-noc0-and-noc1-be-used_c0d23551-517a-473a-ba42-6e1f98298ee4)

---

## 4. DM_DYNAMIC_NOC Mode

In dynamic NOC mode, each RISC can issue reads/writes on **either** NOC by passing `noc_index` or `1 - noc_index` to the API.

### Key behaviors
- **Barriers are per-NOC**: `noc_async_read_barrier(noc)` only barriers that one NOC. Must loop over `NUM_NOCS` to barrier both.
- **TRID management**: Must `reset_noc_trid_barrier_counter()` per NOC with inter-RISC semaphore sync before/after.
- **Blackhole caveat**: `noc_inline_dw_write` is disabled due to HW issue.

### Configuration
```cpp
DataMovementConfig{
    .processor = DataMovementProcessor::RISCV_0,  // or RISCV_1
    .noc = NOC::RISCV_0_default,
    .noc_mode = tt_metal::NOC_MODE::DM_DYNAMIC_NOC,
    .compile_args = compile_args
};
```

### Code pointers
- **Test kernel**: `tests/tt_metal/tt_metal/test_kernels/dataflow/dynamic_noc_writer.cpp` — dual-NOC read pattern with interleaved reads on `noc_index` and `1 - noc_index`
- **Test driver**: `tests/tt_metal/tt_metal/noc/test_dynamic_noc.cpp` — configures both BRISC and NCRISC with `DM_DYNAMIC_NOC`
- **API**: `tt_metal/hw/inc/api/dataflow/dataflow_api.h` (lines ~1735-1909 for dynamic mode barrier implementations)
- **Enum**: `tt_metal/api/tt-metalium/kernel_types.hpp` (lines 39-42)

Source: DeepWiki — [Dynamic NOC mode query](https://deepwiki.com/search/in-ttmetal-when-using-dynamic_3943d692-4e53-4630-98cd-d7cc3e43ded0)

---

## 5. Existing Split Reader Patterns

### 5a. Conv2d-Style Split Reader

Two reader kernels (BRISC + NCRISC) read different portions of the same activation block into a shared CB. Semaphore-based synchronization.

- `is_split_reader_supported()` — checks layout/conv type/block height
- `is_split_reader_viable()` — cost-benefit analysis
- Kernel: `reader_conv_activations_2d_mcast_padded_with_halo_3x3_weights_v2.cpp`
- Factory: `Conv2dShardedProgramFactory`
- Pooling: `pool_multi_core_program_factory.cpp` — `reader0_kernel` (RISCV_0) + `reader1_kernel` (RISCV_1)

Source: DeepWiki — [Split reader pattern query](https://deepwiki.com/search/what-is-the-split-reader-patte_715d08bc-4069-409c-993c-0128886d9920)

### 5b. FlashMLA Dual-NOC Pattern (DeepSeek V3) — Most Relevant

The closest analog to what could apply to SDPA ring. Uses `DM_DYNAMIC_NOC` with both RISCs reading on separate NOCs.

| RISC | Operation | NOC | Method |
|------|-----------|-----|--------|
| NCRISC | Read K from DRAM | NOC0 | `noc_async_read_one_packet_with_state_with_trid` (TRID-pipelined, ~15 outstanding) |
| NCRISC | Synchronization | NOC1 | `noc_semaphore_inc` (atomic) |
| BRISC | Multicast K | NOC0 | `noc_async_write_multicast` |
| BRISC | Read Q | NOC1 | `noc_async_read` |
| BRISC | Write tree reduction | NOC1 | `noc_async_write` |

**NCRISC-BRISC synchronization**: Shared L1 semaphore counters at page-level granularity. NCRISC increments per page read, BRISC waits via `noc_semaphore_wait_min()`. Double-buffered sync pointers (`curr_ptr` / `next_ptr`).

**Key code pointers**:
- **Implementation**: `models/demos/deepseek_v3_b1/unified_kernels/flash_mla.hpp`
  - NCRISC K reads: lines ~305-333 (TRID-pipelined)
  - BRISC Q read: lines ~406-412
  - BRISC K multicast: lines ~467-503
  - NCRISC-BRISC sync: lines ~335-336, 500-501
- **Kernel entry**: `models/demos/deepseek_v3_b1/micro_ops/flash_mla/kernels/flash_mla_kernel.cpp`
- **Factory config**: `models/demos/deepseek_v3_b1/micro_ops/flash_mla/op.py` (line 915: `noc_mode=ttnn.NOC_MODE.DM_DYNAMIC_NOC`)
- **Static assert**: `static_assert(noc_mode == DM_DYNAMIC_NOC, "Flash MLA Decode kernel only supports DM_DYNAMIC_NOC");`

---

## 6. Applicability to SDPA Ring V2

### What the first-K penalty looks like

The 4.17 us K wait is the DRAM read latency for the very first K chunk before the double-buffered reader pipeline is primed. After that, the reader stays fully ahead of compute (0.03 us waits).

### Where dual-NOC / split reader could help

1. **First-K cold start**: If BRISC prefetched the first K chunk on NOC1 while NCRISC reads on NOC0, the initial 4.2 us stall could be halved (~2 us saving).

2. **Overlapping K and V reads within a chunk**: Currently sequential. If BRISC read V on NOC1 while NCRISC reads K on NOC0, V read latency would be removed from the critical path. However, profiling shows V wait is already 0.03 us steady-state (V reads overlap with Phase 1 Q@KT compute).

3. **Accumulator restore bandwidth**: Writer's restore reads currently use NOC0 (same as reader's K/V reads). Dynamic NOC could isolate these onto separate NOCs. However, profiling shows all restore barriers are already instant (0.03 us P50).

### Tradeoffs

- **Complexity**: Switching to `DM_DYNAMIC_NOC` requires per-NOC barriers, TRID management, BRISC-NCRISC semaphore synchronization (FlashMLA shows this is non-trivial)
- **Writer impact**: Writer currently uses NOC1 for writes. In dynamic mode, writer could still use NOC1 for writes and NOC0 for restore reads, but barrier management becomes more complex
- **ROI**: Steady-state is already optimal. Benefit is limited to ~2 us off the first-K stall per Q-chunk (3 Q-chunks × 4 ring iters = 12 first-K events per core, saving ~24 us total out of ~8044 us kernel duration = 0.3%)

### Alternative: prefetch first K before compute starts

Instead of dual-NOC, the reader could issue the first K read earlier (before compute's init sequence). This would overlap the 4.2 us DRAM latency with the 3.4 us compute init, reducing the first-K penalty to ~0.8 us with zero NOC architecture changes.

---

## 7. Key Files Reference

| File | Purpose |
|------|---------|
| `ttnn/.../sdpa/device/kernels/dataflow/ring_joint_reader.cpp` | Current SDPA ring reader (NCRISC, NOC0, dedicated) |
| `ttnn/.../sdpa/device/kernels/dataflow/ring_joint_writer.cpp` | Current SDPA ring writer (BRISC, NOC1, dedicated) |
| `ttnn/.../sdpa/device/ring_joint_sdpa_program_factory.cpp` | Factory creating reader/writer kernels |
| `models/demos/deepseek_v3_b1/unified_kernels/flash_mla.hpp` | FlashMLA dual-NOC pattern (reference implementation) |
| `models/demos/deepseek_v3_b1/micro_ops/flash_mla/op.py` | FlashMLA factory with DM_DYNAMIC_NOC config |
| `tests/tt_metal/tt_metal/test_kernels/dataflow/dynamic_noc_writer.cpp` | Dynamic NOC test kernel (dual-NOC read pattern) |
| `tests/tt_metal/tt_metal/noc/test_dynamic_noc.cpp` | Dynamic NOC test driver |
| `tt_metal/hw/inc/api/dataflow/dataflow_api.h` | NOC API (barrier semantics for dynamic mode) |
| `tt_metal/api/tt-metalium/kernel_types.hpp` | NOC_MODE enum definition |
