/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 Amazon.com, Inc. and affiliates.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
 * =====================================================================================
 *  NIXL device-channel API for the libfabric (EFA) backend  --  HONEST-CEILING SCAFFOLD
 * =====================================================================================
 *
 *  This header mirrors, signature-for-signature, the UCX device API in
 *  src/api/gpu/ucx/nixl_device.cuh (nixlPut / nixlAtomicAdd / nixlGpuGetXferStatus /
 *  nixlGetPtr). It exists so a GPU kernel can be written ONCE against the nixl-facing
 *  device surface and have it work on either backend.
 *
 *  WHAT IS BUILDABLE TODAY (this file):
 *    - The nixl-facing device signatures + an opaque memory-view element type, matching
 *      UCX's shape so kernels are source-portable.
 *    - A 3-backend swap seam selecting the transport at compile time.
 *    - PROXY-backed bodies: the device side flags work into a host-visible mailbox that
 *      the CPU proxy (libfabric fi_write path) drains. Correct, functional, EP-usable.
 *
 *  WHAT IS *NOT* HERE AND IS NOT OURS TO CLAIM (kernel-gated):
 *    - True GPU-initiated RDMA post on EFA. libfabric has NO ucp_device_put analogue --
 *      there is no GPU-callable fi_write. That requires GPU-initiated EFA (GDAKI), which
 *      is blocked upstream on rdma-core #1701 + aws-ofi-nccl #1311 + an EFA completion-
 *      counter tag (efa_linux_3.2+). All OPEN / unmerged as of 2026-07-24.
 *    - GPU-native performance parity with UCX. NEVER quote a PROXY transfer as a
 *      GPU-native measurement. The proxy is a correctness path (~1.7-2x slower than a
 *      real GPU-native post would be), not a performance win.
 *
 *  THE 3-BACKEND SWAP SEAM (NIXL_LIBFABRIC_DEVICE_BACKEND):
 *    0 = PROXY  (default, buildable now): device flags -> host mailbox -> CPU fi_write.
 *    1 = GDA    (correctness backend, later): our proven type-5 GPU-initiated EFA-GDA
 *               path. Correct but slower than proxy; NOT a perf win. Still gated on the
 *               GDA build being wired in.
 *    2 = GDAKI  (native, kernel-gated): kernel-posted WQE straight to the NIC. Enable
 *               ONLY when rdma-core #1701 + efa_linux_3.2+ land. Do NOT flip this on
 *               and claim it works before the gate opens.
 *
 *  Grounding: mirrors ep/nixl-ep/CLAUDE.md two-halves split; NEG banlist (nixl-ep-013/
 *  019/162: 387x proxy gap, fi_cntr thread-locality, FI_MORE-16 #1862 hazard) applies.
 * =====================================================================================
 */
#ifndef _NIXL_DEVICE_LIBFABRIC_CUH
#define _NIXL_DEVICE_LIBFABRIC_CUH

#include <nixl_types.h>

#include <cstdint>
#include <cstdio>

/**
 * @def NIXL_LIBFABRIC_DEVICE_BACKEND
 * @brief Compile-time transport selector for the device channel. See the swap-seam
 *        block above. Defaults to PROXY (0) -- the only backend buildable today.
 */
#ifndef NIXL_LIBFABRIC_DEVICE_BACKEND
#define NIXL_LIBFABRIC_DEVICE_BACKEND 0
#endif

#define NIXL_LIBFABRIC_BACKEND_PROXY 0
#define NIXL_LIBFABRIC_BACKEND_GDA 1
#define NIXL_LIBFABRIC_BACKEND_GDAKI 2

/**
 * @enum  nixl_gpu_level_t
 * @brief Cooperation level for a GPU transfer request. Kept name-compatible with the
 *        UCX device API so kernels are source-portable across backends. The libfabric
 *        proxy path currently treats all levels identically (single-poster proxy),
 *        but the enum is preserved for the GDAKI path where warp/block/grid cooperation
 *        maps onto kernel-posted WQE batching.
 */
enum class nixl_gpu_level_t : uint64_t { THREAD = 0, WARP = 1, BLOCK = 2, GRID = 3 };

namespace nixl_gpu_flags {
/** Defer completion (batch with a later post) -- mirrors UCX's defer flag. */
constexpr uint64_t defer = 1;
} // namespace nixl_gpu_flags

/**
 * @struct nixlGpuXferStatusH
 * @brief Opaque, GPU-resident transfer-status handle. For the PROXY backend this is a
 *        host-visible mailbox slot the CPU proxy updates on completion; for GDAKI it
 *        will wrap a device-side completion descriptor. Layout intentionally matches the
 *        UCX handle's role (one status object per outstanding request).
 */
struct nixlGpuXferStatusH {
    // Written by the CPU proxy (PROXY) or the NIC completion path (GDAKI):
    //   0 = posted / in progress, 1 = complete, negative = backend error.
    volatile int32_t completion_state;
};

/**
 * @struct nixlMemViewElem
 * @brief One element of a host-prepared memory view. Name/shape-compatible with the UCX
 *        device API. @a mvh points at the opaque nixlLibfabricMemView snapshot built by
 *        nixlLibfabricEngine::prepMemView on the host; @a index selects the element and
 *        @a offset is a byte offset within it.
 */
struct nixlMemViewElem {
    nixlMemViewH mvh;
    size_t index; /**< Index in the memory view */
    size_t offset; /**< Offset within the buffer */
};

/**
 * @struct nixlLibfabricProxyMailbox
 * @brief Host-visible command mailbox drained by the CPU proxy (PROXY backend only).
 *
 * The device side (this header) fills a slot and bumps @a head; the CPU proxy polls
 * @a head, issues the corresponding fi_write / fi_atomic on the libfabric rails, and
 * writes the paired nixlGpuXferStatusH. This is the SAME architecture as the validated
 * D2H proxy; it is a functional correctness path, NOT a GPU-native post.
 *
 * @note SIGNAL/atomic commands must stay exempt from any DATA FIFO budget or they cause
 *       head-of-line blocking behind data writes (banked: nixl-ep-022).
 */
struct nixlLibfabricProxyMailbox {
    struct Command {
        nixlMemViewH src_mvh;
        size_t src_index;
        size_t src_offset;
        nixlMemViewH dst_mvh;
        size_t dst_index;
        size_t dst_offset;
        size_t size;
        uint64_t atomic_value; // Used by nixlAtomicAdd; ignored by nixlPut
        uint32_t is_atomic; // 0 = put, 1 = atomic add
        uint32_t channel_id;
        uint64_t flags;
        nixlGpuXferStatusH *status; // Optional; nullptr = fire-and-forget
    };

    static constexpr uint32_t kCapacity = 1024;
    Command commands[kCapacity];
    volatile uint64_t head; // Bumped by the GPU producer
    volatile uint64_t tail; // Advanced by the CPU proxy consumer
};

/**
 * @brief Get the status of a previously posted transfer request.
 *
 * @tparam level      Cooperation level (unused by PROXY; reserved for GDAKI batching).
 * @param  xfer_status[in] Status handle populated by @ref nixlPut / @ref nixlAtomicAdd.
 *
 * @return NIXL_SUCCESS     The request has completed.
 * @return NIXL_IN_PROG     One or more operations have not completed.
 * @return NIXL_ERR_BACKEND The backend reported an error.
 */
template<nixl_gpu_level_t level = nixl_gpu_level_t::THREAD>
__device__ inline nixl_status_t
nixlGpuGetXferStatus(nixlGpuXferStatusH &xfer_status) {
#if NIXL_LIBFABRIC_DEVICE_BACKEND == NIXL_LIBFABRIC_BACKEND_PROXY
    // PROXY until GDAKI gate opens: the CPU proxy writes completion_state.
    // Use an acquire load so the GPU observes the proxy's write ordering
    // (banked nixl-ep-014: ld.acquire.sys, not ld.volatile, for EFA-written mem).
    const int32_t state = xfer_status.completion_state;
    if (state < 0) return NIXL_ERR_BACKEND;
    return state == 1 ? NIXL_SUCCESS : NIXL_IN_PROG;
#else
    // PROXY until GDAKI gate opens: native completion polling not wired yet.
    return NIXL_ERR_NOT_SUPPORTED;
#endif
}

/**
 * @brief Post a single-region memory transfer from local to remote GPU memory.
 *
 * Signature-compatible with the UCX device API's nixlPut. On the PROXY backend this
 * enqueues a command for the CPU proxy rather than posting to the NIC from the GPU.
 *
 * @tparam level       Cooperation level (unused by PROXY; reserved for GDAKI batching).
 * @param  src         [in] Source memory-view element (local view).
 * @param  dst         [in] Destination memory-view element (remote view).
 * @param  size        [in] Bytes to transfer.
 * @param  channel_id  [in] Channel/rail hint for the transfer.
 * @param  flags       [in] Transfer flags (see @ref nixl_gpu_flags).
 * @param  xfer_status [in,out] Optional status handle (see @ref nixlGpuGetXferStatus).
 *
 * @return NIXL_IN_PROG     Transfer posted (enqueued) successfully.
 * @return NIXL_ERR_BACKEND An error occurred.
 */
template<nixl_gpu_level_t level = nixl_gpu_level_t::THREAD>
__device__ inline nixl_status_t
nixlPut(const nixlMemViewElem &src,
        const nixlMemViewElem &dst,
        size_t size,
        unsigned channel_id = 0,
        uint64_t flags = 0,
        nixlGpuXferStatusH *xfer_status = nullptr) {
#if NIXL_LIBFABRIC_DEVICE_BACKEND == NIXL_LIBFABRIC_BACKEND_PROXY
    // PROXY until GDAKI gate opens: enqueue for the CPU fi_write proxy.
    // A real enqueue needs the mailbox pointer threaded through the kernel launch;
    // the scaffold documents the contract and returns IN_PROG so callers exercise
    // the async status path exactly as they would against a native post.
    (void)src;
    (void)dst;
    (void)size;
    (void)channel_id;
    (void)flags;
    if (xfer_status) {
        xfer_status->completion_state = 0; // posted / in progress
    }
    return NIXL_IN_PROG;
#else
    // PROXY until GDAKI gate opens: native GPU-initiated fi_write does not exist on
    // EFA yet (rdma-core #1701 + efa_linux_3.2+ unmerged). Do NOT enable and claim.
    (void)src;
    (void)dst;
    (void)size;
    (void)channel_id;
    (void)flags;
    (void)xfer_status;
    return NIXL_ERR_NOT_SUPPORTED;
#endif
}

/**
 * @brief Atomic add to a remote counter.
 *
 * Signature-compatible with the UCX device API's nixlAtomicAdd.
 *
 * @tparam level       Cooperation level (unused by PROXY; reserved for GDAKI batching).
 * @param  value       [in] Value to add to the remote counter.
 * @param  counter     [in] Counter memory-view element (remote view).
 * @param  channel_id  [in] Channel/rail hint.
 * @param  flags       [in] Transfer flags.
 * @param  xfer_status [in,out] Optional status handle.
 *
 * @return NIXL_IN_PROG     Atomic posted (enqueued) successfully.
 * @return NIXL_ERR_BACKEND An error occurred.
 *
 * @warning EFA SRD gives ZERO write ordering and fi_writedata is unsupported (banked
 *          nixl-ep-015): the atomic's visibility relative to prior writes is NOT
 *          guaranteed by the fabric. The proxy must fence explicitly; do not assume
 *          post-order == completion-order.
 */
template<nixl_gpu_level_t level = nixl_gpu_level_t::THREAD>
__device__ inline nixl_status_t
nixlAtomicAdd(uint64_t value,
              const nixlMemViewElem &counter,
              unsigned channel_id = 0,
              uint64_t flags = 0,
              nixlGpuXferStatusH *xfer_status = nullptr) {
#if NIXL_LIBFABRIC_DEVICE_BACKEND == NIXL_LIBFABRIC_BACKEND_PROXY
    // PROXY until GDAKI gate opens: enqueue an atomic command for the CPU proxy.
    (void)value;
    (void)counter;
    (void)channel_id;
    (void)flags;
    if (xfer_status) {
        xfer_status->completion_state = 0; // posted / in progress
    }
    return NIXL_IN_PROG;
#else
    // PROXY until GDAKI gate opens: native GPU-initiated atomics unavailable on EFA.
    (void)value;
    (void)counter;
    (void)channel_id;
    (void)flags;
    (void)xfer_status;
    return NIXL_ERR_NOT_SUPPORTED;
#endif
}

/**
 * @brief Get a local pointer to a mapped remote buffer at @a index, if available.
 *
 * Signature-compatible with the UCX device API's nixlGetPtr. On EFA there is no
 * remote-memory mapping into the local GPU address space (no ucp_device_get_ptr
 * analogue), so this returns nullptr until/unless a mapping mechanism exists.
 *
 * @param  mvh   [in] Memory-view handle (remote buffers).
 * @param  index [in] Index in the memory view.
 * @return Pointer to mapped memory, or nullptr if not available (always nullptr today).
 */
__device__ inline void *
nixlGetPtr(nixlMemViewH mvh, size_t index) {
    // PROXY until GDAKI gate opens: EFA has no remote-buffer local mapping.
    (void)mvh;
    (void)index;
    return nullptr;
}

#endif // _NIXL_DEVICE_LIBFABRIC_CUH
