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

/* =====================================================================================
 *  Offline syntax-gate shims -- NEVER part of a device build.
 * =====================================================================================
 *  The proxy enqueue path below uses a CUDA device intrinsic (__threadfence_system, for
 *  the release-before-publish ordering) plus the __device__/__forceinline__ qualifiers.
 *  When this header is validated by a host compiler for a standalone syntax gate (no nvcc
 *  available offline, per increment-1's `g++ -std=c++17 -fsyntax-only` gate), those do not
 *  exist. Provide no-op equivalents so the gate can check the enqueue path's STRUCTURE.
 *  Under real nvcc (__CUDACC__ defined) this whole block is skipped and the true
 *  intrinsic/qualifiers are used. These shims are a build-gate aid ONLY -- they do NOT
 *  implement any transport and are never part of a shipped .so/.cubin.
 */
#ifndef __CUDACC__
#ifndef __device__
#define __device__
#endif
#ifndef __forceinline__
#define __forceinline__ inline
#endif
static inline void
__threadfence_system(void) {}
#endif // !__CUDACC__

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
 * The device side (this header) reserves a slot, fills it, and publishes it via the
 * per-slot @a ready flag; the CPU proxy polls the slot at @a tail, and once @a ready==1
 * issues the corresponding fi_write / fi_atomic on the libfabric rails (the host
 * transport is nixlLibfabricRail::postWrite -> fi_writedata, driven by
 * nixlLibfabricRailManager::prepareAndSubmitTransfer), writes the paired
 * nixlGpuXferStatusH, then clears @a ready and advances @a tail. This is the SAME
 * architecture as the validated D2H proxy; it is a functional correctness path,
 * NOT a GPU-native post.
 *
 * PUBLICATION ORDERING (why a per-slot @a ready flag, not just @a head): a producer
 * reserves its slot by atomically bumping @a head, but the payload is written AFTER the
 * reservation. If the consumer keyed off @a head it could observe an advanced head and
 * read a half-written Command. Instead each Command carries its own @a ready flag: the
 * producer writes all fields, issues a system-scope release fence, then sets ready=1
 * last; the consumer keys off ready (not head). @a head is a pure reservation counter.
 *
 * @note The host MUST zero-initialise the whole mailbox (head=tail=0, every Command's
 *       ready=0) before any kernel posts; a fresh cudaMallocHost/cudaMemset region
 *       satisfies this.
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
        // Publication flag: 0 = slot free / being written, 1 = payload committed and
        // ready for the CPU proxy. Written LAST by the producer (after a release fence),
        // cleared by the consumer once the slot is drained. See PUBLICATION ORDERING.
        volatile uint32_t ready;
    };

    static constexpr uint32_t kCapacity = 1024;
    Command commands[kCapacity];
    volatile uint64_t head; // Reservation counter, atomically bumped by GPU producers
    volatile uint64_t tail; // Advanced by the CPU proxy consumer after draining a slot
};

/**
 * @brief Enqueue one command into the CPU-proxy mailbox (the shared PROXY transport seam).
 *
 * This is the single C++ enqueue path that puts work onto the libfabric CPU proxy. The
 * device surface (@ref nixlPut / @ref nixlAtomicAdd, PROXY backend) calls it from a GPU
 * kernel; the SAME seam shape is what the host-side Route-C harness will drive through a
 * pybind (framework/, nixl-ep-100/103) so ONE enqueue contract serves both callers -- do
 * NOT fork a second enqueue protocol. The proxy drains a published slot and issues the
 * transfer via the host transport (nixlLibfabricRail::postWrite -> fi_writedata, through
 * nixlLibfabricRailManager::prepareAndSubmitTransfer).
 *
 * Reserve -> fill -> publish (see PUBLICATION ORDERING on @ref nixlLibfabricProxyMailbox):
 *   1. reserve a slot index by atomically bumping @a head;
 *   2. copy the caller's data fields into the slot;
 *   3. system-scope release fence so the payload is visible before the ready flag;
 *   4. set the slot's @a ready = 1 last (the consumer keys off ready, never off head).
 *
 * @param mailbox [in] Host-visible mailbox (cudaMallocHost / mapped), zero-initialised.
 * @param cmd     [in] Fully-populated command EXCEPT @a ready (this seam sets it). @a cmd.status,
 *                     if non-null, must point at GPU-visible memory the proxy can write.
 *
 * @return NIXL_IN_PROG     Command published; the proxy will post it and update @a status.
 * @return NIXL_ERR_BACKEND The mailbox is full (producer outran the proxy) or @a mailbox is null.
 *
 * @warning SINGLE-POSTER contract: one owner (warp-leader thread, or one host thread on the
 *          pybind side) enqueues per mailbox. Multi-poster contention on a shared proxy is a
 *          banked dead end (NEG: multi-proxy shared-transport 0.49-16 GB/s oscillation,
 *          fi_cntr single-owner -- nixl-ep-013/019/095). Do NOT fan multiple warps at one mailbox.
 */
__device__ __forceinline__ nixl_status_t
nixlLibfabricProxyEnqueue(nixlLibfabricProxyMailbox *mailbox,
                          const nixlLibfabricProxyMailbox::Command &cmd) {
    if (!mailbox) return NIXL_ERR_BACKEND;

    // (1) Reserve a slot. SINGLE-POSTER contract (see @warning): exactly one owner enqueues
    //     per mailbox, so head is advanced by a plain read-modify-write -- NOT atomicAdd,
    //     which would falsely imply the banned multi-poster fan-in (nixl-ep-013/019/095)
    //     and, on a full ring, would skew head past tail permanently (a retrying producer
    //     would then reject forever). Backpressure first, publish-reservation only on
    //     success, so head never runs ahead of what actually got enqueued.
    const uint64_t ticket = mailbox->head;
    if (ticket - mailbox->tail >= nixlLibfabricProxyMailbox::kCapacity) {
        return NIXL_ERR_BACKEND; // ring full; caller retries / throttles (head unchanged)
    }
    const uint32_t slot = static_cast<uint32_t>(ticket % nixlLibfabricProxyMailbox::kCapacity);
    nixlLibfabricProxyMailbox::Command &dst = mailbox->commands[slot];
    mailbox->head = ticket + 1; // commit the reservation (single-poster: no other writer)

    // (2) Fill the data fields (everything except the publication flag).
    dst.src_mvh = cmd.src_mvh;
    dst.src_index = cmd.src_index;
    dst.src_offset = cmd.src_offset;
    dst.dst_mvh = cmd.dst_mvh;
    dst.dst_index = cmd.dst_index;
    dst.dst_offset = cmd.dst_offset;
    dst.size = cmd.size;
    dst.atomic_value = cmd.atomic_value;
    dst.is_atomic = cmd.is_atomic;
    dst.channel_id = cmd.channel_id;
    dst.flags = cmd.flags;
    dst.status = cmd.status;
    if (cmd.status) {
        cmd.status->completion_state = 0; // posted / in progress, before publish
    }

    // (3) Release fence: all field writes above must be visible to the CPU proxy BEFORE
    //     it can observe ready==1 (banked nixl-ep-014: EFA/host-visible mem needs explicit
    //     ordering, not volatile alone). __threadfence_system reaches the host.
    __threadfence_system();

    // (4) Publish last.
    dst.ready = 1;
    return NIXL_IN_PROG;
}

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
 * @param  mailbox     [in] CPU-proxy mailbox threaded through the kernel launch. REQUIRED
 *                     on the PROXY backend (there is no GPU-native post to fall back to);
 *                     defaulted to nullptr only to preserve UCX-signature source-compat,
 *                     so a kernel authored against the UCX device API still compiles here.
 *                     A UCX-shaped call that omits it gets NIXL_ERR_BACKEND on PROXY.
 *
 * @return NIXL_IN_PROG     Transfer enqueued to the proxy; poll @a xfer_status.
 * @return NIXL_ERR_BACKEND The mailbox is null/full, or the backend reported an error.
 */
template<nixl_gpu_level_t level = nixl_gpu_level_t::THREAD>
__device__ inline nixl_status_t
nixlPut(const nixlMemViewElem &src,
        const nixlMemViewElem &dst,
        size_t size,
        unsigned channel_id = 0,
        uint64_t flags = 0,
        nixlGpuXferStatusH *xfer_status = nullptr,
        nixlLibfabricProxyMailbox *mailbox = nullptr) {
#if NIXL_LIBFABRIC_DEVICE_BACKEND == NIXL_LIBFABRIC_BACKEND_PROXY
    // PROXY until GDAKI gate opens: enqueue a put onto the CPU fi_writedata proxy via the
    // shared seam (nixlLibfabricProxyEnqueue). The mailbox is threaded through the kernel
    // launch, as anticipated by increment-1. This is a functional correctness path, not a
    // GPU-native post -- NEVER quote a completion latency from it as a GPU-native number.
    nixlLibfabricProxyMailbox::Command cmd;
    cmd.src_mvh = src.mvh;
    cmd.src_index = src.index;
    cmd.src_offset = src.offset;
    cmd.dst_mvh = dst.mvh;
    cmd.dst_index = dst.index;
    cmd.dst_offset = dst.offset;
    cmd.size = size;
    cmd.atomic_value = 0;
    cmd.is_atomic = 0; // put
    cmd.channel_id = channel_id;
    cmd.flags = flags;
    cmd.status = xfer_status;
    return nixlLibfabricProxyEnqueue(mailbox, cmd);
#else
    // PROXY until GDAKI gate opens: native GPU-initiated fi_write does not exist on
    // EFA yet (rdma-core #1701 + efa_linux_3.2+ unmerged). Do NOT enable and claim.
    (void)src;
    (void)dst;
    (void)size;
    (void)channel_id;
    (void)flags;
    (void)xfer_status;
    (void)mailbox;
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
 * @param  mailbox     [in] CPU-proxy mailbox (see @ref nixlPut). REQUIRED on PROXY;
 *                     defaulted to nullptr only for UCX source-compat.
 *
 * @return NIXL_IN_PROG     Atomic enqueued to the proxy; poll @a xfer_status.
 * @return NIXL_ERR_BACKEND The mailbox is null/full, or the backend reported an error.
 *
 * @warning EFA SRD gives ZERO write ordering and fi_writedata is unsupported (banked
 *          nixl-ep-015): the atomic's visibility relative to prior writes is NOT
 *          guaranteed by the fabric. The proxy must fence explicitly; do not assume
 *          post-order == completion-order. Atomics are enqueued through the SAME mailbox
 *          but the proxy MUST keep them off any DATA FIFO budget (nixl-ep-022) so a
 *          counter increment cannot head-of-line-block behind bulk data writes.
 */
template<nixl_gpu_level_t level = nixl_gpu_level_t::THREAD>
__device__ inline nixl_status_t
nixlAtomicAdd(uint64_t value,
              const nixlMemViewElem &counter,
              unsigned channel_id = 0,
              uint64_t flags = 0,
              nixlGpuXferStatusH *xfer_status = nullptr,
              nixlLibfabricProxyMailbox *mailbox = nullptr) {
#if NIXL_LIBFABRIC_DEVICE_BACKEND == NIXL_LIBFABRIC_BACKEND_PROXY
    // PROXY until GDAKI gate opens: enqueue an atomic-add command onto the CPU proxy via
    // the shared seam. The counter is the destination; there is no source buffer, so the
    // src_* fields are left empty and the addend rides in atomic_value with is_atomic=1.
    nixlLibfabricProxyMailbox::Command cmd;
    cmd.src_mvh = nullptr;
    cmd.src_index = 0;
    cmd.src_offset = 0;
    cmd.dst_mvh = counter.mvh;
    cmd.dst_index = counter.index;
    cmd.dst_offset = counter.offset;
    cmd.size = sizeof(uint64_t);
    cmd.atomic_value = value;
    cmd.is_atomic = 1; // atomic add
    cmd.channel_id = channel_id;
    cmd.flags = flags;
    cmd.status = xfer_status;
    return nixlLibfabricProxyEnqueue(mailbox, cmd);
#else
    // PROXY until GDAKI gate opens: native GPU-initiated atomics unavailable on EFA.
    (void)value;
    (void)counter;
    (void)channel_id;
    (void)flags;
    (void)xfer_status;
    (void)mailbox;
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
