/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <algorithm>
#include <atomic>
#include <barrier>
#include <cerrno>
#include <charconv>
#include <chrono>
#include <cstring>
#include <exception>
#include <latch>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <system_error>
#include <utility>

#include <pthread.h>
#include <sched.h>

#include <cuda_runtime.h>

#include "common/backend.h"
#include "common/nixl_log.h"
#include "gds_batch_engine.h"

namespace {
/** Setting the default values to check the batch limit */
constexpr unsigned DEFAULT_BATCH_LIMIT = 128;
/** Setting the max request size to 16 MB */
constexpr unsigned DEFAULT_MAX_REQUEST_SIZE = 16 * 1024 * 1024; // 16MB
/** Create a batch pool of size 16 */
constexpr unsigned DEFAULT_BATCH_POOL_SIZE = 16;
/** Submit batches from four persistent workers by default. */
constexpr unsigned DEFAULT_SUBMIT_THREADS = 4;

size_t
ceilDiv(size_t value, size_t divisor) {
    return (value / divisor) + ((value % divisor) != 0);
}

size_t
getBatchCount(size_t request_count, size_t batch_limit, size_t submit_threads) {
    return std::max(ceilDiv(request_count, batch_limit),
                    std::min(request_count, submit_threads));
}

std::string_view
trim(std::string_view value) {
    constexpr std::string_view whitespace = " \t\n\r";
    const size_t first = value.find_first_not_of(whitespace);
    if (first == std::string_view::npos) {
        return {};
    }
    const size_t last = value.find_last_not_of(whitespace);
    return value.substr(first, last - first + 1);
}

std::vector<unsigned int>
parseSubmitCpus(const std::string &value, unsigned int submit_threads) {
    std::vector<unsigned int> cpus;
    if (trim(value).empty()) {
        return cpus;
    }

    cpu_set_t allowed_cpus;
    CPU_ZERO(&allowed_cpus);
    if (sched_getaffinity(0, sizeof(allowed_cpus), &allowed_cpus) != 0) {
        throw std::system_error(errno, std::generic_category(), "sched_getaffinity");
    }

    size_t start = 0;
    while (start <= value.size()) {
        const size_t comma = value.find(',', start);
        const size_t end = (comma == std::string::npos) ? value.size() : comma;
        const std::string_view token = trim(std::string_view(value).substr(start, end - start));
        if (token.empty()) {
            throw std::invalid_argument("GDS: submit_cpus contains an empty CPU ID");
        }

        unsigned int cpu = 0;
        const auto [ptr, error] = std::from_chars(token.data(), token.data() + token.size(), cpu);
        if (error != std::errc() || ptr != token.data() + token.size()) {
            throw std::invalid_argument("GDS: submit_cpus must be comma-separated CPU IDs");
        }
        if (cpu >= CPU_SETSIZE || !CPU_ISSET(static_cast<int>(cpu), &allowed_cpus)) {
            throw std::invalid_argument("GDS: submit_cpus contains a CPU unavailable to this process");
        }
        if (std::find(cpus.begin(), cpus.end(), cpu) != cpus.end()) {
            throw std::invalid_argument("GDS: submit_cpus must not contain duplicate CPU IDs");
        }
        cpus.push_back(cpu);

        if (comma == std::string::npos) {
            break;
        }
        start = comma + 1;
    }

    if (cpus.size() != submit_threads) {
        throw std::invalid_argument("GDS: submit_cpus must contain one CPU ID per submit thread");
    }

    // TODO: Select CPUs automatically from the backing devices' blk-mq CPU
    // maps and NUMA topology. This must account for dm/md and multi-device
    // filesystems before it can safely replace explicit submit_cpus.
    return cpus;
}

class GdsWorkerAffinity : public tf::WorkerInterface {
public:
    GdsWorkerAffinity(std::vector<unsigned int> cpus, unsigned int worker_count)
        : cpus_(std::move(cpus)),
          started_(static_cast<std::ptrdiff_t>(worker_count)) {}

    void
    scheduler_prologue(tf::Worker &worker) override {
        const cudaError_t cuda_error = cudaSetDevice(0);
        if (cuda_error != cudaSuccess) {
            int expected = cudaSuccess;
            cuda_error_.compare_exchange_strong(expected, cuda_error);
        }

        int affinity_error = 0;
        if (!cpus_.empty()) {
            cpu_set_t cpu_set;
            CPU_ZERO(&cpu_set);
            CPU_SET(static_cast<int>(cpus_[worker.id()]), &cpu_set);
            affinity_error = pthread_setaffinity_np(
                worker.thread().native_handle(), sizeof(cpu_set), &cpu_set);
        }
        if (affinity_error != 0) {
            int expected = 0;
            affinity_error_.compare_exchange_strong(expected, affinity_error);
        }
        started_.count_down();
    }

    void
    scheduler_epilogue(tf::Worker &, std::exception_ptr) override {}

    int
    waitForStartup(cudaError_t &cuda_error) {
        started_.wait();
        cuda_error = static_cast<cudaError_t>(cuda_error_.load());
        return affinity_error_.load();
    }

private:
    const std::vector<unsigned int> cpus_;
    std::latch started_;
    std::atomic<int> affinity_error_{0};
    std::atomic<int> cuda_error_{cudaSuccess};
};
} // namespace

nixlGdsIOBatch::nixlGdsIOBatch(unsigned int size)
    : io_batch_events(std::make_unique<CUfileIOEvents_t[]>(size)),
      io_batch_params(std::make_unique<CUfileIOParams_t[]>(size)),
      max_reqs(size) {

    const CUfileError_t err = cuFileBatchIOSetUp(&batch_handle, size);
    if (err.err != 0) {
        NIXL_ERROR << "Error in setting up Batch";
        init_err = err;
    }
}

nixlGdsIOBatch::~nixlGdsIOBatch() {
    if (active) {
        NIXL_ERROR << "GDS: destroying an active batch; canceling outstanding I/O";
        cancelBatch();
    }
    if (batch_handle != nullptr) {
        cuFileBatchIODestroy(batch_handle);
    }
}

nixl_status_t
nixlGdsIOBatch::addToBatch(CUfileHandle_t fh,
                           void *buffer,
                           size_t size,
                           size_t file_offset,
                           size_t ptr_offset,
                           CUfileOpcode_t type) {
    if (!isValid() || active || batch_size >= max_reqs) {
        return NIXL_ERR_BACKEND;
    }

    CUfileIOParams_t *params = &io_batch_params[batch_size];
    *params = {};
    params->mode = CUFILE_BATCH;
    params->fh = fh;
    params->u.batch.devPtr_base = buffer;
    params->u.batch.file_offset = file_offset;
    params->u.batch.devPtr_offset = ptr_offset;
    params->u.batch.size = size;
    params->opcode = type;
    params->cookie = params;
    batch_size++;

    return NIXL_SUCCESS;
}

nixl_status_t
nixlGdsIOBatch::cancelBatch() {
    if (!active) {
        return NIXL_SUCCESS;
    }
    const CUfileError_t err = cuFileBatchIOCancel(batch_handle);
    if (err.err != 0) {
        NIXL_ERROR << "Error in canceling batch";
        return NIXL_ERR_BACKEND;
    }
    active = false;
    current_status = NIXL_ERR_CANCELED;
    return NIXL_SUCCESS;
}

nixl_status_t
nixlGdsIOBatch::submitBatch(int flags) {
    if (!isValid() || batch_size == 0) {
        return NIXL_ERR_INVALID_PARAM;
    }
    const CUfileError_t err =
        cuFileBatchIOSubmit(batch_handle, batch_size, io_batch_params.get(), flags);
    if (err.err != 0) {
        NIXL_ERROR << "Error submitting GDS batch";
        current_status = NIXL_ERR_BACKEND;
        return NIXL_ERR_BACKEND;
    }
    active = true;
    current_status = NIXL_IN_PROG;
    return NIXL_SUCCESS;
}

nixl_status_t
nixlGdsIOBatch::checkStatus() {
    if (current_status != NIXL_IN_PROG) {
        return current_status;
    }

    if (entries_completed > batch_size) {
        current_status = NIXL_ERR_UNKNOWN;
        return current_status;
    }

    unsigned int nr = batch_size - entries_completed;
    const CUfileError_t errBatch =
        cuFileBatchIOGetStatus(batch_handle, 0, &nr, io_batch_events.get(), nullptr);
    if (errBatch.err != 0) {
        NIXL_ERROR << "Error in IO Batch Get Status";
        current_status = NIXL_ERR_BACKEND;
        return current_status;
    }

    if (nr > batch_size - entries_completed) {
        current_status = NIXL_ERR_UNKNOWN;
        return current_status;
    }

    for (unsigned int i = 0; i < nr; ++i) {
        const CUfileIOEvents_t &event = io_batch_events[i];
        if (event.status != CUFILE_COMPLETE || event.cookie == nullptr) {
            NIXL_ERROR << "GDS batch entry failed with status " << event.status
                       << ", result " << static_cast<ssize_t>(event.ret);
            current_status = NIXL_ERR_BACKEND;
            return current_status;
        }

        const auto *params = static_cast<const CUfileIOParams_t *>(event.cookie);
        if (event.ret != params->u.batch.size) {
            NIXL_ERROR << "GDS batch entry completed " << event.ret << " of "
                       << params->u.batch.size << " bytes";
            current_status = NIXL_ERR_BACKEND;
            return current_status;
        }
    }

    entries_completed += nr;
    if (entries_completed == batch_size) {
        active = false;
        current_status = NIXL_SUCCESS;
    } else {
        current_status = NIXL_IN_PROG;
    }

    return current_status;
}

void
nixlGdsIOBatch::reset() {
    if (active) {
        NIXL_ERROR << "GDS: attempted to reset an active batch";
        return;
    }
    entries_completed = 0;
    batch_size = 0;
    current_status = NIXL_ERR_NOT_POSTED;
}

nixlGdsBatchReqH::~nixlGdsBatchReqH() {
    if (host_transfer.valid()) {
        host_transfer.wait();
    }
}

nixlGdsBatchEngine::nixlGdsBatchEngine(const nixlBackendInitParams *init_params)
    : nixlGdsEngine(init_params) {
    // Base ctor opened the cuFile driver; bail if that failed.
    if (this->initErr) {
        return;
    }

    try {
        nixl_b_params_t *custom_params = init_params->customParams;
        batch_pool_size_ =
            nixl::getBackendParamDefaulted(custom_params, "batch_pool_size", DEFAULT_BATCH_POOL_SIZE);
        batch_limit_ =
            nixl::getBackendParamDefaulted(custom_params, "batch_limit", DEFAULT_BATCH_LIMIT);
        max_request_size_ = nixl::getBackendParamDefaulted(
            custom_params, "max_request_size", DEFAULT_MAX_REQUEST_SIZE);
        submit_threads_ =
            nixl::getBackendParamDefaulted(custom_params, "submit_threads", DEFAULT_SUBMIT_THREADS);

        if (batch_pool_size_ == 0 || batch_limit_ == 0 || max_request_size_ == 0 ||
            submit_threads_ == 0) {
            throw std::invalid_argument(
                "GDS: batch_pool_size, batch_limit, max_request_size, and submit_threads "
                "must be greater than zero");
        }
        if (batch_pool_size_ < submit_threads_) {
            throw std::invalid_argument(
                "GDS: batch_pool_size must be at least submit_threads");
        }

        const std::string submit_cpu_config =
            nixl::getBackendParamDefaulted(custom_params, "submit_cpus", std::string());
        const std::vector<unsigned int> submit_cpus =
            parseSubmitCpus(submit_cpu_config, submit_threads_);

        batch_pool_.reserve(batch_pool_size_);
        batch_storage_.reserve(batch_pool_size_);
        for (unsigned int i = 0; i < batch_pool_size_; i++) {
            auto batch = std::make_unique<nixlGdsIOBatch>(batch_limit_);
            if (!batch->isValid()) {
                throw std::runtime_error("GDS: failed to initialize cuFile batch pool");
            }
            batch_pool_.push_back(batch.get());
            batch_storage_.push_back(std::move(batch));
        }

        auto worker_affinity =
            std::make_shared<GdsWorkerAffinity>(submit_cpus, submit_threads_);
        executor_ = std::make_unique<tf::Executor>(submit_threads_, worker_affinity);
        cudaError_t cuda_error = cudaSuccess;
        const int affinity_error = worker_affinity->waitForStartup(cuda_error);
        if (cuda_error != cudaSuccess) {
            throw std::runtime_error(std::string("GDS worker CUDA initialization: ") +
                                     cudaGetErrorString(cuda_error));
        }
        if (affinity_error != 0) {
            throw std::system_error(
                affinity_error, std::generic_category(), "GDS worker CPU affinity");
        }

        NIXL_DEBUG << "GDS: submit threads=" << submit_threads_
                   << (submit_cpus.empty() ? " (unpinned)" : " (CPU-pinned)");
    }
    catch (const std::exception &e) {
        NIXL_ERROR << e.what();
        executor_.reset();
        this->initErr = true;
    }
}

nixlGdsBatchEngine::~nixlGdsBatchEngine() {
    executor_.reset();
    batch_pool_.clear();
    batch_storage_.clear();
}

nixlGdsIOBatch *
nixlGdsBatchEngine::getBatchFromPool(unsigned int /*size*/) const {
    const std::lock_guard<std::mutex> lock(batch_pool_lock_);
    if (!batch_pool_.empty()) {
        nixlGdsIOBatch *batch = batch_pool_.back();
        batch_pool_.pop_back();
        batch->reset();
        return batch;
    }
    // Pool exhausted - don't create new batches in the data path.
    return nullptr;
}

void
nixlGdsBatchEngine::returnBatchToPool(nixlGdsIOBatch *batch) const {
    const std::lock_guard<std::mutex> lock(batch_pool_lock_);
    batch_pool_.push_back(batch);
}

nixl_status_t
nixlGdsBatchEngine::finalizePrep(std::vector<GdsXferReq> &&reqs,
                                 nixlBackendReqH *&handle) const {
    auto gds_handle = std::make_unique<nixlGdsBatchReqH>();

    size_t chunk_count = 0;
    bool can_reuse_requests = true;
    const size_t max_request_size = max_request_size_;
    for (const GdsXferReq &req : reqs) {
        if (!req.addr) {
            return NIXL_ERR_INVALID_PARAM;
        }

        const size_t chunks =
            (req.size / max_request_size) + ((req.size % max_request_size) != 0);
        can_reuse_requests &= (chunks == 1);
        if (chunks > std::numeric_limits<size_t>::max() - chunk_count) {
            return NIXL_ERR_INVALID_PARAM;
        }
        chunk_count += chunks;
    }

    if (chunk_count == 0) {
        return NIXL_ERR_INVALID_PARAM;
    }

    if (can_reuse_requests) {
        gds_handle->request_list = std::move(reqs);
    } else {
        // Split large transfers into multiple requests bounded by max_request_size.
        gds_handle->request_list.reserve(chunk_count);
        for (const GdsXferReq &req : reqs) {
            size_t remaining_size = req.size;
            size_t current_offset = 0;
            while (remaining_size > 0) {
                const size_t request_size = std::min(remaining_size, max_request_size);

                GdsXferReq chunk;
                chunk.addr = req.addr;
                chunk.size = request_size;
                chunk.file_offset = req.file_offset + current_offset;
                chunk.ptr_offset = req.ptr_offset + current_offset;
                chunk.host_memory = req.host_memory;
                chunk.fh = req.fh;
                chunk.op = req.op;
                gds_handle->request_list.push_back(chunk);

                remaining_size -= request_size;
                current_offset += request_size;
            }
        }
    }

    gds_handle->host_memory = gds_handle->request_list.front().host_memory;
    if (std::any_of(gds_handle->request_list.begin(),
                    gds_handle->request_list.end(),
                    [&](const GdsXferReq &req) {
                        return req.host_memory != gds_handle->host_memory;
                    })) {
        return NIXL_ERR_INVALID_PARAM;
    }

    if (gds_handle->host_memory) {
        for (GdsXferReq &req : gds_handle->request_list) {
            GdsXferReq *captured_req = &req;
            gds_handle->host_taskflow.emplace(
                [captured_req, status = &gds_handle->host_status]() {
                    const nixl_status_t result = runGdsCuFileOp(*captured_req, "GDS");
                    if (result != NIXL_SUCCESS) {
                        status->store(result);
                    }
                });
        }
        handle = gds_handle.release();
        return NIXL_SUCCESS;
    }

    const size_t request_count = gds_handle->request_list.size();
    const size_t batch_count = getBatchCount(request_count, batch_limit_, submit_threads_);
    if (batch_count > batch_pool_size_) {
        NIXL_ERROR << "GDS: transfer requires " << batch_count << " batches but the pool has "
                   << batch_pool_size_;
        return NIXL_ERR_BACKEND;
    }
    gds_handle->batch_io_list.reserve(batch_count);

    handle = gds_handle.release();
    return NIXL_SUCCESS;
}

nixl_status_t
nixlGdsBatchEngine::createAndSubmitBatch(const std::vector<GdsXferReq> &requests,
                                         size_t start_idx,
                                         size_t batch_size,
                                         nixlGdsIOBatch *&batch_out) const {
    batch_out = nullptr;
    nixlGdsIOBatch *batch = getBatchFromPool(batch_size);
    if (!batch) {
        NIXL_ERROR << "GDS batch pool exhausted";
        return NIXL_ERR_BACKEND;
    }

    for (size_t i = 0; i < batch_size; i++) {
        const auto &req = requests[start_idx + i];
        if (!req.addr || !req.fh) {
            returnBatchToPool(batch);
            return NIXL_ERR_INVALID_PARAM;
        }

        nixl_status_t status =
            batch->addToBatch(req.fh,
                              req.addr,
                              req.size,
                              req.file_offset,
                              req.ptr_offset,
                              req.op);
        if (status != NIXL_SUCCESS) {
            returnBatchToPool(batch);
            return NIXL_ERR_INVALID_PARAM;
        }
    }

    nixl_status_t status = batch->submitBatch(0);
    if (status != NIXL_SUCCESS) {
        returnBatchToPool(batch);
        return NIXL_ERR_BACKEND;
    }

    batch_out = batch;
    return NIXL_SUCCESS;
}

nixl_status_t
nixlGdsBatchEngine::cancelAndReclaimBatches(
    std::vector<nixlGdsIOBatch *> &batch_list) const {
    nixl_status_t status = NIXL_SUCCESS;
    auto keep = batch_list.begin();

    for (nixlGdsIOBatch *batch : batch_list) {
        if (batch == nullptr) {
            continue;
        }
        if (batch->cancelBatch() == NIXL_SUCCESS) {
            returnBatchToPool(batch);
        } else {
            *keep++ = batch;
            status = NIXL_ERR_BACKEND;
        }
    }

    batch_list.erase(keep, batch_list.end());
    return status;
}

nixl_status_t
nixlGdsBatchEngine::postXfer(const nixl_xfer_op_t &operation,
                             const nixl_meta_dlist_t &local,
                             const nixl_meta_dlist_t &remote,
                             const std::string &remote_agent,
                             nixlBackendReqH *&handle,
                             const nixl_opt_b_args_t *opt_args) const {
    auto *gds_handle = static_cast<nixlGdsBatchReqH *>(handle);

    if (gds_handle->request_list.empty()) {
        NIXL_ERROR << "Empty request list";
        return NIXL_ERR_INVALID_PARAM;
    }
    if (!gds_handle->batch_io_list.empty()) {
        return NIXL_ERR_REPOST_ACTIVE;
    }
    if (!executor_) {
        return NIXL_ERR_BACKEND;
    }

    if (gds_handle->host_memory) {
        if (gds_handle->host_transfer.valid()) {
            return NIXL_ERR_REPOST_ACTIVE;
        }
        gds_handle->host_status.store(NIXL_SUCCESS);
        try {
            gds_handle->host_transfer = executor_->run(gds_handle->host_taskflow);
        }
        catch (const std::exception &e) {
            NIXL_ERROR << "GDS: failed to run host-memory transfer: " << e.what();
            gds_handle->host_status.store(NIXL_ERR_BACKEND);
            return NIXL_ERR_BACKEND;
        }
        return NIXL_IN_PROG;
    }

    const auto &request_list = gds_handle->request_list;
    const size_t batch_count = getBatchCount(request_list.size(), batch_limit_, submit_threads_);
    if (batch_count > batch_pool_size_) {
        return NIXL_ERR_BACKEND;
    }

    gds_handle->overall_status = NIXL_SUCCESS;
    gds_handle->batch_io_list.assign(batch_count, nullptr);

    std::atomic<nixl_status_t> first_error{NIXL_SUCCESS};
    const size_t requests_per_batch = request_list.size() / batch_count;
    const size_t batches_with_extra_request = request_list.size() % batch_count;
    const size_t active_workers = std::min(batch_count, static_cast<size_t>(submit_threads_));
    std::vector<size_t> batch_starts(batch_count);
    std::vector<size_t> batch_sizes(batch_count);
    size_t current_req = 0;
    for (size_t i = 0; i < batch_count; ++i) {
        batch_starts[i] = current_req;
        batch_sizes[i] = requests_per_batch + (i < batches_with_extra_request ? 1 : 0);
        current_req += batch_sizes[i];
    }

    // Only one barrier-backed submission graph may run at a time. This keeps
    // concurrent transfers from occupying a subset of workers in different
    // barriers and deadlocking the executor.
    const std::lock_guard<std::mutex> dispatch_lock(submit_dispatch_lock_);
    std::barrier submission_start(static_cast<std::ptrdiff_t>(active_workers));
    tf::Taskflow submission_flow;

    try {
        for (size_t worker = 0; worker < active_workers; ++worker) {
            submission_flow.emplace([&, worker]() {
                // Blocking here forces each lane onto a distinct executor
                // worker, whose CPU affinity was set at executor startup.
                submission_start.arrive_and_wait();

                for (size_t batch_index = worker; batch_index < batch_count;
                     batch_index += active_workers) {
                    nixl_status_t status = NIXL_ERR_BACKEND;
                    try {
                        status = createAndSubmitBatch(request_list,
                                                      batch_starts[batch_index],
                                                      batch_sizes[batch_index],
                                                      gds_handle->batch_io_list[batch_index]);
                    }
                    catch (const std::exception &e) {
                        NIXL_ERROR << "GDS: batch submission worker failed: " << e.what();
                    }
                    catch (...) {
                        NIXL_ERROR
                            << "GDS: batch submission worker failed with an unknown exception";
                    }

                    if (status != NIXL_SUCCESS) {
                        nixl_status_t expected = NIXL_SUCCESS;
                        first_error.compare_exchange_strong(expected, status);
                    }
                }
            });
        }
        executor_->run(submission_flow).get();
    }
    catch (const std::exception &e) {
        NIXL_ERROR << "GDS: failed to run batch submission graph: " << e.what();
        first_error.store(NIXL_ERR_BACKEND);
    }

    const nixl_status_t submit_status = first_error.load();
    if (submit_status != NIXL_SUCCESS) {
        gds_handle->overall_status = submit_status;
        if (cancelAndReclaimBatches(gds_handle->batch_io_list) != NIXL_SUCCESS) {
            return NIXL_ERR_BACKEND;
        }
        return submit_status;
    }

    return NIXL_IN_PROG;
}

nixl_status_t
nixlGdsBatchEngine::checkXfer(nixlBackendReqH *handle) const {
    auto *gds_handle = static_cast<nixlGdsBatchReqH *>(handle);

    if (gds_handle->host_memory) {
        if (!gds_handle->host_transfer.valid()) {
            return gds_handle->host_status.load();
        }
        if (gds_handle->host_transfer.wait_for(std::chrono::seconds(0)) !=
            std::future_status::ready) {
            return NIXL_IN_PROG;
        }
        try {
            gds_handle->host_transfer.get();
        }
        catch (const std::exception &e) {
            NIXL_ERROR << "GDS: host-memory transfer failed: " << e.what();
            gds_handle->host_status.store(NIXL_ERR_BACKEND);
            return NIXL_ERR_BACKEND;
        }
        return gds_handle->host_status.load();
    }

    if (gds_handle->batch_io_list.empty()) {
        return gds_handle->overall_status;
    }

    auto current = gds_handle->batch_io_list.begin();
    while (current != gds_handle->batch_io_list.end()) {
        nixlGdsIOBatch *batch = *current;
        const nixl_status_t status = batch->checkStatus();

        if (status == NIXL_IN_PROG) {
            ++current;
            continue;
        }
        if (status == NIXL_SUCCESS) {
            returnBatchToPool(batch);
            current = gds_handle->batch_io_list.erase(current);
            continue;
        }
        if (gds_handle->overall_status == NIXL_SUCCESS) {
            gds_handle->overall_status = status;
        }
        ++current;
    }

    if (gds_handle->overall_status != NIXL_SUCCESS) {
        if (cancelAndReclaimBatches(gds_handle->batch_io_list) != NIXL_SUCCESS) {
            gds_handle->overall_status = NIXL_ERR_BACKEND;
        }
        return gds_handle->overall_status;
    }

    return gds_handle->batch_io_list.empty() ? NIXL_SUCCESS : NIXL_IN_PROG;
}

nixl_status_t
nixlGdsBatchEngine::releaseReqH(nixlBackendReqH *handle) const {
    auto *gds_handle = static_cast<nixlGdsBatchReqH *>(handle);
    if (cancelAndReclaimBatches(gds_handle->batch_io_list) != NIXL_SUCCESS) {
        return NIXL_ERR_BACKEND;
    }
    delete gds_handle;
    return NIXL_SUCCESS;
}
