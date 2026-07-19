#include "engram/thread_pool.h"

#include <algorithm>
#include <cerrno>
#include <stdexcept>
#include <string>
#include <system_error>

#if defined(__linux__)
#include <pthread.h>
#include <sched.h>
#endif

namespace engram {
namespace {

thread_local ThreadPool* current_pool = nullptr;

}  // namespace

ThreadPool::ThreadPool(const std::size_t thread_count,
                       std::vector<unsigned> affinity) {
  if (thread_count == 0) {
    throw std::invalid_argument("thread_count must be positive");
  }
  if (!affinity.empty() && affinity.size() != thread_count) {
    throw std::invalid_argument(
        "affinity list size must equal the configured thread count");
  }
  validate_affinity(affinity);
  workers_.reserve(thread_count);
  try {
    for (std::size_t index = 0; index < thread_count; ++index) {
      workers_.emplace_back([this] { worker_loop(); });
    }
    apply_affinity(affinity);
  } catch (...) {
    shutdown();
    throw;
  }
}

ThreadPool::~ThreadPool() { shutdown(); }

std::size_t ThreadPool::thread_count() const noexcept {
  return workers_.size();
}

bool ThreadPool::stopped() const noexcept {
  std::lock_guard lock(job_mutex_);
  return stopping_;
}

void ThreadPool::shutdown() noexcept {
  std::lock_guard submission_lock(submission_mutex_);
  {
    std::lock_guard job_lock(job_mutex_);
    if (stopping_) {
      return;
    }
    stopping_ = true;
    cancelled_.store(true, std::memory_order_relaxed);
  }
  work_ready_.notify_all();
  for (std::thread& worker : workers_) {
    if (worker.joinable()) {
      worker.join();
    }
  }
}

void ThreadPool::parallel_for_raw(const std::size_t begin,
                                  const std::size_t end,
                                  const std::size_t grain_size,
                                  const Callback callback, void* context) {
  if (grain_size == 0) {
    throw std::invalid_argument("grain_size must be positive");
  }
  if (end < begin) {
    throw std::invalid_argument("parallel_for end must not precede begin");
  }
  if (callback == nullptr) {
    throw std::invalid_argument("parallel_for callback must not be null");
  }
  if (current_pool == this) {
    throw std::logic_error("recursive parallel_for on the same pool is not supported");
  }
  std::unique_lock submission_lock(submission_mutex_);
  if (end == begin) {
    std::lock_guard job_lock(job_mutex_);
    if (stopping_) {
      throw std::runtime_error("parallel_for called after thread-pool shutdown");
    }
    return;
  }
  std::unique_lock job_lock(job_mutex_);
  if (stopping_) {
    throw std::runtime_error("parallel_for called after thread-pool shutdown");
  }
  job_end_ = end;
  job_grain_ = grain_size;
  next_index_.store(begin, std::memory_order_relaxed);
  cancelled_.store(false, std::memory_order_relaxed);
  callback_ = callback;
  callback_context_ = context;
  job_exception_ = nullptr;
  completed_workers_ = 0;
  job_active_ = true;
  ++generation_;
  work_ready_.notify_all();
  work_done_.wait(job_lock, [this] { return !job_active_; });
  const std::exception_ptr exception = job_exception_;
  callback_ = nullptr;
  callback_context_ = nullptr;
  job_lock.unlock();
  if (exception) {
    std::rethrow_exception(exception);
  }
}

void ThreadPool::worker_loop() {
  current_pool = this;
  std::uint64_t observed_generation = 0;
  std::unique_lock lock(job_mutex_);
  while (true) {
    work_ready_.wait(lock, [this, observed_generation] {
      return stopping_ ||
             (job_active_ && generation_ != observed_generation);
    });
    if (stopping_) {
      break;
    }
    observed_generation = generation_;
    lock.unlock();
    execute_chunks();
    lock.lock();
    ++completed_workers_;
    if (completed_workers_ == workers_.size()) {
      job_active_ = false;
      work_done_.notify_one();
    }
  }
  current_pool = nullptr;
}

void ThreadPool::execute_chunks() {
  while (!cancelled_.load(std::memory_order_relaxed)) {
    std::size_t chunk_begin = next_index_.load(std::memory_order_relaxed);
    std::size_t chunk_end{};
    while (true) {
      if (chunk_begin >= job_end_) {
        return;
      }
      chunk_end = chunk_begin + std::min(job_grain_, job_end_ - chunk_begin);
      if (next_index_.compare_exchange_weak(chunk_begin, chunk_end,
                                            std::memory_order_relaxed)) {
        break;
      }
    }
    try {
      for (std::size_t index = chunk_begin; index < chunk_end; ++index) {
        callback_(callback_context_, index);
      }
    } catch (...) {
      bool expected = false;
      if (cancelled_.compare_exchange_strong(expected, true,
                                             std::memory_order_relaxed)) {
        std::lock_guard lock(job_mutex_);
        job_exception_ = std::current_exception();
      }
      return;
    }
  }
}

void ThreadPool::validate_affinity(
    const std::vector<unsigned>& affinity) const {
  if (affinity.empty()) {
    return;
  }
#if defined(__linux__)
  cpu_set_t allowed;
  CPU_ZERO(&allowed);
  if (sched_getaffinity(0, sizeof(allowed), &allowed) != 0) {
    throw std::system_error(errno, std::generic_category(),
                            "failed to query allowed CPU affinity");
  }
  for (std::size_t index = 0; index < affinity.size(); ++index) {
    const unsigned cpu = affinity[index];
    if (cpu >= CPU_SETSIZE || !CPU_ISSET(cpu, &allowed)) {
      throw std::invalid_argument("affinity CPU is not available to this process");
    }
    if (std::find(affinity.begin(), affinity.begin() + index, cpu) !=
        affinity.begin() + index) {
      throw std::invalid_argument("affinity CPUs must be unique");
    }
  }
#else
  throw std::invalid_argument(
      "thread affinity is supported only on Linux in this runtime");
#endif
}

void ThreadPool::apply_affinity(const std::vector<unsigned>& affinity) {
  if (affinity.empty()) {
    return;
  }
#if defined(__linux__)
  for (std::size_t index = 0; index < workers_.size(); ++index) {
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(affinity[index], &set);
    const int error = pthread_setaffinity_np(workers_[index].native_handle(),
                                             sizeof(set), &set);
    if (error != 0) {
      throw std::system_error(error, std::generic_category(),
                              "failed to set worker CPU affinity");
    }
  }
#endif
}

}  // namespace engram
