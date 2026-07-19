#pragma once

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <type_traits>
#include <utility>
#include <vector>
#include <thread>

namespace engram {

class ThreadPool {
 public:
  explicit ThreadPool(std::size_t thread_count,
                      std::vector<unsigned> affinity = {});
  ~ThreadPool();

  ThreadPool(const ThreadPool&) = delete;
  ThreadPool& operator=(const ThreadPool&) = delete;
  ThreadPool(ThreadPool&&) = delete;
  ThreadPool& operator=(ThreadPool&&) = delete;

  [[nodiscard]] std::size_t thread_count() const noexcept;
  [[nodiscard]] bool stopped() const noexcept;
  void shutdown() noexcept;

  // The callable is stored on the submitting thread's stack for the duration
  // of the rendezvous. The pool reuses one internal job descriptor and does not
  // allocate a task object per invocation or per chunk.
  template <class Function>
  void parallel_for(std::size_t begin, std::size_t end,
                    std::size_t grain_size, Function&& function) {
    using Callable = std::remove_reference_t<Function>;
    static_assert(std::is_invocable_v<Callable&, std::size_t>,
                  "parallel_for callable must accept a size_t index");
    Callable* callable = std::addressof(function);
    parallel_for_raw(
        begin, end, grain_size,
        [](void* context, const std::size_t index) {
          std::invoke(*static_cast<Callable*>(context), index);
        },
        const_cast<void*>(static_cast<const void*>(callable)));
  }

 private:
  using Callback = void (*)(void*, std::size_t);

  void parallel_for_raw(std::size_t begin, std::size_t end,
                        std::size_t grain_size, Callback callback,
                        void* context);
  void worker_loop();
  void execute_chunks();
  void validate_affinity(const std::vector<unsigned>& affinity) const;
  void apply_affinity(const std::vector<unsigned>& affinity);

  std::vector<std::thread> workers_;
  mutable std::mutex job_mutex_;
  std::mutex submission_mutex_;
  std::condition_variable work_ready_;
  std::condition_variable work_done_;
  bool stopping_{};
  bool job_active_{};
  std::uint64_t generation_{};
  std::size_t completed_workers_{};
  std::size_t job_end_{};
  std::size_t job_grain_{};
  std::atomic<std::size_t> next_index_{};
  std::atomic<bool> cancelled_{};
  Callback callback_{};
  void* callback_context_{};
  std::exception_ptr job_exception_;
};

}  // namespace engram
