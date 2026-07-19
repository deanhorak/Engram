#include "engram/thread_pool.h"

#include <atomic>
#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct MoveOnlyTask {
  explicit MoveOnlyTask(std::atomic<std::size_t>& target) : target(&target) {}
  MoveOnlyTask(const MoveOnlyTask&) = delete;
  MoveOnlyTask& operator=(const MoveOnlyTask&) = delete;
  void operator()(const std::size_t) {
    target->fetch_add(1, std::memory_order_relaxed);
  }
  std::atomic<std::size_t>* target;
};

int fail(const char* message) {
  std::cerr << message << '\n';
  return 1;
}

}  // namespace

int main() {
  bool rejected_zero = false;
  try {
    const engram::ThreadPool invalid(0);
  } catch (const std::invalid_argument&) {
    rejected_zero = true;
  }
  if (!rejected_zero) {
    return fail("zero-thread pool was accepted");
  }

  engram::ThreadPool pool(4);
  if (pool.thread_count() != 4 || pool.stopped()) {
    return fail("thread-pool construction metrics mismatch");
  }
  constexpr std::size_t count = 4099;
  std::vector<std::size_t> squares(count);
  std::vector<std::atomic<unsigned>> visits(count);
  for (auto& visit : visits) {
    visit.store(0, std::memory_order_relaxed);
  }
  pool.parallel_for(0, count, 17, [&](const std::size_t index) {
    squares[index] = index * index;
    visits[index].fetch_add(1, std::memory_order_relaxed);
  });
  for (std::size_t index = 0; index < count; ++index) {
    if (squares[index] != index * index ||
        visits[index].load(std::memory_order_relaxed) != 1) {
      return fail("parallel_for skipped or repeated an index");
    }
  }

  // Reuse the persistent workers and descriptor many times with varied grains.
  std::atomic<std::size_t> total{0};
  for (std::size_t iteration = 0; iteration < 40; ++iteration) {
    pool.parallel_for(3, 1003, iteration % 23 + 1,
                      [&](const std::size_t) {
                        total.fetch_add(1, std::memory_order_relaxed);
                      });
  }
  if (total.load(std::memory_order_relaxed) != 40 * 1000) {
    return fail("reused parallel_for produced a nondeterministic count");
  }
  std::atomic<std::size_t> move_only_total{0};
  MoveOnlyTask move_only_task(move_only_total);
  pool.parallel_for(0, 257, 11, move_only_task);
  if (move_only_total.load(std::memory_order_relaxed) != 257) {
    return fail("move-only callable execution mismatch");
  }

  bool propagated = false;
  try {
    pool.parallel_for(0, 1000, 1, [](const std::size_t index) {
      if (index == 137) {
        throw std::runtime_error("deterministic worker failure");
      }
    });
  } catch (const std::runtime_error& error) {
    propagated = std::string(error.what()) == "deterministic worker failure";
  }
  if (!propagated) {
    return fail("worker exception was not propagated to the submitter");
  }
  std::atomic<std::size_t> recovery{0};
  pool.parallel_for(0, 128, 8, [&](const std::size_t) {
    recovery.fetch_add(1, std::memory_order_relaxed);
  });
  if (recovery.load(std::memory_order_relaxed) != 128) {
    return fail("pool was not reusable after a worker exception");
  }

  bool rejected_grain = false;
  try {
    pool.parallel_for(0, 1, 0, [](const std::size_t) {});
  } catch (const std::invalid_argument&) {
    rejected_grain = true;
  }
  if (!rejected_grain) {
    return fail("zero grain size was accepted");
  }

  bool rejected_affinity = false;
  try {
    const engram::ThreadPool invalid_affinity(2, {0});
  } catch (const std::invalid_argument&) {
    rejected_affinity = true;
  } catch (const std::system_error&) {
    // A restricted environment may reject CPU 0 before duplicate validation.
    rejected_affinity = true;
  }
  if (!rejected_affinity) {
    return fail("invalid affinity list was accepted");
  }

  pool.shutdown();
  pool.shutdown();
  if (!pool.stopped()) {
    return fail("thread-pool shutdown state mismatch");
  }
  bool rejected_after_shutdown = false;
  try {
    pool.parallel_for(0, 1, 1, [](const std::size_t) {});
  } catch (const std::runtime_error&) {
    rejected_after_shutdown = true;
  }
  if (!rejected_after_shutdown) {
    return fail("job submission after shutdown was accepted");
  }

  engram::ThreadPool single(1);
  std::size_t deterministic = 0;
  single.parallel_for(0, 100, 9,
                      [&](const std::size_t index) { deterministic += index; });
  if (deterministic != 4950) {
    return fail("single-worker execution mismatch");
  }
  return 0;
}
