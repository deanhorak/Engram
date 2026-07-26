cmake_minimum_required(VERSION 3.20)
project(engram VERSION 0.1.0 LANGUAGES CXX)
find_package(Threads REQUIRED)

option(ENGRAM_BUILD_TESTS "Build native tests" ON)

add_library(engram_kernels STATIC
  native/src/controller.cpp
  native/src/cpu_features.cpp
  native/src/dip.cpp
  native/src/episodic.cpp
  native/src/npy.cpp
  native/src/native_bitnet.cpp
  native/src/native_shell_c.cpp
  native/src/native_stage_c.cpp
  native/src/operator_residual_c.cpp
  native/src/package.cpp
  native/src/runtime.cpp
  native/src/semantic.cpp
  native/src/semantic_ivf.cpp
  native/src/semantic_quantized.cpp
  native/src/swiglu.cpp
  native/src/streaming_attention.cpp
  native/src/ternary_projection.cpp
  native/src/thread_pool.cpp
  native/src/transition_cache.cpp
  native/src/vocabulary.cpp
  native/src/vocabulary_ivf.cpp
  native/src/vector_kernels.cpp
)
target_include_directories(engram_kernels PUBLIC native/include)
target_link_libraries(engram_kernels PUBLIC Threads::Threads)
target_compile_features(engram_kernels PUBLIC cxx_std_20)
target_compile_options(engram_kernels PRIVATE -Wall -Wextra -Wpedantic)
set_target_properties(engram_kernels PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_library(engram_bitnet SHARED
  native/src/native_attention_stage_c.cpp
  native/src/native_bitnet_c.cpp
  native/src/native_shell_c.cpp
  native/src/native_stage_c.cpp
  native/src/native_stage_runner_c.cpp
  native/src/operator_residual_c.cpp
  native/src/streaming_attention_c.cpp
  native/src/ternary_projection_c.cpp
)
target_link_libraries(engram_bitnet PRIVATE engram_kernels)
target_compile_features(engram_bitnet PRIVATE cxx_std_20)
target_compile_options(engram_bitnet PRIVATE -Wall -Wextra -Wpedantic)
set_target_properties(engram_bitnet PROPERTIES OUTPUT_NAME engram_bitnet)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  # The Python evaluation environment may already have loaded an older Conda
  # libstdc++. Keep the C ABI module self-contained so ctypes does not inherit
  # that process-global GLIBCXX version constraint.
  target_link_options(engram_bitnet PRIVATE -static-libstdc++ -static-libgcc)
endif()

add_library(engram_attention SHARED native/src/streaming_attention_c.cpp)
target_link_libraries(engram_attention PRIVATE engram_kernels)
target_compile_features(engram_attention PRIVATE cxx_std_20)
target_compile_options(engram_attention PRIVATE -Wall -Wextra -Wpedantic)
set_target_properties(engram_attention PROPERTIES OUTPUT_NAME engram_attention)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_link_options(engram_attention PRIVATE -static-libstdc++ -static-libgcc)
endif()

add_executable(engram-native-info native/src/info.cpp)
target_link_libraries(engram-native-info PRIVATE engram_kernels)
target_compile_features(engram-native-info PRIVATE cxx_std_20)

add_executable(engram-run native/src/run.cpp)
target_link_libraries(engram-run PRIVATE engram_kernels)
target_compile_features(engram-run PRIVATE cxx_std_20)

add_executable(engram-inspect native/src/inspect.cpp)
target_link_libraries(engram-inspect PRIVATE engram_kernels)
target_compile_features(engram-inspect PRIVATE cxx_std_20)

add_executable(engram-bench native/src/bench.cpp)
target_link_libraries(engram-bench PRIVATE engram_kernels)
target_compile_features(engram-bench PRIVATE cxx_std_20)

add_executable(engram-dip-bench native/src/dip_bench.cpp)
target_link_libraries(engram-dip-bench PRIVATE engram_kernels)
target_compile_features(engram-dip-bench PRIVATE cxx_std_20)

if(ENGRAM_BUILD_TESTS)
  enable_testing()
  add_executable(engram_native_tests native/tests/test_swiglu.cpp)
  target_link_libraries(engram_native_tests PRIVATE engram_kernels)
  add_test(NAME engram_native_tests COMMAND engram_native_tests)

  foreach(test_name IN ITEMS controller dip episodic native_stage npy package semantic semantic_ivf semantic_quantized streaming_attention thread_pool transition_cache vector_kernels vocabulary vocabulary_ivf)
    add_executable(engram_native_${test_name}_tests native/tests/test_${test_name}.cpp)
    target_link_libraries(engram_native_${test_name}_tests PRIVATE engram_kernels)
    add_test(NAME engram_native_${test_name}_tests COMMAND engram_native_${test_name}_tests)
  endforeach()
endif()
