# SoftVector micro-benchmarks.
#
# Build a standalone benchmark driver that links against the SoftVector shared
# library and times a representative set of RVV instructions through the public
# C API. Enabled via -DBUILD_BENCHMARKS=ON at the top level.

set(BENCHMARKS_TARGET svbench)

add_executable(${BENCHMARKS_TARGET}
    bench.cpp
)

target_compile_features(${BENCHMARKS_TARGET} PRIVATE cxx_std_20)

target_include_directories(${BENCHMARKS_TARGET} PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/include/target/riscv
    ${PROJECT_SOURCE_DIR}/lib/softfloat-spike/include
)

target_compile_options(${BENCHMARKS_TARGET} PRIVATE
    $<$<CONFIG:Release>:-O3>
)

target_link_libraries(${BENCHMARKS_TARGET} PRIVATE ${PROJECT_NAME})

set_target_properties(${BENCHMARKS_TARGET} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)
