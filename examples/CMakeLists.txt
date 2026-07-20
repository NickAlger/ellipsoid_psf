# SPDX-License-Identifier: MIT
set(PSFI_EXAMPLE_NAMES
    frog_compression
    frog_kernel)

foreach(example_name ${PSFI_EXAMPLE_NAMES})
    add_executable(example_${example_name} ${example_name}.cpp)
    target_link_libraries(example_${example_name} PRIVATE psfi::psfi)
    set_target_properties(example_${example_name} PROPERTIES OUTPUT_NAME ${example_name})
    target_compile_options(example_${example_name} PRIVATE
        $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall;-Wextra>)
endforeach()
