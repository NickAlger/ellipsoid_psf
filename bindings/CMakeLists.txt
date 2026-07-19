# SPDX-License-Identifier: MIT
set(PYBIND11_FINDPYTHON ON)

# Inside a pip build, scikit-build-core provides pybind11 via find_package;
# otherwise fall back to a pinned download.
find_package(pybind11 2.12 CONFIG QUIET)
if(NOT pybind11_FOUND)
    include(FetchContent)
    FetchContent_Declare(pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG v2.13.6
        GIT_SHALLOW TRUE)
    FetchContent_MakeAvailable(pybind11)
endif()

pybind11_add_module(psfi_python psfi_bindings.cpp)
target_link_libraries(psfi_python PRIVATE psfi::psfi)
set_target_properties(psfi_python PROPERTIES OUTPUT_NAME psfi)

if(SKBUILD)
    install(TARGETS psfi_python LIBRARY DESTINATION .)
endif()
