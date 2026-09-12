cmake_minimum_required(VERSION 3.16)
project(bpx_sar_rl LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux" OR
   NOT CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64|AMD64)$")
    message(FATAL_ERROR "Bundled dependencies require Linux x86_64")
endif()

set(MUJOCO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/library/mujoco")
set(SIMULATE_ROOT "${MUJOCO_ROOT}/simulate")

set(RUNTIME_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/library/inference_runtime")
set(ONNXRUNTIME_ROOT "${RUNTIME_ROOT}/onnxruntime")
set(LIBTORCH_ROOT "${RUNTIME_ROOT}/libtorch")

set(MUJOCO_LIBRARY "${MUJOCO_ROOT}/lib/libmujoco.so.3.2.7")
set(ONNXRUNTIME_LIBRARY
    "${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so.1.22.0"
)

set(SIMULATE_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/library/thirdparty/mujoco_simulate.cc"
    "${SIMULATE_ROOT}/glfw_adapter.cc"
    "${SIMULATE_ROOT}/glfw_dispatch.cc"
    "${SIMULATE_ROOT}/platform_ui_adapter.cc"
)

foreach(required_file IN ITEMS
    ${SIMULATE_SOURCES}
    "${SIMULATE_ROOT}/simulate.h"
    "${SIMULATE_ROOT}/glfw_adapter.h"
    "${SIMULATE_ROOT}/glfw_dispatch.h"
    "${SIMULATE_ROOT}/platform_ui_adapter.h"
    "${SIMULATE_ROOT}/array_safety.h"

    "${MUJOCO_ROOT}/include/mujoco/mujoco.h"
    "${MUJOCO_LIBRARY}"

    "${ONNXRUNTIME_ROOT}/include/onnxruntime_cxx_api.h"
    "${ONNXRUNTIME_LIBRARY}"
    "${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so.1"

    "${LIBTORCH_ROOT}/include/torch/script.h"
    "${LIBTORCH_ROOT}/lib/libtorch.so"
    "${LIBTORCH_ROOT}/lib/libtorch_cpu.so"
    "${LIBTORCH_ROOT}/lib/libc10.so"
    "${LIBTORCH_ROOT}/share/cmake/Torch/TorchConfig.cmake"
    "${LIBTORCH_ROOT}/share/cmake/Caffe2/Caffe2Config.cmake"
)
    if(NOT EXISTS "${required_file}")
        message(FATAL_ERROR "Missing dependency: ${required_file}")
    endif()
endforeach()

find_package(yaml-cpp REQUIRED)
find_package(glfw3 REQUIRED)
find_package(OpenGL REQUIRED)
find_package(Threads REQUIRED)

# 固定使用项目内的 LibTorch，避免外部环境或旧缓存混入另一版本。
set(ENV{TORCH_INSTALL_PREFIX} "${LIBTORCH_ROOT}")

set(Torch_DIR
    "${LIBTORCH_ROOT}/share/cmake/Torch"
    CACHE PATH "Bundled Torch" FORCE
)
set(Caffe2_DIR
    "${LIBTORCH_ROOT}/share/cmake/Caffe2"
    CACHE PATH "Bundled Caffe2" FORCE
)
set(TORCH_LIBRARY
    "${LIBTORCH_ROOT}/lib/libtorch.so"
    CACHE FILEPATH "Bundled torch" FORCE
)
set(c10_LIBRARY
    "${LIBTORCH_ROOT}/lib/libc10.so"
    CACHE FILEPATH "Bundled c10" FORCE
)

find_package(Torch REQUIRED CONFIG)

if(NOT TORCH_CXX_FLAGS MATCHES "_GLIBCXX_USE_CXX11_ABI=1")
    message(FATAL_ERROR
        "Use cxx11-ABI LibTorch with the system yaml-cpp"
    )
endif()

separate_arguments(
    TORCH_COMPILE_FLAGS
    NATIVE_COMMAND
    "${TORCH_CXX_FLAGS}"
)

add_library(mujoco SHARED IMPORTED)
set_target_properties(mujoco PROPERTIES
    IMPORTED_LOCATION "${MUJOCO_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${MUJOCO_ROOT}/include"
)

add_library(onnxruntime SHARED IMPORTED)
set_target_properties(onnxruntime PROPERTIES
    IMPORTED_LOCATION "${ONNXRUNTIME_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_ROOT}/include"
)

add_library(bpx_core STATIC
    src/inference_runtime.cpp
    src/rl.cpp
    src/fsm.cpp
)

target_include_directories(bpx_core
    PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include"
)
target_include_directories(bpx_core
    SYSTEM PRIVATE ${TORCH_INCLUDE_DIRS}
)
target_link_libraries(bpx_core
    PUBLIC
        yaml-cpp
        onnxruntime
        ${TORCH_LIBRARIES}
        Threads::Threads
)
target_compile_options(bpx_core
    PUBLIC ${TORCH_COMPILE_FLAGS}
)
target_compile_options(bpx_core
    PRIVATE -Wall -Wextra -Wpedantic
)

add_library(mujoco_simulate STATIC ${SIMULATE_SOURCES})

target_include_directories(mujoco_simulate
    SYSTEM PUBLIC "${SIMULATE_ROOT}"
)
target_link_libraries(mujoco_simulate
    PUBLIC
        mujoco
        glfw
        OpenGL::GL
        Threads::Threads
)
target_compile_options(mujoco_simulate
    PUBLIC ${TORCH_COMPILE_FLAGS}
)

add_executable(rl_sim_mujoco
    src/rl_sim_mujoco.cpp
)
target_compile_definitions(rl_sim_mujoco
    PRIVATE BPX_PROJECT_ROOT="${CMAKE_CURRENT_SOURCE_DIR}"
)
target_link_libraries(rl_sim_mujoco
    PRIVATE bpx_core mujoco_simulate
)
target_compile_options(rl_sim_mujoco
    PRIVATE -Wall -Wextra -Wpedantic
)

set(BPX_RUNTIME_PATHS "")

foreach(directory IN ITEMS
    "${MUJOCO_ROOT}/lib"
    "${ONNXRUNTIME_ROOT}/lib"
    "${LIBTORCH_ROOT}/lib"
)
    file(
        RELATIVE_PATH relative_directory
        "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}"
        "${directory}"
    )
    list(APPEND BPX_RUNTIME_PATHS
        "\$ORIGIN/${relative_directory}"
    )
endforeach()

target_link_options(rl_sim_mujoco
    PRIVATE "LINKER:--disable-new-dtags"
)
set_target_properties(rl_sim_mujoco PROPERTIES
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH "${BPX_RUNTIME_PATHS}"
)

message(STATUS "LibTorch version: ${Torch_VERSION}")
message(STATUS "LibTorch root: ${LIBTORCH_ROOT}")
message(STATUS "Simulate sources: ${SIMULATE_ROOT}")