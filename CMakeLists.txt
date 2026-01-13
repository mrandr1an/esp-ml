cmake_minimum_required(VERSION 3.16)
project(esp-ml C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# ---- Library ----
# Collect library sources.
file(GLOB_RECURSE ML_SOURCES CONFIGURE_DEPENDS
  "${CMAKE_SOURCE_DIR}/src/*.c"
)

add_library(ml STATIC ${ML_SOURCES})

# Public headers
target_include_directories(ml PUBLIC
  "${CMAKE_SOURCE_DIR}/include"
)

# math library for expf/logf on non windows
if (NOT MSVC)
  target_link_libraries(ml PUBLIC m)
endif()

# ---- Example helpers (desktop_examples/common) ----
file(GLOB_RECURSE EXAMPLE_COMMON_SOURCES CONFIGURE_DEPENDS
  "${CMAKE_SOURCE_DIR}/desktop-examples/common/*.c"
)

add_library(example_common STATIC ${EXAMPLE_COMMON_SOURCES})

target_include_directories(example_common PUBLIC
  "${CMAKE_SOURCE_DIR}/desktop-examples/common/include"
)

# Helpers also include esp-ml headers
target_include_directories(example_common PUBLIC
  "${CMAKE_SOURCE_DIR}/include"
)

# ---- Executable example ----
add_executable(iris
  "${CMAKE_SOURCE_DIR}/desktop-examples/iris.c"
)

target_link_libraries(iris PRIVATE ml example_common)

# Optional: make it easy to run with `cmake --build . --target run_basic`
add_custom_target(run_iris
  COMMAND iris
  DEPENDS iris
  WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
)
