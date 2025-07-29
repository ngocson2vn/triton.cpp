cmake_minimum_required(VERSION 3.20)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

project(triton_compiler CXX C)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D__STDC_FORMAT_MACROS")

# #########
# LLVM
# #########
cmake_path(GET PROJECT_SOURCE_DIR PARENT_PATH LEARNMLIR_DIR)
message(STATUS "LEARNMLIR_DIR: ${LEARNMLIR_DIR}")

set(LLVM_BINARY_DIR "${LEARNMLIR_DIR}/llvm-project/build")
add_subdirectory(${LEARNMLIR_DIR}/llvm-project/llvm ${LLVM_BINARY_DIR})

# LLVM and MLIR headers
include_directories(${LEARNMLIR_DIR}/llvm-project/llvm/include)
include_directories(${LEARNMLIR_DIR}/llvm-project/mlir/include)

# Some headers are generated automatically, which is why we need to include the following directories
include_directories(${LLVM_BINARY_DIR}/include)
include_directories(${LLVM_BINARY_DIR}/tools/mlir/include)

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

# Utilities
function(add_triton_object name)
  cmake_parse_arguments(ARG "" "" "DEPENDS;LINK_LIBS" ${ARGN})
  add_library(${name} OBJECT)
  target_sources(${name}
    PRIVATE ${ARG_UNPARSED_ARGUMENTS}
    INTERFACE $<TARGET_OBJECTS:${name}>
  )


  # add_library(${name} OBJECT ${ARG_UNPARSED_ARGUMENTS})
  if(ARG_DEPENDS)
    add_dependencies(${name} ${ARG_DEPENDS})
  endif()
  if(ARG_LINK_LIBS)
    target_link_libraries(${name} PUBLIC ${ARG_LINK_LIBS})
  endif()
endfunction(add_triton_object)

set_property(GLOBAL PROPERTY TRITON_LIBS "")
function(add_triton_library name)
  set_property(GLOBAL APPEND PROPERTY TRITON_LIBS ${name})
  add_triton_object(${name} ${ARGN})
  llvm_update_compile_flags(${name})
endfunction()

set_property(GLOBAL PROPERTY TRITON_PLUGINS "")
function(add_triton_plugin name)
  set_property(GLOBAL APPEND PROPERTY TRITON_PLUGINS ${name})
  add_triton_object(${name} ${ARGN})
endfunction()


# Disable warnings that show up in external code (gtest;pybind11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror -Wno-covered-switch-default -fvisibility=hidden -fno-rtti -g -O0")

include_directories(".")
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_BINARY_DIR}/include) # Tablegen'd files
include_directories(${PROJECT_SOURCE_DIR}/third_party)
include_directories(${PROJECT_BINARY_DIR}/third_party) # Tablegen'd files

# link_directories(${LLVM_LIBRARY_DIR})
add_subdirectory(include)
add_subdirectory(lib)

# TODO: Figure out which target is sufficient to fix errors; triton is
# apparently not enough. Currently set linking libstdc++fs for all targets
# to support some old version GCC compilers like 8.3.0.
if (NOT WIN32 AND NOT APPLE AND NOT BSD)
  link_libraries(stdc++fs)
endif()

if (UNIX AND NOT APPLE)
  set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--exclude-libs,ALL")
endif()

if(NOT TRITON_BUILD_PYTHON_MODULE)
  foreach(CODEGEN_BACKEND ${TRITON_CODEGEN_BACKENDS})
    add_subdirectory(third_party/${CODEGEN_BACKEND})
  endforeach()
  add_subdirectory(third_party/proton/dialect)
endif()

find_package(Threads REQUIRED)

add_subdirectory(third_party/f2reduce)
add_subdirectory(bin)
