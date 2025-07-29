get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(triton_libs GLOBAL PROPERTY TRITON_LIBS)


# triton_compiler
add_llvm_executable(triton_compiler triton_compiler.cpp PARTIAL_SOURCES_INTENDED)
target_compile_options(triton_compiler PRIVATE -fno-rtti -g -O0)
llvm_update_compile_flags(triton_compiler)
target_link_libraries(triton_compiler PRIVATE
  ${dialect_libs}
  ${conversion_libs}
  ${triton_libs}

  # MLIR core
  MLIROptLib
  MLIRPass
  MLIRTransforms
  MLIRLLVMToLLVMIRTranslation

  # LLVM
  LLVMTarget
  LLVMX86CodeGen
  LLVMX86AsmParser
  LLVMX86Disassembler
  LLVMX86TargetMCA
  LLVMX86Desc
  LLVMX86Info
)

mlir_check_all_link_libraries(triton_compiler)

set(CUDA_HOME "/usr/local/cuda-12.4")
include_directories(${CUDA_HOME}/include)

# Find the CUDA library
find_library(CUDA_LIBRARY
  NAMES cuda
  PATHS /usr/lib/x86_64-linux-gnu
  PATH_SUFFIXES lib
  NO_DEFAULT_PATH
)

message(STATUS "CUDA_LIBRARY: ${CUDA_LIBRARY}")

add_executable(runner runner.cpp)
target_compile_options(runner PRIVATE -fno-rtti -g -O0)

target_link_libraries(runner PRIVATE
  ${CUDA_LIBRARY}
)
