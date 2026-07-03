get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(triton_libs GLOBAL PROPERTY TRITON_LIBS)

# Find the CUDA libraries
find_library(CUDA_DRIVER
  NAMES cuda
  PATHS /usr/lib/x86_64-linux-gnu
  PATH_SUFFIXES lib
  NO_DEFAULT_PATH
)
message(STATUS "CUDA_DRIVER: ${CUDA_DRIVER}")

find_library(CUDA_RUNTIME
  NAMES cudart
  PATHS ${CUDA_HOME}/lib64
  PATH_SUFFIXES lib
  NO_DEFAULT_PATH
)
message(STATUS "CUDA_RUNTIME: ${CUDA_RUNTIME}")

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
  MLIRRegisterAllPasses
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

  # CUDA
  ${CUDA_RUNTIME}
  ${CUDA_DRIVER}
)

mlir_check_all_link_libraries(triton_compiler)

# Compile triton_debug.cpp
add_llvm_executable(triton_debug triton_debug.cpp PARTIAL_SOURCES_INTENDED)
target_compile_options(triton_debug PRIVATE -fno-rtti -g -O0)
llvm_update_compile_flags(triton_debug)
target_link_libraries(triton_debug PRIVATE
  ${dialect_libs}
  ${conversion_libs}
  ${triton_libs}

  # MLIR core
  MLIROptLib
  MLIRPass
  MLIRRegisterAllPasses
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

  # CUDA
  ${CUDA_RUNTIME}
  ${CUDA_DRIVER}
)

mlir_check_all_link_libraries(triton_debug)

# triton_test
add_llvm_executable(triton_test triton_test.cpp PARTIAL_SOURCES_INTENDED)
target_compile_options(triton_test PRIVATE -fno-rtti -g -O0)
llvm_update_compile_flags(triton_test)
target_link_libraries(triton_test PRIVATE
  ${dialect_libs}
  ${conversion_libs}
  ${triton_libs}

  # MLIR core
  MLIROptLib
  MLIRPass
  MLIRRegisterAllPasses
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

  # CUDA
  ${CUDA_RUNTIME}
  ${CUDA_DRIVER}
)

mlir_check_all_link_libraries(triton_test)

# #======================================================
# # test_linear_layout
# #======================================================
# add_llvm_executable(test_linear_layout test_linear_layout.cpp PARTIAL_SOURCES_INTENDED)
# target_compile_options(test_linear_layout PRIVATE -fno-rtti -g -O0)
# llvm_update_compile_flags(test_linear_layout)
# target_link_libraries(test_linear_layout PRIVATE
#   TritonTools
#   # MLIR core
#   MLIRIR
# )

# mlir_check_all_link_libraries(test_linear_layout)

# #======================================================
# # test_ll_strided1D
# #======================================================
# add_llvm_executable(test_ll_strided1D test_ll_strided1D.cpp PARTIAL_SOURCES_INTENDED)
# target_compile_options(test_ll_strided1D PRIVATE -fno-rtti -g -O0)
# llvm_update_compile_flags(test_ll_strided1D)
# target_link_libraries(test_ll_strided1D PRIVATE
#   TritonTools
#   # MLIR core
#   MLIRIR
# )
# mlir_check_all_link_libraries(test_ll_strided1D)

# #======================================================
# # run_add_kernel
# #======================================================
# add_executable(run_add_kernel run_add_kernel.cpp)
# target_compile_options(run_add_kernel PRIVATE -fno-rtti -g -O0)
# target_link_libraries(run_add_kernel PRIVATE
#   runtime
#   ${CUDA_DRIVER}
# )

# add_executable(run_triton_dot_kernel run_triton_dot_kernel.cpp)
# target_compile_options(run_triton_dot_kernel PRIVATE -fno-rtti -g -O0)
# target_link_libraries(run_triton_dot_kernel PRIVATE
#   runtime
#   ${CUDA_DRIVER}
# )
