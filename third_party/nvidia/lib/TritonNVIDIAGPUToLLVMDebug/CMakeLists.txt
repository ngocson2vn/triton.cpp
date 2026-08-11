add_triton_library(TritonNVIDIAGPUToLLVMDebug
    TritonGPUToLLVMDebug.cpp
    MemoryOpToLLVM.cpp
    LoadStoreOpToLLVM.cpp
    BarrierOpToLLVM.cpp
    PTXAsmFormat.cpp
    Utility.cpp
    TargetInfo.cpp
    Allocation.cpp

    DEPENDS
    TritonNVIDIAGPUConversionPassIncGenDebug
    NVGPUAttrDefsIncGen

    LINK_LIBS PUBLIC
    TritonAnalysis
    TritonGPUToLLVM
    TritonInstrumentToLLVM
    MLIRReconcileUnrealizedCasts
    NVGPUIR
    MLIRUBToLLVM
)
