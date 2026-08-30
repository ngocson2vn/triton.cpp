# TODO List
## 1. Understand AsyncTMACopyGlobalToLocalOpConversion
Notes: [AsyncTMACopyGlobalToLocalOpConversion.md](./LinearLayoutCode/AsyncTMACopyGlobalToLocalOpConversion.md)

- [x] Understand `Value matrixVectorProd(TritonLLVMOpBuilder &b, const LinearLayout &A, Value x)` in `lib/Conversion/TritonGPUToLLVM/Utility.cpp`

- [DOING] Understand `LinearLayout LinearLayout::invertAndCompose(const LinearLayout &outer) const` in `lib/Tools/LinearLayout.cpp`
  - Build bin/triton_debug.cpp
  - Run: `./debug.sh`
