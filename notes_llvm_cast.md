# llvm::cast<TensorOrMemDesc>(RankedTensorType type)

## What is TensorOrMemDesc?
include/triton/Dialect/TritonGPU/IR/TritonGPUTypeInterfaces.td
```MLIR
// Interface dynamically attached to RankedTensorType and MemDescType.
def TTG_TensorOrMemDesc : TypeInterface<"TensorOrMemDesc"> {
  let cppNamespace = "::mlir::triton::gpu";
  let methods = [
    InterfaceMethod<"Returns the encoding of the tensor or memory descriptor",
      "mlir::Attribute", "getEncoding", (ins)>,
    InterfaceMethod<"Returns element type",
      "mlir::Type", "getElementType", (ins)>,
    InterfaceMethod<"Returns the type shape",
      "llvm::ArrayRef<int64_t>", "getShape", (ins)>,
    InterfaceMethod<"Returns the tensor or buffer rank",
      "int64_t", "getRank", (ins)>,
    InterfaceMethod<"Returns the element type bit width",
      "int64_t", "getElementTypeBitWidth", (ins)>,
  ];
}
```
So `TensorOrMemDesc` is a `TypeInterface`.

## Define external Models for TensorOrMemDesc
lib/Dialect/TritonGPU/IR/Dialect.cpp
```C++
namespace {
struct TensorModel
    : public triton::gpu::TensorOrMemDesc::ExternalModel<TensorModel,
                                                         RankedTensorType> {
  Type getElementType(Type pointer) const {
    return cast<RankedTensorType>(pointer).getElementType();
  }
  Attribute getEncoding(Type pointer) const {
    return cast<RankedTensorType>(pointer).getEncoding();
  }
  ArrayRef<int64_t> getShape(Type pointer) const {
    return cast<RankedTensorType>(pointer).getShape();
  }
  int64_t getRank(Type pointer) const {
    return cast<RankedTensorType>(pointer).getRank();
  }
  int64_t getElementTypeBitWidth(Type pointer) const {
    return cast<RankedTensorType>(pointer).getElementTypeBitWidth();
  }
};

struct MemDescModel
    : public triton::gpu::TensorOrMemDesc::ExternalModel<MemDescModel,
                                                         MemDescType> {
  Type getElementType(Type pointer) const {
    return cast<MemDescType>(pointer).getElementType();
  }
  Attribute getEncoding(Type pointer) const {
    return cast<MemDescType>(pointer).getEncoding();
  }
  ArrayRef<int64_t> getShape(Type pointer) const {
    return cast<MemDescType>(pointer).getShape();
  }
  int64_t getRank(Type pointer) const {
    return cast<MemDescType>(pointer).getShape().size();
  }
  int64_t getElementTypeBitWidth(Type pointer) const {
    return cast<MemDescType>(pointer).getElementType().getIntOrFloatBitWidth();
  }
};
} // namespace
```

## Register external models TensorModel and MemDescModel
```C++
void TritonGPUDialect::initialize() {
  // ...
  RankedTensorType::attachInterface<TensorModel>(*getContext());
  MemDescType::attachInterface<MemDescModel>(*getContext());
}
```
Now that 
- a `RankedTensorType` is a `TensorOrMemDesc`
- a `MemDescType` is a `TensorOrMemDesc`

