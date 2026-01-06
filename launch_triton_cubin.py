import os
import importlib
from types import ModuleType
import torch

# The TMA dtype enum values are slightly different on host vs device...
TMA_DTYPE_DEVICE_TO_HOST = dict((i, i) for i in range(16))
TMA_DTYPE_DEVICE_TO_HOST[8] = 10
TMA_DTYPE_DEVICE_TO_HOST[9] = 8
TMA_DTYPE_DEVICE_TO_HOST[10] = 9

BLOCK_SHAPE = [64, 64]

def find_paths():
    triton_cache_dir = "./tmp"
    cuda_utils_path = None
    launcher_path = None
    cubin_path = None
    for item in os.listdir(triton_cache_dir):
        dirname = f"{triton_cache_dir}/{item}"
        if os.path.isdir(dirname):
            for file_name in os.listdir(dirname):
                if "cuda_utils" in file_name:
                    cuda_utils_path = f"{dirname}/{file_name}"
                if "__triton_launcher" in file_name:
                    launcher_path = f"{dirname}/{file_name}"
                elif "triton_dot.cubin" in file_name:
                    cubin_path = f"{dirname}/{file_name}"
    return cuda_utils_path, launcher_path, cubin_path

cuda_utils_path, launcher_path, cubin_path = find_paths()
if cuda_utils_path and launcher_path and cubin_path:
    print(f"Found required paths:\n  - cuda_utils_path: {cuda_utils_path}\n  - launcher_path: {launcher_path}\n  - cubin_path: {cubin_path}")
else:
    raise RuntimeError(f"One of paths is None:\n  - cuda_utils_path: {cuda_utils_path}\n  - launcher_path: {launcher_path}\n  - cubin_path: {cubin_path}")

def get_input_matrices():
    input_file_path = "./triton_dot_input_matrices.pt"
    if os.path.exists(input_file_path):
        matrices = torch.load(input_file_path)
        print(f"Loaded input matrices from {input_file_path}")
        return matrices["a"], matrices["b"]

    a = (torch.rand(BLOCK_SHAPE, dtype=torch.float16).cuda() - 0.5)
    b = (torch.rand([BLOCK_SHAPE[0] // 2, BLOCK_SHAPE[1]], dtype=torch.float16).cuda() - 0.5)
    matrices = {}
    matrices["a"] = a
    matrices["b"] = b
    torch.save(matrices, input_file_path)
    print(f"Saved input matrices into {input_file_path}")
    return a, b

def load_module_from_path(name: str, path: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if not spec or not spec.loader:
        raise RuntimeError(f"Failed to load newly compiled {name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

class CudaUtils:
    def __init__(self, mod):
        import triton.backends.nvidia.driver
        triton.backends.nvidia.driver.PyCUtensorMap = mod.PyCUtensorMap
        self.mod = mod

    def make_tensor_map(self, input_tensor: torch.tensor, metadata):
        print(f"make_tensor_map metadata: {metadata}")
        swizzle = metadata["swizzle"]
        elem_size = metadata["elem_size"]
        elem_type = metadata["elem_type"]
        block_size = metadata["block_size"]
        padding = 0
        shape = tuple(input_tensor.shape)
        strides = tuple(input_tensor.stride())
        cu_tensor_map = self.mod.fill_tma_descriptor(
            input_tensor.data_ptr(),
            swizzle,
            elem_size,
            TMA_DTYPE_DEVICE_TO_HOST[elem_type],
            block_size,
            shape,
            strides,
            padding,
        )
        return cu_tensor_map

    def get_kernel_fn(self, name, kernel, shared, device=0):
        module, function, n_regs, n_spills, n_max_threads = self.mod.load_binary(name, kernel, shared, device)
        return function

cuda_utils_mod = load_module_from_path("cuda_utils", cuda_utils_path)
cuda_utils = CudaUtils(cuda_utils_mod)

launcher_mod = load_module_from_path("__triton_launcher", launcher_path)
print(f"get_tensor_map_type: {launcher_mod.get_tensor_map_type()}")

a, b = get_input_matrices()
print(f"a: shape={a.shape} data={a}")
print(f"b: shape={b.shape} data={b}")
c = torch.empty([a.shape[0], b.shape[0]], dtype=torch.float32).cuda()

# tensordesc_meta: [{'swizzle': 3, 'elem_size': 2, 'elem_type': 6, 'block_size': [64, 64], 'fp4_padded': False}, {'swizzle': 3, 'elem_size': 2, 'elem_type': 6, 'block_size': [32, 64], 'fp4_padded': False}, {'swizzle': 3, 'elem_size': 4, 'elem_type': 7, 'block_size': [64, 32], 'fp4_padded': False}]
a_metadata = {
    "swizzle": 3,
    "elem_size": 2,
    "elem_type": 6,
    "block_size": list(a.shape) # MxK
}

b_metadata = {
    "swizzle": 3,
    "elem_size": 2,
    "elem_type": 6,
    "block_size": list(b.shape) # NxK
}

c_metadata = {
    "swizzle": 3,
    "elem_size": 4,
    "elem_type": 7,
    "block_size": [a.shape[0], b.shape[0]] # MxN
}

a_tensor_map = cuda_utils.make_tensor_map(a, a_metadata)
b_tensor_map = cuda_utils.make_tensor_map(b, b_metadata)
c_tensor_map = cuda_utils.make_tensor_map(c, c_metadata)

# Load cubin from file (or pass bytes directly)
with open(cubin_path, "rb") as f:
    cubin_bytes = f.read()

shared_bytes = 12296

# The entrypoint name is the Triton-generated kernel symbol.
# Often it matches the Python function name: "kernel_fn".
kernel_fn = cuda_utils.get_kernel_fn("triton_dot", cubin_bytes, shared_bytes)

# Kernel launch configuration
M = a.shape[0]
N = b.shape[0]
K = b.shape[1]

# Triton kernels expect raw device pointers and scalars
launcher_mod.launch(
    1, 1, 1,                    # gridX, gridY, gridZ
    0,                          # stream
    kernel_fn,                  # kernel_function
    False, False, None, None,   # launch_cooperative_grid, launch_pdl, global_scratch_obj, profile_scratch_obj
    (4, 1, 12296, 1, 1, 1),     # kernel_metadata: num_warps, num_ctas, shared_memory, clusterDimX, clusterDimY, clusterDimZ
    {},                         # launch_metadata
    None, None,                 # launch_enter_hook, launch_exit_hook
    a_tensor_map, M, K, K, 1,   # args
    b_tensor_map, N, K, K, 1, 
    c_tensor_map, M, N, N, 1
)

# NOTE: `num_warps` is an important kernel metadata. The `launcher_mod.launch(...)` function uses it to compute `blockDimX` as follows:
# config.blockDimX = 32 * num_warps;
# config.blockDimY = 1;
# config.blockDimZ = 1;

torch.cuda.synchronize()

# Retrieve result
print(c)
