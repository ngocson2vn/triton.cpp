import os
import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

BLOCK_SHAPE = [64, 64]

@triton.jit
def triton_dot(a_desc, b_desc, c_desc):
    a = a_desc.load([0, 0])
    b = b_desc.load([0, 0])
    acc = tl.dot(a, b.T)
    c_desc.store([0, 0], acc)

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


def main():
    a, b = get_input_matrices()

    print(f"a: shape={a.shape} data={a}")
    print(f"b: shape={b.shape} data={b}")

    c_torch_matmul = torch.matmul(a.to(torch.float32), b.T.to(torch.float32))
    torch.cuda.synchronize()
    print(f"c_torch_matmul: shape={c_torch_matmul.shape} dtype={c_torch_matmul.dtype} {c_torch_matmul}")

    def grid(u):
        return (1,)

    c_triton_dot = torch.empty([a.shape[0], b.shape[0]], dtype=torch.float32).cuda()
    a_desc = TensorDescriptor.from_tensor(a, list(a.shape))
    b_desc = TensorDescriptor.from_tensor(b, list(b.shape))
    c_desc = TensorDescriptor.from_tensor(c_triton_dot, [a.shape[0], b.shape[0]])
    triton_dot[grid](a_desc, b_desc, c_desc)

    torch.cuda.synchronize()
    print(f"c_triton_dot: shape={c_triton_dot.shape} dtype={c_triton_dot.dtype} {c_triton_dot}")
    print()

    print("Verify results")
    EPSILON = 1e-2
    matched = True
    mismatch_count = 0
    for i in range(c_torch_matmul.shape[0]):
        for j in range(c_torch_matmul.shape[1]):
            diff = abs((c_torch_matmul[i, j] - c_triton_dot[i, j]))
            if diff > EPSILON:
                matched = False
                print(f"{c_torch_matmul[i, j]} != {c_triton_dot[i, j]}")
                mismatch_count += 1

    if matched:
        print(f"OK: matmul_tma matches torch.matmul")
    else:
        print(f"NG: There are {mismatch_count} mismatches between matmul_tma and torch.matmul")

if __name__ == "__main__":
    main()
