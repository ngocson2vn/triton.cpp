import torch
import triton
import triton.language as tl

def keep(config):
    m = config.kwargs["BLOCK_M"]
    n = config.kwargs["BLOCK_N"]
    # from: https://github.com/triton-lang/triton/issues/4502
    if torch.cuda.get_device_properties(0).major >= 9:
        if m == 64 and config.num_warps == 8:
            return False
    return m % n == 0

configs = [
    triton.Config({"BLOCK_M": BM, "BLOCK_N": BN}, num_stages=s, num_warps=w)  \
    for BM in [32]  \
    for BN in [32, 64, 128]  \
    for s in [1, 2, 3, 4]  \
    for w in [4, 8]
]

@triton.jit
def mask_fn(q_mask, k_mask, q_offset, k_offset, TYPE: tl.constexpr):
    tril_causal = q_offset[:, None] >= k_offset[None, :]
    triu_causal = q_offset[:, None] <= k_offset[None, :]
    if TYPE == 1:
        return (triu_causal & (q_mask[:, None] == 1) & (k_mask[None, :] == 1)) | (q_offset[:, None] == k_offset[None, :])
    if TYPE == 2:
        return (tril_causal & (q_mask[:, None] == 1) & (k_mask[None, :] == 1)) | (q_offset[:, None] == k_offset[None, :])
    if TYPE == 3:
        return triu_causal & (((k_offset[None, :] - q_offset[:, None]) < q_mask[:, None]) | (q_mask[:, None] == 0) | (k_mask[None, :] == 0))
    if TYPE == 4:
        return (tril_causal & (q_mask[:, None] != 0) & (k_mask[None, :] == 1)) | ((q_mask[:, None] == k_mask[None, :]) & (k_mask[None, :] > 1))
    if TYPE == 5:
        return (triu_causal & (q_mask[:, None] != 0) & (k_mask[None, :] == 1)) | ((q_mask[:, None] == k_mask[None, :]) & (k_mask[None, :] > 1))
    if TYPE == 6:
        return (tril_causal & (q_mask[:, None] != 0) & (k_mask[None, :] == 1)) | ((q_mask[:, None] == k_mask[None, :]) & (k_mask[None, :] > 1) & tril_causal)
    if TYPE == 8:
        return (tril_causal & (q_mask[:, None] != 0) & (k_mask[None, :] == 1)) | ((q_mask[:, None] == k_mask[None, :]) & (k_mask[None, :] > 1) & (q_offset[:, None] == k_offset[None, :]))
    if TYPE == 100:
        return tril_causal & (((q_offset[:, None] - k_offset[None, :]) < q_mask[:, None]) | (q_mask[:, None] == 0) | (k_mask[None, :] == 0))
    if TYPE == 102:
        p = q_mask[:, None] != k_mask[None, :]
        #q_mask[q_mask != 0] = 1
        t = tl.where(q_mask != 0, 1, q_mask)
        return tril_causal & ((((q_offset[:, None] - k_offset[None, :]) < t[:, None]) | (q_mask[:, None] == 0) | (k_mask[None, :] == 0)) | p)
    if TYPE == 103:
        return q_mask[:, None] == k_mask[None, :]
    if TYPE == 104:
        # k_mask: 0 = u token (visible to all queries within the batch), >0 = g group id
        # q_mask: g group id (consistent with the g group corresponding to k, >0)
        k_is_u = k_mask[None, :] == 0
        same_group = (q_mask[:, None] == k_mask[None, :]) & (k_mask[None, :] > 0)
        return k_is_u | same_group

@triton.autotune(list(filter(keep, configs)), key = ["QK_DIM", "V_DIM", "MASK_FN", "SPARSE_OPT"])
@triton.jit
def fwd_kernel(
    q_ptr, k_ptr, v_ptr, o_ptr, l_ptr,
    q_mask_ptr, k_mask_ptr,
    cu_seqlens_q, cu_seqlens_k,
    q_head, kv_head, scale,
    QK_DIM: tl.constexpr, V_DIM: tl.constexpr, MASK_FN: tl.constexpr, SPARSE_OPT: tl.constexpr, DTYPE: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    dtype = o_ptr.type.element_ty
    start_m = tl.program_id(0)
    start_qh = tl.program_id(1)
    start_b = tl.program_id(2)
    start_kvh = start_qh // (q_head // kv_head)

    q_start = tl.load(cu_seqlens_q + start_b)
    q_end = tl.load(cu_seqlens_q + start_b + 1)
    q_len = q_end - q_start
    if start_m * BLOCK_M >= q_len:
        return

    k_start = tl.load(cu_seqlens_k + start_b)
    k_end = tl.load(cu_seqlens_k + start_b + 1)
    k_len = k_end - k_start

    if SPARSE_OPT:
        begin = 0
        end = k_len
        if k_len==0:
            acc = tl.zeros((BLOCK_M, V_DIM), dtype=tl.bfloat16)
            o_block_ptr = tl.make_block_ptr(
                base = o_ptr + q_start * q_head * V_DIM + start_qh * V_DIM,
                shape = (q_len, V_DIM),
                strides = (q_head * V_DIM, 1),
                offsets = (start_m * BLOCK_M, 0),
                block_shape = (BLOCK_M, V_DIM),
                order = (1, 0)
            )

            # Replaced store_if with native tl.store for TMA compatibility
            tl.store(o_block_ptr, acc.to(dtype), boundary_check=(0,))
            return
    else:
        if MASK_FN & 1:
            begin = start_m * BLOCK_M
            if begin >= k_len:
                return
            end = k_len
        else:
            begin = 0
            end = tl.minimum((start_m + 1) * BLOCK_M, k_len)

    log2e: tl.constexpr = 1.4426950408889634
    qk_scale = scale * log2e
    offset_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)

    q_start = q_start.to(tl.int64)
    k_start = k_start.to(tl.int64)
    q_block_ptr = tl.make_block_ptr(
        base = q_ptr + q_start * q_head * QK_DIM + start_qh * QK_DIM,
        shape = (q_len, QK_DIM),
        strides = (q_head * QK_DIM, 1),
        offsets = (start_m * BLOCK_M, 0),
        block_shape = (BLOCK_M, QK_DIM),
        order = (1, 0)
    )
    k_block_ptr = tl.make_block_ptr(
        base = k_ptr + k_start * kv_head * QK_DIM + start_kvh * QK_DIM,
        shape = (QK_DIM, k_len),
        strides = (1, kv_head * QK_DIM),
        offsets = (0, begin),
        block_shape = (QK_DIM, BLOCK_N),
        order = (0, 1)
    )
    v_block_ptr = tl.make_block_ptr(
        base = v_ptr + k_start * kv_head * V_DIM + start_kvh * V_DIM,
        shape = (k_len, V_DIM),
        strides = (kv_head * V_DIM, 1),
        offsets = (begin, 0),
        block_shape = (BLOCK_N, V_DIM),
        order = (1, 0)
    )
    o_block_ptr = tl.make_block_ptr(
        base = o_ptr + q_start * q_head * V_DIM + start_qh * V_DIM,
        shape = (q_len, V_DIM),
        strides = (q_head * V_DIM, 1),
        offsets = (start_m * BLOCK_M, 0),
        block_shape = (BLOCK_M, V_DIM),
        order = (1, 0)
    )
    
    # Standard pointers for 1D tensors (masks & lengths)
    l_ptrs = l_ptr + q_start * q_head + start_qh + offset_m
    q_mask_ptrs = q_mask_ptr + q_start + offset_m
    
    acc = tl.zeros((BLOCK_M, V_DIM), dtype=tl.float32)
    m = tl.full((BLOCK_M,), value=-2**30, dtype=tl.float32)
    l = tl.zeros((BLOCK_M,), dtype=tl.float32)

    # Native loads exposed to the compiler
    q = tl.load(q_block_ptr, boundary_check=(0,))
    q_mask = tl.load(q_mask_ptrs, mask=offset_m < q_len, other=0)

    for start_n in range(begin, end, BLOCK_N, warp_specialize=True):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        offset_n = start_n + tl.arange(0, BLOCK_N)
        k_mask_ptrs = k_mask_ptr + k_start + offset_n
        k_mask = tl.load(k_mask_ptrs, mask=offset_n < k_len, other=0)
        mask = mask_fn(q_mask, k_mask, offset_m, offset_n, MASK_FN)
        if not SPARSE_OPT or tl.sum(mask.cast(tl.int32)) != 0:
            # Native block pointer loads for TMA tracking
            k = tl.load(k_block_ptr, boundary_check=(1,))
            v = tl.load(v_block_ptr, boundary_check=(0,))
            
            s = tl.dot(q, k)
            boundary_mask = (offset_n < k_len)[None, :]
            s = tl.where(mask & boundary_mask, s, -2**30)
            m_new = tl.maximum(m, tl.max(s, 1))
            alpha = tl.math.exp2((m - m_new) * qk_scale)
            p = tl.math.exp2((s - m_new[:, None]) * qk_scale)
            p_sum = tl.sum(p, 1)
            acc *= alpha[:, None]
            acc += tl.dot(p.to(dtype), v)
            l = l * alpha + p_sum
            m = m_new
        k_block_ptr = tl.advance(k_block_ptr, (0, BLOCK_N))
        v_block_ptr = tl.advance(v_block_ptr, (BLOCK_N, 0))

    is_nonempty = l > 0
    inv_l = tl.where(is_nonempty, 1.0 / l, 0.0)
    acc = acc * inv_l[:, None]
    lse = tl.where(is_nonempty, tl.log(l), -float("inf"))
    l_out = m * scale + lse
    
    # Store results natively
    tl.store(l_ptrs, l_out, mask=offset_m < q_len)
    tl.store(o_block_ptr, acc.to(dtype), boundary_check=(0,))


# max_seqlen_q = 256
# q_head = 64
# batch_size = 8
# grid = lambda META: (triton.cdiv(max_seqlen_q, META["BLOCK_M"]), q_head, batch_size)
# fwd_kernel[grid](
#     q, k, v, o, l,
#     q_mask, k_mask,
#     cu_seqlens_q, cu_seqlens_k,
#     q_head, kv_head, scale,
#     QK_DIM = qk_dim,
#     V_DIM = v_dim,
#     MASK_FN = mask_fn,
#     SPARSE_OPT = sparse_opt,
#     DTYPE = (19 if q.dtype == torch.float16 else 14),
# )
