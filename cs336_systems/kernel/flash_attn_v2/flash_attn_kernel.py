import torch
import triton
import triton.language as tl

def Ceil(x, y):
    return (x + y - 1) // y

# Q [batch_size * num_heads, seq_len, head_dim]
# K [batch_size * num_heads, seq_len, head_dim]
# V [batch_size * num_heads, seq_len, head_dim]
# O [batch_size * num_heads, seq_len, head_dim]
# L [batch_size * num_heads, seq_len]
@triton.jit
def flash_attn_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    stride_qb, stride_qs, stride_qd,
    stride_kb, stride_ks, stride_kd,
    stride_vb, stride_vs, stride_vd,
    stride_ob, stride_os, stride_od,
    stride_lb, stride_ls,
    q_seq,
    k_seq,
    scale,
    D_k: tl.constexpr,
    D_v: tl.constexpr,
    q_tile: tl.constexpr,
    k_tile: tl.constexpr
):
    bs_id = tl.program_id(0)
    q_tile_id = tl.program_id(1)
    q_block_ptr = tl.make_block_ptr(
        base = Q_ptr + bs_id * stride_qb,
        shape = (q_seq, D_k),
        strides = (stride_qs, stride_qd),
        offsets = (q_tile_id * q_tile, 0),
        block_shape = (q_tile, D_k),
        order = (1, 0)
    )
    k_block_ptr = tl.make_block_ptr(
        base = K_ptr + bs_id * stride_kb,
        shape = (k_seq, D_k),
        strides = (stride_ks, stride_kd),
        offsets = (0, 0),
        block_shape = (k_tile, D_k),
        order = (1, 0)
    )
    v_block_ptr = tl.make_block_ptr(
        base = V_ptr + bs_id * stride_vb,
        shape = (k_seq, D_v),
        strides = (stride_vs, stride_vd),
        offsets = (0, 0),
        block_shape = (k_tile, D_v),
        order = (1, 0)
    )
    output_block_ptr = tl.make_block_ptr(
        base = O_ptr + bs_id * stride_ob,
        shape = (q_seq, D_v),
        strides = (stride_os, stride_od),
        offsets = (q_tile_id * q_tile, 0),
        block_shape = (q_tile, D_v),
        order = (1, 0)
    )
    l_block_ptr = tl.make_block_ptr(
        base = L_ptr + bs_id * stride_lb,
        shape = (q_seq,),
        strides = (stride_ls,),
        offsets = (q_tile_id * q_tile,),
        block_shape = (q_tile,),
        order = (0,)
    )
    l = tl.zeros((q_tile,), dtype=tl.float32)
    m = tl.full((q_tile,), -float("inf"), dtype=tl.float32)
    o_block = tl.zeros((q_tile, D_v), dtype=tl.float32)
    q_block = tl.load(q_block_ptr)
    for k_tile_id in range(0, k_seq, k_tile):
        k_block = tl.load(k_block_ptr)
        v_block = tl.load(v_block_ptr)
        k_block = tl.trans(k_block)
        s = tl.dot(q_block, k_block) * scale
        new_m = tl.maximum(m, tl.max(s, axis = 1))
        p = tl.exp(s - new_m[:, None])
        delta_m = tl.exp(m - new_m)
        l = delta_m * l + tl.sum(p, axis = 1)
        # o_block = delta_m[:, None] * o_block + tl.dot(p, v_block)
        o_block = delta_m[:, None] * o_block
        o_block = tl.dot(p, v_block, acc=o_block)
        m = new_m
        k_block_ptr = tl.advance(k_block_ptr, (k_tile, 0))
        v_block_ptr = tl.advance(v_block_ptr, (k_tile, 0))
    tl.store(output_block_ptr, o_block / l[:, None])
    tl.store(l_block_ptr, tl.log(l) + m)

def flash_attn_fwd(
    Query, Key, Value, Output, L, scale, head_tile, q_tile, k_tile
):
    batch_size, num_heads, q_seq, D_k = Query.shape
    _, _, k_seq, _ = Key.shape
    _, _, _, D_v = Value.shape
    Query = Query.view(-1, q_seq, D_k)
    Key = Key.view(-1, k_seq, D_k)
    Value = Value.view(-1, k_seq, D_v)
    flash_attn_fwd_kernel[Ceil(batch_size * num_heads, head_tile), Ceil(q_seq, q_tile)](
        Q_ptr = Query,
        K_ptr = Key,
        V_ptr = Value,
        O_ptr = Output,
        L_ptr = L,
        stride_qb = Query.stride(0),
        stride_qs = Query.stride(1),
        stride_qd = Query.stride(2),
        stride_kb = Key.stride(0),
        stride_ks = Key.stride(1),
        stride_kd = Key.stride(2),
        stride_vb = Value.stride(0),
        stride_vs = Value.stride(1),
        stride_vd = Value.stride(2),
        stride_ob = Output.stride(0),
        stride_os = Output.stride(1),
        stride_od = Output.stride(2),
        stride_lb = L.stride(0),
        stride_ls = L.stride(1),
        q_seq = q_seq,
        k_seq = k_seq,
        scale = scale,
        D_k = D_k,
        D_v = D_v,
        q_tile = q_tile,
        k_tile = k_tile
    )