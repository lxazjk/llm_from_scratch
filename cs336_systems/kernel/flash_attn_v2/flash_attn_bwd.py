import torch
import triton
import triton.language as tl

@triton.jit
def flash_attn_bwd_kernel(
    Q_ptr, K_ptr, V_ptr, L_ptr, D_ptr,
    DO_ptr, DQ_ptr, DK_ptr, DV_ptr,
    stride_qb, stride_qs, stride_qd,
    stride_kb, stride_ks, stride_kd,
    stride_vb, stride_vs, stride_vd,
    stride_lb, stride_ls,
    stride_db, stride_ds,
    stride_dob, stride_dos, stride_dod,
    stride_dqb, stride_dqs, stride_dqd,
    stride_dkb, stride_dks, stride_dkd,
    stride_dvb, stride_dvs, stride_dvd,
    q_seq, k_seq, scale,
    D_QK: tl.constexpr, D_V: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    bs_id = tl.program_id(0)
    k_tile_id = tl.program_id(1)
    k_block_ptr = tl.make_block_ptr(
        base=K_ptr + bs_id * stride_kb,
        shape=(k_seq, D_QK),
        strides=(stride_ks, stride_kd),
        offsets=(k_tile_id * BLOCK_N, 0),
        block_shape=(BLOCK_N, D_QK),
        order=(1, 0)
    )
    v_block_ptr = tl.make_block_ptr(
        base=V_ptr + bs_id * stride_vb,
        shape=(k_seq, D_V),
        strides=(stride_vs, stride_vd),
        offsets=(k_tile_id * BLOCK_N, 0),
        block_shape=(BLOCK_N, D_V),
        order=(1, 0)
    )
    k = tl.load(k_block_ptr)
    v = tl.load(v_block_ptr)
    dk = tl.zeros((BLOCK_N, D_QK), dtype=tl.float32)
    dv = tl.zeros((BLOCK_N, D_V), dtype=tl.float32)
    for i in range(0, q_seq, BLOCK_M):
        q_ptr = tl.make_block_ptr(
            base=Q_ptr + bs_id * stride_qb,
            shape=(q_seq, D_QK),
            strides=(stride_qs, stride_qd),
            offsets=(i, 0), block_shape=(BLOCK_M, D_QK), order=(1, 0))
        do_ptr = tl.make_block_ptr(
            base=DO_ptr + bs_id * stride_dob,
            shape=(q_seq, D_V),
            strides=(stride_dos, stride_dod),
            offsets=(i, 0),
            block_shape=(BLOCK_M, D_V),
            order=(1, 0)
        )
        l_ptr = L_ptr + bs_id * stride_lb + i + tl.arange(0, BLOCK_M)
        d_ptr = D_ptr + bs_id * stride_db + i + tl.arange(0, BLOCK_M)
        q = tl.load(q_ptr)
        do = tl.load(do_ptr)
        l_i = tl.load(l_ptr)
        di = tl.load(d_ptr)
        s = tl.dot(q, tl.trans(k)) * scale
        p = tl.exp(s - l_i[:, None])
        dv += tl.dot(tl.trans(p.to(do.dtype)), do)
        dp = tl.dot(do, tl.trans(v))
        ds = p * (dp - di[:, None]) * scale
        dq_tile = tl.dot(ds.to(k.dtype), k)
        dq_off = DQ_ptr + bs_id * stride_dqb + (i + tl.arange(0, BLOCK_M))[:, None] * stride_dqs + tl.arange(0, D_QK)[None, :] * stride_dqd
        tl.atomic_add(dq_off, dq_tile)
        dk += tl.dot(tl.trans(ds.to(q.dtype)), q)
    dk_block_ptr = tl.make_block_ptr(
        base=DK_ptr + bs_id * stride_dkb,
        shape=(k_seq, D_QK),
        strides=(stride_dks, stride_dkd),
        offsets=(k_tile_id * BLOCK_N, 0),
        block_shape=(BLOCK_N, D_QK),
        order=(1, 0)
    )
    dv_block_ptr = tl.make_block_ptr(
        base=DV_ptr + bs_id * stride_dvb,
        shape=(k_seq, D_V),
        strides=(stride_dvs, stride_dvd),
        offsets=(k_tile_id * BLOCK_N, 0),
        block_shape=(BLOCK_N, D_V),
        order=(1, 0)
    )
    tl.store(dk_block_ptr, dk.to(DK_ptr.dtype.element_ty))
    tl.store(dv_block_ptr, dv.to(DV_ptr.dtype.element_ty))

def flash_attn_bwd(Q, K, V, O, L, dO, sm_scale, BLOCK_M, BLOCK_N):
    B, q_seq, d_k = Q.shape
    d_v = V.shape[-1]
    k_seq = K.shape[2]

    D = torch.sum(dO.to(torch.float32) * O.to(torch.float32), dim=-1)

    dQ = torch.zeros_like(Q)
    dK = torch.empty_like(K)
    dV = torch.empty_like(V)
    grid = (B, triton.cdiv(k_seq, BLOCK_N))

    flash_attn_bwd_kernel[grid](
        Q, K, V, L, D,
        dO, dQ, dK, dV,
        Q.stride(0), Q.stride(1), Q.stride(2),
        K.stride(0), K.stride(1), K.stride(2),
        V.stride(0), V.stride(1), V.stride(2),
        L.stride(0), L.stride(1),
        D.stride(0), D.stride(1),
        dO.stride(0), dO.stride(1), dO.stride(2),
        dQ.stride(0), dQ.stride(1), dQ.stride(2),
        dK.stride(0), dK.stride(1), dK.stride(2),
        dV.stride(0), dV.stride(1), dV.stride(2),
        q_seq, k_seq, sm_scale,
        D_QK=d_k, D_V=d_v,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
    )

    return dQ, dK, dV