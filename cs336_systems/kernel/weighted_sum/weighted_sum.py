from einops import rearrange
import torch
import triton
import triton.language as tl

def Cell_Div(x, y):
    return (x + y - 1) // y

def weighted_sum_fwd_baseline(x: torch.Tensor, weight: torch.Tensor):
    return torch.einsum("... c, c -> ...", x, weight)

def weighted_sum_bwd_baseline(x: torch.Tensor, weight: torch.Tensor, output_grad: torch.Tensor):
    # dx = grad_out * weight
    x_grad = torch.einsum("... , c -> ... c", output_grad, weight)
    # dw = sum(grad_out * x)
    w_grad = torch.einsum("... , ... c -> c", output_grad, x)
    return x_grad, w_grad

@triton.jit
def weighted_sum_fwd_kernel(
    x_ptr, weight_ptr,
    output_ptr,
    x_row_stride, x_dim_stride,
    weight_dim_stride,
    output_row_stride,
    rows, dims,
    ROW_TILE_SIZE: tl.constexpr,
    DIM_TILE_SIZE: tl.constexpr
):
    row_tile_idx = tl.program_id(0)
    x_block_ptr = tl.make_block_ptr(
        base = x_ptr, shape = (rows, dims), strides = (x_row_stride, x_dim_stride),
        offsets = (row_tile_idx * ROW_TILE_SIZE, 0), block_shape = (ROW_TILE_SIZE, DIM_TILE_SIZE), order = (1, 0)
    )
    weight_block_ptr = tl.make_block_ptr(
        base = weight_ptr, shape = (dims,), strides = (weight_dim_stride,),
        offsets = (0,), block_shape = (DIM_TILE_SIZE,), order = (0,)
    )
    output_block_ptr = tl.make_block_ptr(
        base = output_ptr, shape = (rows,), strides = (output_row_stride,),
        offsets = (row_tile_idx * ROW_TILE_SIZE,), block_shape = (ROW_TILE_SIZE,), order = (0,)
    )
    output = tl.zeros((ROW_TILE_SIZE,), dtype=tl.float32)
    for dim_tile_idx in range(tl.cdiv(dims, DIM_TILE_SIZE)):
        x = tl.load(x_block_ptr, boundary_check=(0,1), padding_option="zero")
        weight = tl.load(weight_block_ptr, boundary_check=(0,), padding_option="zero")
        output += tl.sum(x * weight[None, :], axis=1)
        x_block_ptr = tl.advance(x_block_ptr, (0, DIM_TILE_SIZE))
        weight_block_ptr = tl.advance(weight_block_ptr, (DIM_TILE_SIZE,))
    tl.store(output_block_ptr, output, boundary_check=(0,))

@triton.jit
def weighted_sum_bwd_kernel(
    x_ptr, weight_ptr, output_grad_ptr,
    x_grad_ptr, weight_grad_ptr,
    x_row_stride, x_dim_stride,
    weight_dim_stride,
    output_grad_row_stride,
    rows, dims,
    ROW_TILE_SIZE: tl.constexpr,
    DIM_TILE_SIZE: tl.constexpr
):
    row_tile_idx = tl.program_id(0)
    x_block_ptr = tl.make_block_ptr(
        base = x_ptr, shape = (rows, dims), strides = (x_row_stride, x_dim_stride),
        offsets = (row_tile_idx * ROW_TILE_SIZE, 0), block_shape = (ROW_TILE_SIZE, DIM_TILE_SIZE), order = (1, 0)
    )
    x_grad_block_ptr = tl.make_block_ptr( # 为 x_grad 配置专属指针
        base = x_grad_ptr, shape = (rows, dims), strides = (x_row_stride, x_dim_stride),
        offsets = (row_tile_idx * ROW_TILE_SIZE, 0), block_shape = (ROW_TILE_SIZE, DIM_TILE_SIZE), order = (1, 0)
    )
    weight_block_ptr = tl.make_block_ptr(
        base = weight_ptr, shape = (dims,), strides = (weight_dim_stride,),
        offsets = (0,), block_shape = (DIM_TILE_SIZE,), order = (0,)
    )
    output_grad_block_ptr = tl.make_block_ptr(
        base = output_grad_ptr, shape = (rows,), strides = (output_grad_row_stride,),
        offsets = (row_tile_idx * ROW_TILE_SIZE,), block_shape = (ROW_TILE_SIZE,), order = (0,)
    )
    output_grad = tl.load(output_grad_block_ptr, boundary_check=(0,), padding_option="zero")
    output_grad_expanded = tl.expand_dims(output_grad, 1) # [ROW_TILE_SIZE, 1]

    for dim_tile_idx in range(tl.cdiv(dims, DIM_TILE_SIZE)):
        x = tl.load(x_block_ptr, boundary_check=(0, 1), padding_option="zero")
        weight = tl.load(weight_block_ptr, boundary_check=(0,), padding_option="zero")
        # 1. x_grad: dx = grad_out * w
        x_grad = output_grad_expanded * weight[None, :]
        tl.store(x_grad_block_ptr, x_grad, boundary_check=(0, 1))
        # 2. weight_grad: dw = sum(grad_out * x)
        dw_partial = tl.sum(output_grad_expanded * x, axis=0)
        # atomic add
        col_offsets = dim_tile_idx * DIM_TILE_SIZE + tl.arange(0, DIM_TILE_SIZE)
        tl.atomic_add(weight_grad_ptr + col_offsets * weight_dim_stride, dw_partial, mask=col_offsets < dims)
        x_block_ptr = tl.advance(x_block_ptr, (0, DIM_TILE_SIZE))
        x_grad_block_ptr = tl.advance(x_grad_block_ptr, (0, DIM_TILE_SIZE))
        weight_block_ptr = tl.advance(weight_block_ptr, (DIM_TILE_SIZE,))


class weighted_sum():
    def __init__(self, ROW_TILE_SIZE=16, DIM_TILE_SIZE=16):
        self.ROW_TILE_SIZE = ROW_TILE_SIZE
        self.DIM_TILE_SIZE = DIM_TILE_SIZE
    def forward(self, ctx, x, weight):
        D, output_dims = x.shape[-1], x.shape[:-1]
        input_shape = x.shape
        x = rearrange(x, "... d -> (...) d")
        ctx.save_for_backward(x, weight)
        # 修复：保存真实的 D 和 shape
        ctx.D = D
        ctx.input_shape = input_shape
        ctx.ROWS_TILE_SIZE = self.ROW_TILE_SIZE
        # 修复：确保 DIM_TILE_SIZE 至少为 16，避免 0 或太小导致越界
        ctx.D_TILE_SIZE = max(16, triton.next_power_of_2(D) // 16)
        y = torch.empty(x.shape[0], device=x.device)
        n_rows = y.numel()
        weighted_sum_fwd_kernel[(Cell_Div(n_rows, ctx.ROWS_TILE_SIZE),)](
            x, weight, y,
            x.stride(0), x.stride(1), weight.stride(0), y.stride(0),
            rows=n_rows, dims=D,
            ROW_TILE_SIZE=ctx.ROWS_TILE_SIZE, DIM_TILE_SIZE=ctx.D_TILE_SIZE,
        )
        return y.view(output_dims)

    def backward(self, ctx, grad_out):
        x, weight = ctx.saved_tensors
        D, input_shape = ctx.D, ctx.input_shape
        grad_out_1d = grad_out.contiguous().view(-1)
        x_grad = torch.empty_like(x)
        weight_grad = torch.zeros_like(weight)
        n_rows = x_grad.shape[0]
        weighted_sum_bwd_kernel[(Cell_Div(n_rows, ctx.ROWS_TILE_SIZE),)](
            x, weight, grad_out_1d,
            x_grad, weight_grad,
            x.stride(0), x.stride(1), weight.stride(0), grad_out_1d.stride(0),
            rows=n_rows, dims=D,
            ROW_TILE_SIZE=ctx.ROWS_TILE_SIZE, DIM_TILE_SIZE=ctx.D_TILE_SIZE,
        )
        return x_grad.view(input_shape), weight_grad

class Ctx:
    def __init__(self):
        self.saved_tensors = ()
    def save_for_backward(self, *tensors):
        self.saved_tensors = tensors

if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(64, 128, 65536, device="cuda")
    weight = torch.randn(65536, device="cuda")
    fn = weighted_sum()
    ctx = Ctx()
    # 1. 测试前向传播
    y_optim = fn.forward(ctx, x, weight)
    y_baseline = weighted_sum_fwd_baseline(x, weight)
    print("Forward:", (y_optim - y_baseline).abs().max().item())
    # 2. 测试反向传播
    grad_out = torch.randn_like(y_optim)
    x_grad, w_grad = fn.backward(ctx, grad_out)
    # 获取 Baseline 梯度
    x_grad_baseline, w_grad_baseline = weighted_sum_bwd_baseline(x, weight, grad_out)
    print("Backward x:", (x_grad - x_grad_baseline).abs().max().item())
    print("Backward weight:", (w_grad - w_grad_baseline).abs().max().item())