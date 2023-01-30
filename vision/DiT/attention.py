import mindspore.nn as nn
import mindspore as ms


class Attention(nn.Cell):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super(Attention, self).__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Dense(dim, dim * 3, has_bias=qkv_bias)
        self.attn_drop = nn.Dropout(1.0 - attn_drop)
        self.proj = nn.Dense(dim, dim)
        self.proj_drop = nn.Dropout(1.0 - proj_drop)

    def construct(self, x):
        unstack = ms.ops.Unstack()
        transpose = ms.ops.Transpose()
        input_perm = (2, 0, 3, 1, 4)
        B, N, C = x.shape
        qkv = transpose(self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads), input_perm)
        q, k, v = unstack(qkv)  # make torchscript happy (cannot use tensor as tuple)
        batmatmul = ms.ops.BatchMatMul()
        attn = (batmatmul(q, transpose(k, (0, 1, 3, 2)))) * self.scale
        attn = ms.ops.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)

        x = transpose((batmatmul(attn, v)), (0, 2, 1, 3)).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
