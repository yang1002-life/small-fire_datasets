import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# 论文题目：CATANet: Efficient Content-Aware Token Aggregation for Lightweight Image Super-Resolution CVPR2025
# 论文链接：https://arxiv.org/pdf/2503.06896
# 官方github：https://github.com/EquationWalker/CATANet/tree/main
# 代码改进者：一勺汤


def patch_divide(x, step, ps):
    """
    将输入图像裁剪成多个小块。
    参数:
        x (Tensor): 输入特征图，形状为 (b, c, h, w)，b 是批量大小，c 是通道数，h 是高度，w 是宽度。
        step (int): 裁剪步长。
        ps (int): 小块的尺寸。
    返回:
        crop_x (Tensor): 裁剪后的小块。
        nh (int): 水平方向上的小块数量。
        nw (int): 垂直方向上的小块数量。
    """
    b, c, h, w = x.size()
    # 如果输入图像的高度和宽度等于小块尺寸，将步长设置为小块尺寸
    if h == ps and w == ps:
        step = ps
    crop_x = []
    nh = 0
    # 遍历图像的高度方向
    for i in range(0, h + step - ps, step):
        top = i
        down = i + ps
        # 处理边界情况
        if down > h:
            top = h - ps
            down = h
        nh += 1
        # 遍历图像的宽度方向
        for j in range(0, w + step - ps, step):
            left = j
            right = j + ps
            # 处理边界情况
            if right > w:
                left = w - ps
                right = w
            # 裁剪出小块并添加到列表中
            crop_x.append(x[:, :, top:down, left:right])
    nw = len(crop_x) // nh
    # 将小块列表堆叠成一个张量
    crop_x = torch.stack(crop_x, dim=0)  # (n, b, c, ps, ps)
    # 调整张量的维度
    crop_x = crop_x.permute(1, 0, 2, 3, 4).contiguous()  # (b, n, c, ps, ps)
    return crop_x, nh, nw


def patch_reverse(crop_x, x, step, ps):
    """
    将裁剪后的小块还原成图像。
    参数:
        crop_x (Tensor): 裁剪后的小块。
        x (Tensor): 原始特征图，形状为 (b, c, h, w)。
        step (int): 裁剪步长。
        ps (int): 小块的尺寸。
    返回:
        ouput (Tensor): 还原后的图像。
    """
    b, c, h, w = x.size()
    # 初始化输出张量
    output = torch.zeros_like(x)
    index = 0
    # 遍历图像的高度方向
    for i in range(0, h + step - ps, step):
        top = i
        down = i + ps
        # 处理边界情况
        if down > h:
            top = h - ps
            down = h
        # 遍历图像的宽度方向
        for j in range(0, w + step - ps, step):
            left = j
            right = j + ps
            # 处理边界情况
            if right > w:
                left = w - ps
                right = w
            # 将小块添加到输出张量中
            output[:, :, top:down, left:right] += crop_x[:, index]
            index += 1
    # 处理重叠区域
    for i in range(step, h + step - ps, step):
        top = i
        down = i + ps - step
        if top + ps > h:
            top = h - ps
        output[:, :, top:down, :] /= 2
    # 处理重叠区域
    for j in range(step, w + step - ps, step):
        left = j
        right = j + ps - step
        if left + ps > w:
            left = w - ps
        output[:, :, :, left:right] /= 2
    return output


class PreNorm(nn.Module):
    """
    归一化层。
    参数:
        dim (int): 基础通道数。
        fn (Module): 归一化后的模块。
    """

    def __init__(self, dim, fn):
        super().__init__()
        # 定义层归一化层
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        # 先进行归一化，再传入后续模块
        return self.fn(self.norm(x), **kwargs)


class dwconv(nn.Module):
    def __init__(self, hidden_features, kernel_size=5):
        super(dwconv, self).__init__()
        # 定义深度可分离卷积层
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(hidden_features, hidden_features, kernel_size=kernel_size, stride=1,
                      padding=(kernel_size - 1) // 2, dilation=1,
                      groups=hidden_features), nn.GELU())
        self.hidden_features = hidden_features

    def forward(self, x, x_size):
        # 调整输入张量的维度
        x = x.transpose(1, 2).view(x.shape[0], self.hidden_features, x_size[0], x_size[1]).contiguous()  # b Ph*Pw c
        # 进行深度可分离卷积
        x = self.depthwise_conv(x)
        # 调整输出张量的维度
        x = x.flatten(2).transpose(1, 2).contiguous()
        return x


class ConvFFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, kernel_size=5, act_layer=nn.GELU):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        # 定义第一个全连接层
        self.fc1 = nn.Linear(in_features, hidden_features)
        # 定义激活函数
        self.act = act_layer()
        # 定义深度可分离卷积层
        self.dwconv = dwconv(hidden_features=hidden_features, kernel_size=kernel_size)
        # 定义第二个全连接层
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x, x_size):
        # 通过第一个全连接层
        x = self.fc1(x)
        # 通过激活函数
        x = self.act(x)
        # 加上深度可分离卷积的输出
        x = x + self.dwconv(x, x_size)
        # 通过第二个全连接层
        x = self.fc2(x)
        return x


class Attention(nn.Module):
    """
    注意力模块。
    参数:
        dim (int): 基础通道数。
        heads (int): 注意力头的数量。
        qk_dim (int): 查询和键的通道数。
    """

    def __init__(self, dim, heads, qk_dim):
        super().__init__()

        self.heads = heads
        self.dim = dim
        self.qk_dim = qk_dim
        # 缩放因子
        self.scale = qk_dim ** -0.5

        # 定义查询、键和值的线性层
        self.to_q = nn.Linear(dim, qk_dim, bias=False)
        self.to_k = nn.Linear(dim, qk_dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        # 定义投影层
        self.proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        # 计算查询、键和值
        q, k, v = self.to_q(x), self.to_k(x), self.to_v(x)

        # 调整查询、键和值的维度
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), (q, k, v))

        # 计算注意力输出
        out = F.scaled_dot_product_attention(q, k, v)
        # 调整输出的维度
        out = rearrange(out, 'b h n d -> b n (h d)')
        # 通过投影层
        return self.proj(out)


class LRSA(nn.Module):
    """
    注意力模块。
    参数:
        dim (int): 基础通道数。
        num (int): 块的数量。
        qk_dim (int): 注意力中查询和键的通道数。
        mlp_dim (int): MLP 中隐藏层的通道数。
        heads (int): 注意力头的数量。
    """

    def __init__(self, dim, qk_dim=32, heads=1):
        super().__init__()

        # 定义模块列表
        mlp_dim = 2*dim
        self.layer = nn.ModuleList([
            PreNorm(dim, Attention(dim, heads, qk_dim)),
            PreNorm(dim, ConvFFN(dim, mlp_dim))])

    def forward(self, x):
        ps=8
        step = ps - 2
        # 将输入图像裁剪成小块
        crop_x, nh, nw = patch_divide(x, step, ps)  # (b, n, c, ps, ps)
        b, n, c, ph, pw = crop_x.shape
        # 调整小块的维度
        crop_x = rearrange(crop_x, 'b n c h w -> (b n) (h w) c')

        attn, ff = self.layer
        # 通过注意力模块并加上残差连接
        crop_x = attn(crop_x) + crop_x
        # 调整小块的维度
        crop_x = rearrange(crop_x, '(b n) (h w) c  -> b n c h w', n=n, w=pw)

        # 将小块还原成图像
        x = patch_reverse(crop_x, x, step, ps)
        _, _, h, w = x.shape
        # 调整图像的维度
        x = rearrange(x, 'b c h w-> b (h w) c')
        # 通过 MLP 模块并加上残差连接
        x = ff(x, x_size=(h, w)) + x
        # 调整图像的维度
        x = rearrange(x, 'b (h w) c->b c h w', h=h)

        return x


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        x = x.to(self.conv.weight.device)  # 确保输入张量在卷积层所在设备
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        x = x.to(self.conv.weight.device)  # 确保输入张量在卷积层所在设备
        return self.act(self.conv(x))


class PSABloc_LRSA(nn.Module):
    """
    PSABlock class implementing a Position-Sensitive Attention block for neural networks.

    This class encapsulates the functionality for applying multi-head attention and feed-forward neural network layers
    with optional shortcut connections.

    Attributes:
        attn (Attention): Multi-head attention module.
        ffn (nn.Sequential): Feed-forward neural network module.
        add (bool): Flag indicating whether to add shortcut connections.

    Methods:
        forward: Performs a forward pass through the PSABlock, applying attention and feed-forward layers.

    Examples:
        Create a PSABlock and perform a forward pass
        >>> psablock = PSABlock(c=128, attn_ratio=0.5, num_heads=4, shortcut=True)
        >>> input_tensor = torch.randn(1, 128, 32, 32)
        >>> output_tensor = psablock(input_tensor)
    """

    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True) -> None:
        """Initializes the PSABlock with attention and feed-forward layers for enhanced feature extraction."""
        super().__init__()

        self.attn = LRSA(c)
        self.ffn = nn.Sequential(Conv(c, c * 2, 1), Conv(c * 2, c, 1, act=False))
        self.add = shortcut

    def forward(self, x):
        """Executes a forward pass through PSABlock, applying attention and feed-forward layers to the input tensor."""
        x = x + self.attn(x) if self.add else self.attn(x)
        x = x + self.ffn(x) if self.add else self.ffn(x)
        return x


class C2PSA_LRSA(nn.Module):
    """
    C2PSA module with attention mechanism for enhanced feature extraction and processing.

    This module implements a convolutional block with attention mechanisms to enhance feature extraction and processing
    capabilities. It includes a series of PSABlock modules for self-attention and feed-forward operations.

    Attributes:
        c (int): Number of hidden channels.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c.
        m (nn.Sequential): Sequential container of PSABlock modules for attention and feed-forward operations.

    Methods:
        forward: Performs a forward pass through the C2PSA module, applying attention and feed-forward operations.

    Notes:
        This module essentially is the same as PSA module, but refactored to allow stacking more PSABlock modules.

    Examples:
        >>> c2psa = C2PSA(c1=256, c2=256, n=3, e=0.5)
        >>> input_tensor = torch.randn(1, 256, 64, 64)
        >>> output_tensor = c2psa(input_tensor)
    """

    def __init__(self, c1, c2, n=1, e=0.5):
        """Initializes the C2PSA module with specified input/output channels, number of layers, and expansion ratio."""
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)

        self.m = nn.Sequential(*(PSABloc_LRSA(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))

    def forward(self, x):
        """Processes the input tensor 'x' through a series of PSA blocks and returns the transformed tensor."""
        x = x.to(self.cv1.conv.weight.device)  # 确保输入张量在第一个卷积层所在设备
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = self.m(b)
        return self.cv2(torch.cat((a, b), 1))





if __name__ == "__main__":
    # 定义输入张量
    x = torch.randn(2, 128, 27, 31)
    # 定义基础通道数
    dim = 128
    # 定义查询和键的通道数
    qk_dim = 32
    # 定义小块尺寸
    ps = 8
    # 创建 LRSA 模块
    lrsa = LRSA(dim, qk_dim)
    # 前向传播
    output = lrsa(x)
    print("输出形状:", output.shape)
