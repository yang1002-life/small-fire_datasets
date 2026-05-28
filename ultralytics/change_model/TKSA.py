import torch
import torch.nn as nn

from einops import rearrange


#
# 论文题目：Learning A Sparse Transformer Network for Effective Image Deraining
# 论文链接：https://arxiv.org/pdf/2303.11950
# 官方github：https://github.com/cschenxiang/DRSformer
# 代码改进者：一勺汤

##  Top-K Sparse Attention (TKSA)
class TKSAttention(nn.Module):
    def __init__(self, dim, num_heads=4, bias=False):
        """
        初始化注意力模块。

        参数:
        dim (int): 输入特征图的通道维度。
        num_heads (int): 注意力头的数量。
        bias (bool): 是否在卷积层中使用偏置。
        """
        super(TKSAttention, self).__init__()
        self.num_heads = num_heads

        # 可学习的温度参数，用于调整注意力分数的缩放
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        # 用于生成查询、键和值的 1x1 卷积层
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        # 对查询、键和值进行深度可分离卷积的卷积层
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        # 用于输出结果的 1x1 卷积层
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        # Dropout 层，用于防止过拟合
        self.attn_drop = nn.Dropout(0.)

        # 可学习的权重参数，用于融合不同 Top-K 计算得到的注意力结果
        self.attn1 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn2 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn3 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn4 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)

    def forward(self, x):
        """
        前向传播函数。

        参数:
        x (torch.Tensor): 输入的特征图，形状为 (b, c, h, w)，其中 b 是批量大小，c 是通道数，h 是高度，w 是宽度。

        返回:
        torch.Tensor: 经过注意力计算后的输出特征图，形状为 (b, c, h, w)。
        """
        b, c, h, w = x.shape

        # 生成查询、键和值，并进行深度可分离卷积
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        # 将查询、键和值重新排列，以便按头进行计算
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        # 对查询和键进行归一化处理
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        _, _, C, _ = q.shape

        # 初始化四个掩码张量，用于标记 Top-K 选择的位置
        mask1 = torch.zeros(b, self.num_heads, C, C, device=x.device, requires_grad=False)
        mask2 = torch.zeros(b, self.num_heads, C, C, device=x.device, requires_grad=False)
        mask3 = torch.zeros(b, self.num_heads, C, C, device=x.device, requires_grad=False)
        mask4 = torch.zeros(b, self.num_heads, C, C, device=x.device, requires_grad=False)

        # 计算注意力分数，并乘以温度参数
        attn = (q @ k.transpose(-2, -1)) * self.temperature

        # 选择前 C/2 个最大的注意力分数的索引，并更新掩码 mask1
        index = torch.topk(attn, k=int(C/2), dim=-1, largest=True)[1]
        mask1.scatter_(-1, index, 1.)
        # 根据掩码 mask1 生成注意力分数 attn1，将未选择的位置设置为负无穷
        attn1 = torch.where(mask1 > 0, attn, torch.full_like(attn, float('-inf')))

        # 选择前 C*2/3 个最大的注意力分数的索引，并更新掩码 mask2
        index = torch.topk(attn, k=int(C*2/3), dim=-1, largest=True)[1]
        mask2.scatter_(-1, index, 1.)
        # 根据掩码 mask2 生成注意力分数 attn2，将未选择的位置设置为负无穷
        attn2 = torch.where(mask2 > 0, attn, torch.full_like(attn, float('-inf')))

        # 选择前 C*3/4 个最大的注意力分数的索引，并更新掩码 mask3
        index = torch.topk(attn, k=int(C*3/4), dim=-1, largest=True)[1]
        mask3.scatter_(-1, index, 1.)
        # 根据掩码 mask3 生成注意力分数 attn3，将未选择的位置设置为负无穷
        attn3 = torch.where(mask3 > 0, attn, torch.full_like(attn, float('-inf')))

        # 选择前 C*4/5 个最大的注意力分数的索引，并更新掩码 mask4
        index = torch.topk(attn, k=int(C*4/5), dim=-1, largest=True)[1]
        mask4.scatter_(-1, index, 1.)
        # 根据掩码 mask4 生成注意力分数 attn4，将未选择的位置设置为负无穷
        attn4 = torch.where(mask4 > 0, attn, torch.full_like(attn, float('-inf')))

        # 对四个注意力分数分别进行 softmax 操作，得到注意力权重
        attn1 = attn1.softmax(dim=-1)
        attn2 = attn2.softmax(dim=-1)
        attn3 = attn3.softmax(dim=-1)
        attn4 = attn4.softmax(dim=-1)

        # 根据注意力权重计算输出
        out1 = (attn1 @ v)
        out2 = (attn2 @ v)
        out3 = (attn3 @ v)
        out4 = (attn4 @ v)

        # 融合四个输出结果，使用可学习的权重参数进行加权求和
        out = out1 * self.attn1 + out2 * self.attn2 + out3 * self.attn3 + out4 * self.attn4

        # 将输出重新排列回原始的形状
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        # 通过输出卷积层得到最终结果
        out = self.project_out(out)
        return out


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


class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a standard bottleneck module with optional shortcut connection and configurable parameters."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initializes a CSP bottleneck with 2 convolutions and n Bottleneck blocks for faster processing."""
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = self.cv1(x).split((self.c, self.c), 1)
        y = [y[0], y[1]]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

class C3(nn.Module):
    """CSP Bottleneck with 3 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize the CSP Bottleneck with given channels, number, shortcut, groups, and expansion values."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))

class Bottleneck_TKSAttention(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a standard bottleneck module with optional shortcut connection and configurable parameters."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.cv3 = TKSAttention(c2)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Applies the YOLO FPN to input data."""
        return x + self.cv3(self.cv2(self.cv1(x))) if self.add else self.cv3(self.cv2(self.cv1(x)))

class C3k(C3):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        """Initializes the C3k module with specified channels, number of layers, and configurations."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        # self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))
        self.m = nn.Sequential(*(Bottleneck_TKSAttention(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))

# 在c3k=True时，使用Bottleneck_LLSKM特征融合，为false的时候我们使用普通的Bottleneck提取特征
class C3k2_TKSA(C2f):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        """Initializes the C3k2 module, a faster CSP Bottleneck with 2 convolutions and optional C3k blocks."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck_TKSAttention(self.c, self.c, shortcut, g) for _ in range(n)
        )


if __name__ == "__main__":
    # 定义输入特征图的通道维度、注意力头的数量和是否使用偏置
    dim = 64
    num_heads = 4
    bias = True

    # 创建注意力模块实例
    attention = TKSAttention(dim, num_heads, bias)
    # 生成随机输入特征图
    x = torch.randn(1, dim, 27, 31)
    # 进行前向传播
    output = attention(x)
    print("输出特征图的形状:", output.shape)