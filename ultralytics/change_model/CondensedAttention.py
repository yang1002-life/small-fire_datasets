import torch
import torch.nn as nn
import torch.nn.functional as F


#
# 论文题目：Comprehensive-and-Delicate-An-Efficient-Transformer-for-Image-Restoration
# 论文链接：https://pengxi.me/wp-content/uploads/2023/04/Comprehensive-and-Delicate-An-Efficient-Transformer-for-Image-Restoration.pdf
# 官方github: https://github.com/XLearning-SCU/2023-CVPR-CODE/blob/main/model.py
# 代码改进者：一勺汤
# --------------------------- 通道注意力模块 ---------------------------#
class ChannelAttention(nn.Module):
    def __init__(self, embed_dim, num_chans, expan_att_chans):
        super().__init__()
        self.expan_att_chans = expan_att_chans
        self.num_heads = int(num_chans * expan_att_chans)
        self.t = nn.Parameter(torch.ones(1, self.num_heads, 1, 1))
        self.group_qkv = nn.Conv2d(embed_dim, embed_dim * expan_att_chans * 3, 1, groups=embed_dim)
        self.group_fus = nn.Conv2d(embed_dim * expan_att_chans, embed_dim, 1, groups=embed_dim)

    def forward(self, x):
        B, C, H, W = x.size()
        # 计算Q、K、V
        q, k, v = (
            self.group_qkv(x).view(B, C, self.expan_att_chans * 3, H, W).transpose(1, 2).contiguous().chunk(3, dim=1)
        )
        C_exp = self.expan_att_chans * C

        # 变换形状以进行自注意力计算
        q = q.view(B, self.num_heads, C_exp // self.num_heads, H * W)
        k = k.view(B, self.num_heads, C_exp // self.num_heads, H * W)
        v = v.view(B, self.num_heads, C_exp // self.num_heads, H * W)

        q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)
        attn = q @ k.transpose(-2, -1) * self.t  # 计算注意力权重

        x_ = attn.softmax(dim=-1) @ v  # 应用注意力权重
        x_ = x_.view(B, self.expan_att_chans, C, H, W).transpose(1, 2).flatten(1, 2).contiguous()

        x_ = self.group_fus(x_)  # 通过融合层
        return x_


# --------------------------- 空间注意力模块 ---------------------------#
class SpatialAttention(nn.Module):
    def __init__(self, embed_dim, num_chans, expan_att_chans):
        super().__init__()
        self.expan_att_chans = expan_att_chans
        self.num_heads = int(num_chans * expan_att_chans)
        self.t = nn.Parameter(torch.ones(1, self.num_heads, 1, 1))
        self.group_qkv = nn.Conv2d(embed_dim, embed_dim * expan_att_chans * 3, 1, groups=embed_dim)
        self.group_fus = nn.Conv2d(embed_dim * expan_att_chans, embed_dim, 1, groups=embed_dim)

    def forward(self, x):
        B, C, H, W = x.size()
        # 计算Q、K、V
        q, k, v = (
            self.group_qkv(x).view(B, C, self.expan_att_chans * 3, H, W).transpose(1, 2).contiguous().chunk(3, dim=1)
        )
        C_exp = self.expan_att_chans * C

        # 变换形状以进行自注意力计算
        q = q.view(B, self.num_heads, C_exp // self.num_heads, H * W)
        k = k.view(B, self.num_heads, C_exp // self.num_heads, H * W)
        v = v.view(B, self.num_heads, C_exp // self.num_heads, H * W)

        q, k = F.normalize(q, dim=-2), F.normalize(k, dim=-2)
        attn = q.transpose(-2, -1) @ k * self.t  # 计算注意力权重

        x_ = attn.softmax(dim=-1) @ v.transpose(-2, -1)  # 应用注意力权重
        x_ = x_.transpose(-2, -1).contiguous()

        x_ = x_.view(B, self.expan_att_chans, C, H, W).transpose(1, 2).flatten(1, 2).contiguous()

        x_ = self.group_fus(x_)
        return x_


# --------------------- Condensed Attention Neural Block ---------------------#
class CondensedAttentionNeuralBlock(nn.Module):
    def __init__(self, embed_dim, squeezes=(4, 2), shuffle=2, expan_att_chans=2):
        super().__init__()
        self.embed_dim = embed_dim

        sque_ch_dim = embed_dim // squeezes[0]  # 通道维度压缩
        shuf_sp_dim = int(sque_ch_dim * (shuffle**2))  # 空间维度变换
        sque_sp_dim = shuf_sp_dim // squeezes[1]

        self.sque_ch_dim = sque_ch_dim
        self.shuffle = shuffle
        self.shuf_sp_dim = shuf_sp_dim
        self.sque_sp_dim = sque_sp_dim

        # 先压缩通道，再调整空间
        self.ch_sp_squeeze = nn.Sequential(
            nn.Conv2d(embed_dim, sque_ch_dim, 1),
            nn.Conv2d(sque_ch_dim, sque_sp_dim, shuffle, shuffle, groups=sque_ch_dim),
        )

        self.channel_attention = ChannelAttention(sque_sp_dim, sque_ch_dim, expan_att_chans)
        self.spatial_attention = SpatialAttention(sque_sp_dim, sque_ch_dim, expan_att_chans)

        # 先恢复空间，再恢复通道
        self.sp_ch_unsqueeze = nn.Sequential(
            nn.Conv2d(sque_sp_dim, shuf_sp_dim, 1, groups=sque_ch_dim),
            nn.PixelShuffle(shuffle),
            nn.Conv2d(sque_ch_dim, embed_dim, 1),
        )

    def forward(self, x):
        _B, _C, H, W = x.size()

        # 直接判断H,W是否为单数
        pad_h = 1 if H % 2 == 1 else 0
        pad_w = 1 if W % 2 == 1 else 0

        # 执行填充
        x_padded = F.pad(x, (0, pad_w, pad_h, 0), mode="constant", value=0)
        _H_padded, _W_padded = H + pad_h, W + pad_w

        x = self.ch_sp_squeeze(x_padded)

        # 重新排列通道
        group_num = self.sque_ch_dim
        each_group = self.sque_sp_dim // self.sque_ch_dim
        idx = [i + j * group_num for i in range(group_num) for j in range(each_group)]
        x = x[:, idx, :, :]

        x = self.channel_attention(x)
        nidx = [i + j * each_group for i in range(each_group) for j in range(group_num)]
        x = x[:, nidx, :, :]

        x = self.spatial_attention(x)
        x = self.sp_ch_unsqueeze(x)
        return x[:, :, :H, :W]


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


class PSABloc_CAttention(nn.Module):
    """PSABlock class implementing a Position-Sensitive Attention block for neural networks.

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

        self.attn = CondensedAttentionNeuralBlock(c)
        self.ffn = nn.Sequential(Conv(c, c * 2, 1), Conv(c * 2, c, 1, act=False))
        self.add = shortcut

    def forward(self, x):
        """Executes a forward pass through PSABlock, applying attention and feed-forward layers to the input tensor."""
        x = x + self.attn(x) if self.add else self.attn(x)
        x = x + self.ffn(x) if self.add else self.ffn(x)
        return x


class C2PSA_CAttention(nn.Module):
    """C2PSA module with attention mechanism for enhanced feature extraction and processing.

    This module implements a convolutional block with attention mechanisms to enhance feature extraction and processing
    capabilities. It includes a series of PSABlock modules for self-attention and feed-forward operations.

    Attributes:
        c (int): Number of hidden channels.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c.
        m (nn.Sequential): Sequential container of PSABlock modules for attention and feed-forward operations.

    Methods:
        forward: Performs a forward pass through the C2PSA module, applying attention and feed-forward operations.

    Examples:
        >>> c2psa = C2PSA(c1=256, c2=256, n=3, e=0.5)
        >>> input_tensor = torch.randn(1, 256, 64, 64)
        >>> output_tensor = c2psa(input_tensor)

    Notes:
        This module essentially is the same as PSA module, but refactored to allow stacking more PSABlock modules.
    """

    def __init__(self, c1, c2, n=1, e=0.5):
        """Initializes the C2PSA module with specified input/output channels, number of layers, and expansion ratio."""
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)

        self.m = nn.Sequential(*(PSABloc_CAttention(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))

    def forward(self, x):
        """Processes the input tensor 'x' through a series of PSA blocks and returns the transformed tensor."""
        x = x.to(self.cv1.conv.weight.device)  # 确保输入张量在第一个卷积层所在设备
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = self.m(b)
        return self.cv2(torch.cat((a, b), 1))


# --------------------------- 主函数测试 ---------------------------#
if __name__ == "__main__":
    B, C, H, W = 2, 64, 31, 27  # 批量大小、通道数、高度、宽度
    x = torch.randn(B, C, H, W)  # 随机输入数据

    model = CondensedAttentionNeuralBlock(embed_dim=C)
    output = model(x)
    print("输入形状:", x.shape)
    print("输出形状:", output.shape)
