import torch
import torch.nn as nn

# 论文题目：Efficient Frequency-Domain Image Deraining with Contrastive Regularization
# 论文链接：https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/05751.pdf
# 官方github：https://github.com/deng-ai-lab/FADformer/tree/main
# 代码改进者：一勺汤


class FourierUnit(nn.Module):
    """Fourier Unit模块，用于在频域中进行特征变换。."""

    def __init__(self, in_channels, out_channels, groups=1):
        """初始化FourierUnit模块。.

        Args:
            in_channels (int): 输入通道数。
            out_channels (int): 输出通道数。
            groups (int, optional): 分组卷积的组数。默认为1。
        """
        super().__init__()
        self.groups = groups
        # 1x1卷积层，用于频域特征变换
        self.conv_layer = torch.nn.Conv2d(
            in_channels=in_channels * 2,
            out_channels=out_channels * 2,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=self.groups,
            bias=False,
        )
        # 批归一化层
        self.bn = torch.nn.BatchNorm2d(out_channels * 2)
        # ReLU激活函数
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        """前向传播函数。.

        Args:
            x (torch.Tensor): 输入张量，形状为(batch, channels, height, width)。

        Returns:
            torch.Tensor: 输出张量，形状与输入相同。
        """
        batch, _c, h, w = x.size()

        # 对输入进行2D快速傅里叶变换 (FFT)
        ffted = torch.fft.rfft2(x, norm="ortho")
        # 提取实部和虚部
        x_fft_real = torch.unsqueeze(torch.real(ffted), dim=-1)
        x_fft_imag = torch.unsqueeze(torch.imag(ffted), dim=-1)
        # 将实部和虚部拼接在一起
        ffted = torch.cat((x_fft_real, x_fft_imag), dim=-1)
        # 调整张量形状以便进行卷积操作
        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()
        ffted = ffted.view((batch, -1, *ffted.size()[3:]))

        # 对频域特征进行卷积操作
        ffted = self.conv_layer(ffted)  # (batch, c*2, h, w/2+1)
        ffted = self.relu(self.bn(ffted))

        # 恢复张量形状以便进行逆傅里叶变换
        ffted = (
            ffted.view((batch, -1, 2, *ffted.size()[2:])).permute(0, 1, 3, 4, 2).contiguous()
        )  # (batch,c, t, h, w/2+1, 2)
        ffted = torch.view_as_complex(ffted)

        # 对频域特征进行逆傅里叶变换，恢复到时域
        output = torch.fft.irfft2(ffted, s=(h, w), norm="ortho")

        return output


class Freq_Fusion(nn.Module):
    """频域融合模块，用于在频域中融合特征。."""

    def __init__(self, dim, kernel_size=[1, 3, 5, 7], se_ratio=4, local_size=8, scale_ratio=2, spilt_num=4):
        """初始化Freq_Fusion模块。.

        Args:
            dim (int): 输入特征的维度。
            kernel_size (list, optional): 卷积核大小列表。默认为[1,3,5,7]。
            se_ratio (int, optional): Squeeze-and-Excitation模块的缩减比例。默认为4。
            local_size (int, optional): 局部窗口大小。默认为8。
            scale_ratio (int, optional): 特征缩放比例。默认为2。
            spilt_num (int, optional): 特征分割数。默认为4。
        """
        super().__init__()
        self.dim = dim
        self.c_down_ratio = se_ratio
        self.size = local_size
        self.dim_sp = dim * scale_ratio // spilt_num
        # 初始化卷积层1 (Pointwise卷积)
        self.conv_init_1 = nn.Sequential(nn.Conv2d(dim, dim, 1), nn.GELU())
        # 初始化卷积层2 (Pointwise卷积)
        self.conv_init_2 = nn.Sequential(nn.Conv2d(dim, dim, 1), nn.GELU())
        # 中间卷积层
        self.conv_mid = nn.Sequential(nn.Conv2d(dim * 2, dim, 1), nn.GELU())
        # 傅里叶变换单元
        self.FFC = FourierUnit(self.dim * 2, self.dim * 2)

        # 批归一化层
        self.bn = torch.nn.BatchNorm2d(dim * 2)
        # ReLU激活函数
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        """前向传播函数。.

        Args:
            x (torch.Tensor): 输入张量，形状为(batch, channels, height, width)。

        Returns:
            torch.Tensor: 输出张量，形状与输入相同。
        """
        # 将输入特征分割为两部分
        x_1, x_2 = torch.split(x, self.dim, dim=1)
        # 分别对两部分特征进行卷积操作
        x_1 = self.conv_init_1(x_1)
        x_2 = self.conv_init_2(x_2)
        # 将两部分特征拼接在一起
        x0 = torch.cat([x_1, x_2], dim=1)
        # 通过傅里叶变换单元进行特征融合
        x = self.FFC(x0) + x0
        # 应用批归一化和ReLU激活函数
        x = self.relu(self.bn(x))

        return x


class Fused_Fourier_Conv_Mixer(nn.Module):
    """融合傅里叶卷积混合器模块，用于在时域和频域中进行特征融合。."""

    def __init__(self, dim, token_mixer_for_gloal=Freq_Fusion, mixer_kernel_size=[1, 3, 5, 7], local_size=8):
        """初始化Fused_Fourier_Conv_Mixer模块。.

        Args:
            dim (int): 输入特征的维度。
            token_mixer_for_gloal (nn.Module, optional): 全局特征混合器。默认为Freq_Fusion。
            mixer_kernel_size (list, optional): 卷积核大小列表。默认为[1,3,5,7]。
            local_size (int, optional): 局部窗口大小。默认为8。
        """
        super().__init__()
        self.dim = dim
        # 全局特征混合器
        self.mixer_gloal = token_mixer_for_gloal(
            dim=self.dim, kernel_size=mixer_kernel_size, se_ratio=8, local_size=local_size
        )

        # 通道注意力卷积层
        self.ca_conv = nn.Sequential(
            nn.Conv2d(2 * dim, dim, 1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, padding_mode="reflect"),
            nn.GELU(),
        )
        # 通道注意力模块
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 4, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(dim // 4, dim, kernel_size=1),
            nn.Sigmoid(),
        )
        # 初始化卷积层 (Pointwise卷积)
        self.conv_init = nn.Sequential(nn.Conv2d(dim, dim * 2, 1), nn.GELU())
        # 深度可分离卷积层1 (3x3卷积)
        self.dw_conv_1 = nn.Sequential(
            nn.Conv2d(self.dim, self.dim, kernel_size=3, padding=3 // 2, groups=self.dim, padding_mode="reflect"),
            nn.GELU(),
        )
        # 深度可分离卷积层2 (5x5卷积)
        self.dw_conv_2 = nn.Sequential(
            nn.Conv2d(self.dim, self.dim, kernel_size=5, padding=5 // 2, groups=self.dim, padding_mode="reflect"),
            nn.GELU(),
        )

    def forward(self, x):
        """前向传播函数。.

        Args:
            x (torch.Tensor): 输入张量，形状为(batch, channels, height, width)。

        Returns:
            torch.Tensor: 输出张量，形状与输入相同。
        """
        # 对输入特征进行初始化卷积操作
        x = self.conv_init(x)
        # 将特征分割为两部分
        x = list(torch.split(x, self.dim, dim=1))
        # 对第一部分特征进行深度可分离卷积操作
        x_local_1 = self.dw_conv_1(x[0])
        # 对第二部分特征进行深度可分离卷积操作
        x_local_2 = self.dw_conv_2(x[0])
        # 将两部分特征拼接并通过全局特征混合器进行融合
        x_gloal = self.mixer_gloal(torch.cat([x_local_1, x_local_2], dim=1))
        # 对融合后的特征进行通道注意力卷积操作
        x = self.ca_conv(x_gloal)
        # 应用通道注意力机制
        x = self.ca(x) * x

        return x


def main():

    model = Fused_Fourier_Conv_Mixer(dim=64)
    x = torch.randn(1, 64, 128, 128)  # Batch size=1, 64通道, 128x128 图像
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")


if __name__ == "__main__":
    main()
