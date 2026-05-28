import torch.nn as nn
import torch
from pytorch_wavelets import DWTForward, DWTInverse


#

# 论文题目：ASCNet: Asymmetric Sampling Correction Network for Infrared Image Destriping TGRS 2025
# 论文链接： https://ieeexplore.ieee.org/document/10855453
# 代码改进者：一勺汤

# 定义一个双卷积模块，包含两个卷积层和LeakyReLU激活函数
class double_conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        # 调用父类的构造函数
        super(double_conv, self).__init__()
        # 定义一个顺序容器，包含两个卷积层和LeakyReLU激活函数
        self.d_conv = nn.Sequential(
            # 第一个卷积层，输入通道数为in_channels，输出通道数为out_channels，卷积核大小为3，填充为1
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            # LeakyReLU激活函数
            nn.LeakyReLU(inplace=True),
            # 第二个卷积层，输入和输出通道数均为out_channels，卷积核大小为3，填充为1
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            # LeakyReLU激活函数
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        # 前向传播，将输入x通过双卷积模块
        x = self.d_conv(x)
        return x


# 定义一个单卷积模块，包含一个卷积层和LeakyReLU激活函数
class single_conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        # 调用父类的构造函数
        super(single_conv, self).__init__()
        # 定义一个顺序容器，包含一个卷积层和LeakyReLU激活函数
        self.s_conv = nn.Sequential(
            # 卷积层，输入通道数为in_channels，输出通道数为out_channels，卷积核大小为3，填充为1
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            # LeakyReLU激活函数
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        # 前向传播，将输入x通过单卷积模块
        x = self.s_conv(x)
        return x


# 定义ASCNet模型
class RHDWT_Block(nn.Module):
    def __init__(self, in_ch, out_ch, feats):
        # 调用父类的构造函数
        super(RHDWT_Block, self).__init__()
        # 用于存储特征的列表
        self.features = []
        # 模型的头部，使用单卷积模块将输入通道数转换为feats
        self.head = single_conv(in_ch, feats)
        # 第一个双卷积模块，用于特征编码
        self.dconv_encode0 = double_conv(feats, feats)
        # 恒等映射卷积层，用于下采样，将通道数翻倍
        self.identety1 = nn.Conv2d(in_channels=feats, out_channels=out_ch, kernel_size=3, stride=2, padding=1)
        # 离散小波变换层，使用Haar小波进行一级分解
        self.DWT = DWTForward(J=1, wave='haar')
        # 第二个单卷积模块，用于特征编码
        self.dconv_encode1 = single_conv(4 * feats, out_ch)

    def _transformer(self, DMT1_yl, DMT1_yh):
        # 存储张量的列表
        list_tensor = []
        # 获取高频分量
        a = DMT1_yh[0]
        # 将低频分量添加到列表中
        list_tensor.append(DMT1_yl)
        # 遍历三个高频分量并添加到列表中
        for i in range(3):
            list_tensor.append(a[:, :, i, :, :])
        # 在通道维度上拼接张量
        return torch.cat(list_tensor, 1)

    def forward(self, x):

        # 通过模型头部和第一个双卷积模块进行特征提取
        x0 = self.dconv_encode0(self.head(x))
        # 对特征图进行离散小波变换，得到低频分量和高频分量
        DMT1_yl, DMT1_yh = self.DWT(x0)
        # 对小波变换的结果进行处理
        DMT1 = self._transformer(DMT1_yl, DMT1_yh)
        # 通过第二个单卷积模块进行特征编码
        x = self.dconv_encode1(DMT1)
        # 通过恒等映射卷积层进行下采样
        res1 = self.identety1(x0)
        # 将编码后的特征图和下采样后的特征图相加
        out = torch.add(x, res1)
        # 这里原代码调用了self.enhance1，但未定义，暂时注释掉
        # x1 = self.enhance1(out)
        return out



if __name__ == "__main__":
    # 定义输入通道数、输出通道数和特征通道数
    in_ch = 64
    out_ch = 128
    feats = 64
    # 创建ASCNet模型实例
    model = RHDWT_Block(in_ch, out_ch, feats)
    # 生成一个随机输入张量，模拟输入图像
    input_tensor = torch.randn(1, in_ch, 256, 256)
    # 前向传播，得到输出
    output = model(input_tensor)
    # 打印输出的形状
    print("Output shape:", output.shape)

