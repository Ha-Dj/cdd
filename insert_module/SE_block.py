import torch
from torch import nn
from torch.nn.parameter import Parameter

'''-------------一、SE模块-----------------------------'''


# 全局平均池化+1*1卷积核+ReLu+1*1卷积核+Sigmoid
class SE_Block(nn.Module):
    def __init__(self, inchannel, ratio=16):
        super(SE_Block, self).__init__()
        # 全局平均池化(Fsq操作)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        # 两个全连接层(Fex操作)
        self.fc = nn.Sequential(
            nn.Linear(inchannel, inchannel // ratio, bias=False),  # 从 c -> c/r
            nn.ReLU(),
            nn.Linear(inchannel // ratio, inchannel, bias=False),  # 从 c/r -> c
            nn.Sigmoid()
        )

    def forward(self, x):
        # 读取批数据图片数量及通道数
        b, c, h, w = x.size()
        # Fsq操作：经池化后输出b*c的矩阵
        y = self.gap(x).view(b, c)
        # Fex操作：经全连接层输出（b，c，1，1）矩阵
        y = self.fc(y).view(b, c, 1, 1)
        # Fscale操作：将得到的权重乘以原来的特征图x
        return x * y.expand_as(x)


class fusion_SE_block(nn.Module):
    def __init__(self, in_channel, ratio=16):
        super(fusion_SE_block, self).__init__()
        self.gap_private_1 = nn.AdaptiveAvgPool2d((1, 1))
        self.gap_private_2 = nn.AdaptiveAvgPool2d((1, 1))
        self.gap_share = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Sequential(
            nn.Linear(in_channel, in_channel // ratio, bias=False),  # 从 c -> c/r
            nn.ReLU(),
            nn.Linear(in_channel // ratio, in_channel, bias=False),  # 从 c/r -> c
            nn.Sigmoid()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(in_channel, in_channel // ratio, bias=False),  # 从 c -> c/r
            nn.ReLU(),
            nn.Linear(in_channel // ratio, in_channel, bias=False),  # 从 c/r -> c
            nn.Sigmoid()
        )
        self.fc3 = nn.Sequential(
            nn.Linear(in_channel, in_channel // ratio, bias=False),  # 从 c -> c/r
            nn.ReLU(),
            nn.Linear(in_channel // ratio, in_channel, bias=False),  # 从 c/r -> c
            nn.Sigmoid()
        )
    def forwardX(self, x, gap, fc):
        # 读取批数据图片数量及通道数
        b, c, h, w = x.size()
        # Fsq操作：经池化后输出b*c的矩阵
        y = gap(x).view(b, c)
        # Fex操作：经全连接层输出（b，c，1，1）矩阵
        y = fc(y).view(b, c, 1, 1)
        # Fscale操作：将得到的权重乘以原来的特征图x
        return x * y.expand_as(x)

    def forward(self, x1, x2):
        y1 = self.forwardX(x1, self.gap_private_1, self.fc1)
        y2 = self.forwardX(x2, self.gap_private_2, self.fc2)
        y3 = self.forwardX(x1 + x2, self.gap_share, self.fc3)
        return y1 + y3, y2 + y3



# input = torch.randn(1, 64, 32, 32)
# model_test = fusion_SE_block(64)
#
# output1, output2 = model_test(input, input)
#
# print(output1.shape)
# print(output2.shape)