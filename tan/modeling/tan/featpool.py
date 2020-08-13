import torch
from torch import nn

class FeatAvgPool(nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size, stride):
        super(FeatAvgPool, self).__init__()
        # class torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
        # in_channels(int) – 输入信号的通道。在文本分类中，即为词向量的维度
        # out_channels(int) – 卷积产生的通道。有多少个out_channels，就需要多少个1维卷积
        # kernel_size(int or tuple) - 卷积核的尺寸，卷积核的大小为(k,)，第二个维度是由in_channels来决定的，所以实际上卷积大小为kernel_size*in_channels
        # stride(int or tuple, optional) - 卷积步长
        # padding (int or tuple, optional)- 输入的每一条边补充0的层数
        # dilation(int or tuple, `optional``) – 卷积核元素之间的间距
        # groups(int, optional) – 从输入通道到输出通道的阻塞连接数
        # bias(bool, optional) - 如果bias=True，添加偏置
        self.conv = nn.Conv1d(input_size, hidden_size, 1, 1)
        self.pool = nn.AvgPool1d(kernel_size, stride)

    def forward(self, x):
        # X轴用0表示，Y轴用1表示；Z轴用2来表示；
        # x batch_size * pre_num_clips * input_size     64*256*4096
        return self.pool(self.conv(x.transpose(1, 2)).relu())

def build_featpool(cfg):
    input_size = cfg.MODEL.TAN.FEATPOOL.INPUT_SIZE
    hidden_size = cfg.MODEL.TAN.FEATPOOL.HIDDEN_SIZE
    kernel_size = cfg.MODEL.TAN.FEATPOOL.KERNEL_SIZE
    stride = cfg.INPUT.NUM_PRE_CLIPS // cfg.MODEL.TAN.NUM_CLIPS
    # 等价于直接调用forward()函数
    return FeatAvgPool(input_size, hidden_size, kernel_size, stride)
