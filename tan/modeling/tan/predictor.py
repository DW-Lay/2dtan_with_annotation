import torch
from torch import nn

def mask2weight(mask2d, mask_kernel, padding=0):
    # from the feat2d.py,we can know the mask2d is 4-d
    # torch.conv2d 是直接卷积出结果，不同于torch.nn.Conv2d()
    weight = torch.conv2d(mask2d[None,None,:,:].float(),
        mask_kernel, padding=padding)[0, 0]  # [0,0]是因为是(1,1,x,x)4维的数据，所以要拿出里面的来
    weight[weight > 0] = 1 / weight[weight > 0]
    return weight

class Predictor(nn.Module):
    def __init__(self, input_size, hidden_size, k, num_stack_layers, mask2d): 
        super(Predictor, self).__init__()
        
        # Padding to ensure the dimension of the output map2d
        mask_kernel = torch.ones(1,1,k,k).to(mask2d.device) 
        # 明确卷积的计算公式：d_out = (d_in - kennel_size + 2 * padding) / stride + 1
        # 最后*num_stack_layers是为了保证在所有叠层卷积之后，尺寸仍然不变
        first_padding = (k - 1) * num_stack_layers // 2

        self.weights = [
            mask2weight(mask2d, mask_kernel, padding=first_padding) 
        ]
        self.convs = nn.ModuleList(
            [nn.Conv2d(input_size, hidden_size, k, padding=first_padding)]
        )  
 
        for _ in range(num_stack_layers - 1):
            # padding都是默认为0了
            # self.weights[-1] > 0 又重新变为0或1
            self.weights.append(mask2weight(self.weights[-1] > 0, mask_kernel))
            self.convs.append(nn.Conv2d(hidden_size, hidden_size, k))
        # 输入通道  hidden_size   输出通道和kernel_size均为1 默认padding=0
        # pred 应该为 32*1*64*64
        self.pred = nn.Conv2d(hidden_size, 1, 1)
    # https://www.cnblogs.com/llfctt/p/10967651.html  解释了forward方法的使用
    # 也就是说，当把定义的网络模型model当作函数调用的时候就自动调用定义的网络模型的forward方法。
    def forward(self, x):
        for conv, weight in zip(self.convs, self.weights):
            x = conv(x).relu() * weight
        # 去掉维度为1的从[4,1,64,64] 到 [4,64,64]
        x = self.pred(x).squeeze_()
        return x
        
# mask2d就是在生成map2d的过程中，标记了哪些位置有数，置为1  （64*64）
def build_predictor(cfg, mask2d):
    input_size = cfg.MODEL.TAN.FEATPOOL.HIDDEN_SIZE
    hidden_size = cfg.MODEL.TAN.PREDICTOR.HIDDEN_SIZE
    kernel_size = cfg.MODEL.TAN.PREDICTOR.KERNEL_SIZE
    num_stack_layers = cfg.MODEL.TAN.PREDICTOR.NUM_STACK_LAYERS
    return Predictor(
        input_size, hidden_size, kernel_size, num_stack_layers, mask2d
    ) 
