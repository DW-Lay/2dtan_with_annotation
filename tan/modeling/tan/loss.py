import torch
from torch.functional import F 

class TanLoss(object):
    def __init__(self, min_iou, max_iou, mask2d):
        self.min_iou, self.max_iou = min_iou, max_iou
        self.mask2d = mask2d

    def scale(self, iou):
        return (iou - self.min_iou) / (self.max_iou - self.min_iou)

    # __call__ 将一个类实例变成一个可调用对象
    def __call__(self, scores2d, ious2d):
        #  clamp()的参数
        # input (Tensor) – 输入张量
        # min (Number) – 限制范围下限
        # max (Number) – 限制范围上限
        # out (Tensor, optional) – 输出张量
        # 下面语句的作用也就是将iou之外的置为0/1
        ious2d = self.scale(ious2d).clamp(0, 1) 
        # binary_cross_entropy_with_logits
        # 接受任意形状的输入，target要求与输入形状一致。切记：target的值必须在[0,N-1]之间，
        # 其中N为类别数，否则会出现莫名其妙的错误，比如loss为负数。
        # 计算其实就是交叉熵，不过输入不要求在0，1之间，该函数会自动添加sigmoid运算
        # 默认的reduction方式为mean
        return F.binary_cross_entropy_with_logits(
            # mask_select会将满足mask（掩码、遮罩等等，随便翻译）的指示，将满足条件的点选出来
            # 根据掩码张量mask中的二元值，取输入张量中的指定项( mask为一个 ByteTensor)，将取值返回到一个新的1D张量，
            # 张量 mask须跟input张量有相同数量的元素数目，但形状或维度不需要相同。
            # 注意： 返回的张量不与原始张量共享内存空间。
            # ！！！！ 输出的为一维向量
            scores2d.masked_select(self.mask2d), 
            ious2d.masked_select(self.mask2d)
        )
        
def build_tanloss(cfg, mask2d):
    # 小于MIN_IOU直接为0
    min_iou = cfg.MODEL.TAN.LOSS.MIN_IOU 
    # 大于MAX_IOU直接为1
    max_iou = cfg.MODEL.TAN.LOSS.MAX_IOU
    return TanLoss(min_iou, max_iou, mask2d) 
