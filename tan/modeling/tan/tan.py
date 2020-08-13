import torch
from torch import nn

from .featpool import build_featpool
from .feat2d import build_feat2d
from .integrator import build_integrator
from .predictor import build_predictor
from .loss import build_tanloss

class TAN(nn.Module):
    def __init__(self, cfg):
        # 先初始化继承类
        super(TAN, self).__init__()
        self.featpool = build_featpool(cfg) 
        self.feat2d = build_feat2d(cfg)
        self.integrator = build_integrator(cfg)
        self.predictor = build_predictor(cfg, self.feat2d.mask2d)
        self.tanloss = build_tanloss(cfg, self.feat2d.mask2d)
    
    def forward(self, batches, ious2d=None):
        """
        Arguments:

        Returns:

        """
        # 先生成特征向量 按照batch的形式
        feats = self.featpool(batches.feats)
        # 将视频生成的特征向量生成二维形式
        map2d = self.feat2d(feats)
        # 将二维视频特征和文本特征向量结合
        map2d = self.integrator(batches.queries, batches.wordlens, map2d)
        # 计算二维得分
        scores2d = self.predictor(map2d)
        # print(self.training) 
        if self.training:
            return self.tanloss(scores2d, ious2d)
        return scores2d.sigmoid_() * self.feat2d.mask2d
