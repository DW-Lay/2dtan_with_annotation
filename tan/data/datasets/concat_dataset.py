# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import bisect

from torch.utils.data.dataset import ConcatDataset as _ConcatDataset


class ConcatDataset(_ConcatDataset):
    """
    Same as torch.utils.data.dataset.ConcatDataset, but exposes an extra
    method for querying the sizes of the image
    """
    # https://blog.csdn.net/cjf1699/article/details/105530628
    # 因为cumulative_size是递增的，所以它用了一个二分查找的包：bisect，
    # 找到第一个大于idx的索引dataset_idx。
    # 注意，idx是一个全局索引，意思是“我要拿到被Concat之后的这个大的数据集里面的第idx个样本”，
    # 所以具体实现的时候我们就需要知道要到第几个子数据集中去找第几个样本。
    # 而dataset_idx就是第几个子数据集，这也是后面几行代码的意思了。
    def __init__(self, datasets):
        super(ConcatDataset, self).__init__(datasets)

    def get_idxs(self, idx):
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return dataset_idx, sample_idx

    def get_img_info(self, idx):
        dataset_idx, sample_idx = self.get_idxs(idx)
        return self.datasets[dataset_idx].get_img_info(sample_idx)
