from terminaltables import AsciiTable
from tqdm import tqdm
import logging

import torch

from tan.data import datasets
from tan.data.datasets.utils import iou, score2d_to_moments_scores
# 非极大值抑制
def nms(moments, scores, topk, thresh):
    # moments 64*64个二维数组，每个都表示所有点的开始和结束时间
    # 按照降序进行分数排名，同时rank为序号
    # scores 64*64个分数  ranks 64*64个顺序号
    scores, ranks = scores.sort(descending=True)
    # 将二维图上的片段也进行排序
    moments = moments[ranks]
    # 初始全部为false
    suppressed = ranks.zero_().bool()
    # suppressd 为 [1104]
    numel = suppressed.numel()
    for i in range(numel - 1):
        if suppressed[i]:
            continue
        # mask为 [[3]]
        mask = iou(moments[i+1:], moments[i]) > thresh
        # suppressd[i+1:]为后面所有的  后面的[]指的是后面的哪个
        suppressed[i+1:][mask] = True
    # ~ 为取反  总之最终的目的是为了将非常相近的去掉
    return moments[~suppressed]

def evaluate(dataset, predictions, nms_thresh, recall_metrics=(1,5), iou_metrics=(0.1,0.3,0.5,0.7)):
    """evaluate dataset using different methods based on dataset type.
    Args:
    Returns:
    """
    dataset_name = dataset.__class__.__name__
    logger = logging.getLogger("tan.inference")
    logger.info("Performing {} evaluation (Size: {}).".format(dataset_name, len(dataset)))
    
    num_recall_metrics, num_iou_metrics = len(recall_metrics), len(iou_metrics)
    table = [['Rank@{},mIoU@{}'.format(i,j) \
        for i in recall_metrics for j in iou_metrics]]
    
    recall_metrics = torch.tensor(recall_metrics)
    iou_metrics = torch.tensor(iou_metrics)
    recall_x_iou = torch.zeros(num_recall_metrics, num_iou_metrics)

    num_clips = predictions[0].shape[-1]
    for idx, score2d in tqdm(enumerate(predictions)):  
        duration = dataset.get_duration(idx)
        moment = dataset.get_moment(idx) 
        # candidates为每个非零位置的时间跨度，scores为非零位置的每个位置的分数
        candidates, scores = score2d_to_moments_scores(score2d, num_clips, duration)
        # recall_metrics[-1]  recall_metric的最后一个元素
        moments = nms(candidates, scores, topk=recall_metrics[-1], thresh=nms_thresh)

        for i, r in enumerate(recall_metrics):
            # r为前几个  i是index  mious 为得分
            mious = iou(moments[:r], dataset.get_moment(idx))
            # 以下的与代码无关
            # ==============================================================================
            # list.append(arg1) 参数类型任意，可以往已有列表中添加元素，若添加的是列表，
            # 就该列表被当成一个元素存在原列表中，只使list长度增加1.
            # list.extend(list1) 参数必须是列表类型，可以将参数中的列表合并到原列表的末尾，
            # 使原来的 list长度增加len(list1)。
            # =============================================================================

            # 将mious扩充到(r,num_iou_metrics)
            bools = mious[:,None].expand(r, num_iou_metrics) > iou_metrics
            # bools.any(dim=0) 按照行，这几行里只要有一个为true,则为true
            recall_x_iou[i] += bools.any(dim=0)

    recall_x_iou /= len(predictions)

    table.append(['{:.02f}'.format(recall_x_iou[i][j]*100) \
        for i in range(num_recall_metrics) for j in range(num_iou_metrics)])
    table = AsciiTable(table)
    for i in range(num_recall_metrics*num_iou_metrics):
        table.justify_columns[i] = 'center'
    logger.info('\n' + table.table)
