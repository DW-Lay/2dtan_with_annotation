import logging
import time
import os
from tqdm import tqdm

import torch

from tan.data.datasets.evaluation import evaluate
from ..utils.comm import is_main_process, get_world_size
from ..utils.comm import all_gather
from ..utils.comm import synchronize
from ..utils.timer import Timer, get_time_str

def compute_on_dataset(model, data_loader, device, timer=None):
    model.eval()
    results_dict = {}
    cpu_device = torch.device("cpu")
    # tqdm是一个进度条显示的
    for batch in tqdm(data_loader):
        # batches,targets,idxs的个数均为batch_size个
        # batches里有feat(256*500(activitynet)或（256*4096(tacos)),query(wordlen*300),wordlen(query.size(0))
        # targets 里为batch_size个真实的iou2d
        # idxs为batch_size个标号（每个小片段一个idx)一共17031个(针对actiitynet)
        batches, targets, idxs = batch
        # 以下不需要计算梯度，也不需要反向传播
        with torch.no_grad():
            if timer:
                timer.tic()
            # output 就是scores2d.sigmoid_() * self.feat2d.mask2d  (batch_size*64*64)或者说(batch_size*num_clips*num_clips)
            output = model(batches.to(device))
            if timer:
                if not device.type == 'cpu':
                    torch.cuda.synchronize()
                timer.toc()
            # [(num_clips*num_clips)]
            output = [o.to(cpu_device) for o in output]
        results_dict.update(
            {img_id: result for img_id, result in zip(idxs, output)}
        )
    # results_dict (id 1:result(4*64*64))
    return results_dict

def _accumulate_predictions_from_multiple_gpus(predictions_per_gpu):
    # 输出的all_predictions是一个list,每个元素为dict
    all_predictions = all_gather(predictions_per_gpu)
    if not is_main_process():
        return
    # merge the list of dicts
    predictions = {}
    for p in all_predictions:
        predictions.update(p)
    # convert a dict where the key is the index in a list
    idxs = list(sorted(predictions.keys()))
    if len(idxs) != idxs[-1] + 1:
        logger = logging.getLogger("tan.inference")
        logger.warning(
            "Number of samples that were gathered from multiple processes is not "
            "a contiguous set. Some samples might be missing from the evaluation"
        )

    # convert to a list
    predictions = [predictions[i] for i in idxs]
    return predictions

def inference(
        model,
        data_loader,
        dataset_name,
        nms_thresh,
        device="cuda",
):
    # convert to a torch.device for efficiency
    device = torch.device(device)
    num_devices = get_world_size()
    logger = logging.getLogger("tan.inference")
    dataset = data_loader.dataset
    logger.info("Start evaluation on {} dataset (Size: {}).".format(dataset_name, len(dataset)))
    total_timer = Timer()
    inference_timer = Timer()
    # tic 表示计时开始   toc表示计时结束
    total_timer.tic()
    # 返回一个dict  {id:predict_scores}
    predictions = compute_on_dataset(model, data_loader, device, inference_timer)
    # wait for all processes to complete before measuring the time
    synchronize()
    total_time = total_timer.toc()
    total_time_str = get_time_str(total_time)
    logger.info(
        "Total run time: {} ({} s / inference per device, on {} devices)".format(
            total_time_str, total_time * num_devices / len(dataset), num_devices
        )
    )
    total_infer_time = get_time_str(inference_timer.total_time)
    logger.info(
        "Model inference time: {} ({} s / inference per device, on {} devices)".format(
            total_infer_time,
            inference_timer.total_time * num_devices / len(dataset),
            num_devices,
        )
    )

    predictions = _accumulate_predictions_from_multiple_gpus(predictions)
    if not is_main_process():
        return
    
    return evaluate(dataset=dataset, predictions=predictions, nms_thresh=nms_thresh)
