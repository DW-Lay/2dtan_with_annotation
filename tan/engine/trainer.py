import datetime
import logging
import os
import time

import torch
import torch.distributed as dist

from tan.data import make_data_loader
from tan.utils.comm import get_world_size, synchronize
from tan.utils.metric_logger import MetricLogger
from tan.engine.inference import inference

def reduce_loss(loss):
    world_size = get_world_size()
    if world_size < 2:
        return loss
    with torch.no_grad():
        dist.reduce(loss, dst=0)
        if dist.get_rank() == 0:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            loss /= world_size
    return loss

def do_train(
    cfg,
    model,
    data_loader,
    data_loader_val,
    optimizer,
    scheduler,
    checkpointer,
    device,
    checkpoint_period,
    test_period,
    arguments,
):
    logger = logging.getLogger("tan.trainer")
    logger.info("Start training")
    meters = MetricLogger(delimiter="  ")
    max_epoch = cfg.SOLVER.MAX_EPOCH
    # ，需要在训练时添加model.train()，在测试时添加model.eval()。
    # 其中model.train()是保证BN层用每一批数据的均值和方差，
    # 而model.eval()是保证BN用全部训练数据的均值和方差；
    # 而对于Dropout，model.train()是随机取一部分网络连接来训练更新参数，
    # 而model.eval()是利用到了所有网络连接。
    model.train()
    start_training_time = time.time()
    end = time.time()
    # arguments["epoch"] =1
    for epoch in range(arguments["epoch"], max_epoch + 1):
        max_iteration = len(data_loader)
        last_epoch_iteration = (max_epoch - epoch) * max_iteration
        arguments["epoch"] = epoch

        for iteration, (batches, targets, _) in enumerate(data_loader):
            iteration += 1
            data_time = time.time() - end
            # 下面这行代码的意思是将所有最开始读取数据时的tensor变量copy一份到device所指定的GPU上去，之后的运算都在GPU上进行。
            batches = batches.to(device)
            targets = targets.to(device)
            
            def closure():
                # optimizer.zero_grad()意思是把梯度置零，也就是把loss关于weight的导数变成0.
                optimizer.zero_grad()
                loss = model(batches, targets)
                if iteration % 20 == 0 or iteration == max_iteration:
                    meters.update(loss=reduce_loss(loss.detach()))
                loss.backward()
                return loss

            optimizer.step(closure)

            batch_time = time.time() - end
            end = time.time()
            meters.update(time=batch_time, data=data_time)

            eta_seconds = meters.time.global_avg * (max_iteration - iteration + last_epoch_iteration)
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

            if iteration % 20 == 0 or iteration == max_iteration:
                logger.info(
                    meters.delimiter.join(
                        [
                            "eta: {eta}",
                            "epoch: {epoch}/{max_epoch}",
                            "iteration: {iteration}/{max_iteration}",
                            "{meters}",
                            "lr: {lr:.6f}",
                            "max mem: {memory:.0f}",
                        ]
                    ).format(
                        eta=eta_string,
                        epoch=epoch,
                        max_epoch=max_epoch,
                        iteration=iteration,
                        max_iteration=max_iteration,
                        meters=str(meters),
                        lr=optimizer.param_groups[0]["lr"],
                        memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                    )
                )

        scheduler.step()

        if epoch % checkpoint_period == 0:
            checkpointer.save(f"model_{epoch}e", **arguments)
            
        if data_loader_val is not None and test_period > 0 and \
            epoch % test_period == 0:
            synchronize()
            inference(
                model,
                data_loader_val,
                dataset_name=cfg.DATASETS.TEST,
                nms_thresh=cfg.TEST.NMS_THRESH,
                device=cfg.MODEL.DEVICE,
            )
            synchronize()
            model.train()

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (max_iteration)
        )
    )
