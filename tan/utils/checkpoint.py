# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import logging
import os

import torch

from tan.utils.model_serialization import load_state_dict
from tan.utils.imports import import_file

class Checkpointer(object):
    def __init__(
        self,
        model,
        optimizer=None,
        scheduler=None,
        save_dir="",
        save_to_disk=None,
        logger=None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.save_dir = save_dir
        self.save_to_disk = save_to_disk
        # 如果没有log,则新建一个当前name的log,通常输入的都是已经存在的
        if logger is None:
            logger = logging.getLogger(__name__)
        self.logger = logger

    def save(self, name, **kwargs):
        # 如果没有保存为true或者没有保存路径，就不保存
        if not self.save_dir:
            return

        if not self.save_to_disk:
            return

        data = {}
        # state_dict()本质上仍是一个字典对象，存放的是训练过程中的权重和系数，将每一层的参数映射为tensor张量，只包含卷积层和全连接层的参数
        data["model"] = self.model.state_dict()
        if self.optimizer is not None:
            data["optimizer"] = self.optimizer.state_dict()
        if self.scheduler is not None:
            data["scheduler"] = self.scheduler.state_dict()
        data.update(kwargs)

        save_file = os.path.join(self.save_dir, "{}.pth".format(name))
        self.logger.info("Saving checkpoint to {}".format(save_file))
        torch.save(data, save_file)
        self.tag_last_checkpoint(save_file)

    def load(self, f=None, use_latest=True):
        if self.has_checkpoint() and use_latest:
            # override argument with existing checkpoint
            # 拿出来的类似 output/2dtan_64x64_pool_k914-activitynet/model_5e.pth
            f = self.get_checkpoint_file()
        if not f:
            # no checkpoint could be found
            self.logger.info("No checkpoint found. Initializing model from scratch")
            return {}
        self.logger.info("Loading checkpoint from {}".format(f))
        # 加载节点文件
        checkpoint = self._load_file(f)
        # 加载模型
        self._load_model(checkpoint)
        # 将参数节点模型中的优化器和学习率调整更新到当前模型中
        if "optimizer" in checkpoint and self.optimizer:
            self.logger.info("Loading optimizer from {}".format(f))
            # 将参数转移到gpu上
            self.optimizer.load_state_dict(checkpoint.pop("optimizer"))
        if "scheduler" in checkpoint and self.scheduler:
            self.logger.info("Loading scheduler from {}".format(f))
            self.scheduler.load_state_dict(checkpoint.pop("scheduler"))

        # return any further checkpoint data
        return checkpoint

    def has_checkpoint(self):
        # 检查是否有后缀为last_checkpoint的文件
        save_file = os.path.join(self.save_dir, "last_checkpoint")
        return os.path.exists(save_file)

    def get_checkpoint_file(self):
        save_file = os.path.join(self.save_dir, "last_checkpoint")
        try:
            with open(save_file, "r") as f:
                last_saved = f.read()
                # 删除头尾的空格   
                last_saved = last_saved.strip()
        except IOError:
            # if file doesn't exist, maybe because it has just been
            # deleted by a separate process
            last_saved = ""
        return last_saved

    def tag_last_checkpoint(self, last_filename):
        save_file = os.path.join(self.save_dir, "last_checkpoint")
        with open(save_file, "w") as f:
            f.write(last_filename)

    def _load_file(self, f):
        return torch.load(f, map_location=torch.device("cpu"))

    def _load_model(self, checkpoint):
        load_state_dict(self.model, checkpoint.pop("model"))


class TanCheckpointer(Checkpointer):
    def __init__(
        self,
        cfg,
        model,
        optimizer=None,
        scheduler=None,
        save_dir="",
        save_to_disk=None,
        logger=None,
    ):
        super(TanCheckpointer, self).__init__(
            model, optimizer, scheduler, save_dir, save_to_disk, logger
        )
        self.cfg = cfg.clone()

    def _load_file(self, f):
        loaded = super(TanCheckpointer, self)._load_file(f)
        if "model" not in loaded:
            loaded = dict(model=loaded)
        return loaded
