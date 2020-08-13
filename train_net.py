import argparse
import os

import torch
from torch import optim
from torch import multiprocessing
multiprocessing.set_sharing_strategy('file_system')

from tan.config import cfg
from tan.data import make_data_loader
from tan.engine.inference import inference
from tan.engine.trainer import do_train
from tan.modeling import build_model
from tan.utils.checkpoint import TanCheckpointer
from tan.utils.comm import synchronize, get_rank
from tan.utils.imports import import_file
from tan.utils.logger import setup_logger
from tan.utils.miscellaneous import mkdir, save_config


# 训练输入的模型  配置文件，当前使用的gpu号，以及是否是分布式训练
def train(cfg, local_rank, distributed):
    # 获取模型  通过配置文件就获得整个模型
    model = build_model(cfg)
    device = torch.device(cfg.MODEL.DEVICE)
    # 将模型加载到到cuda硬件中去
    model.to(device)
    # 设置优化器
    optimizer = optim.Adam(model.parameters(), lr=cfg.SOLVER.LR)
    # 设置动态调整学习率
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg.SOLVER.MILESTONES)
    # 开启分布式训练
    if distributed:
        print("local_rank=",local_rank)
        # 创建分布式并行模型
        model = torch.nn.parallel.DistributedDataParallel(            
            model, device_ids=[local_rank], output_device=local_rank,
            # this should be removed if we update BatchNorm stats
            broadcast_buffers=False,
        )
    # 取出保存文件的路径
    output_dir = cfg.OUTPUT_DIR
    # 只有0号进程才去加载模型，否则开4个进程都会加载
    save_to_disk = get_rank() == 0
    # 加载模型
    checkpointer = TanCheckpointer(
        cfg, model, optimizer, scheduler, output_dir, save_to_disk
    )

    if cfg.MODEL.WEIGHT != "":
        extra_checkpoint_data = checkpointer.load(f=None, use_latest=True)
    else:
        # 首次为"" 所以直接训练，不提取上次的训练模型
        extra_checkpoint_data = checkpointer.load(f=cfg.MODEL.WEIGHT, use_latest=False)
    
    arguments = {"epoch": 1}
    # 把extra_checkpoint_data这个字典里的值更新到arguments里
    arguments.update(extra_checkpoint_data)
    # 加载数据
    # dataloader是batch打包以后的数据，打包前是datasets,包含训练集和验证集或者测试集的文本数据和视频特征数据
    data_loader = make_data_loader(
        cfg,
        is_train=True,
        is_distributed=distributed,
    )

    # 每迭代几次进行一次
    test_period = cfg.SOLVER.TEST_PERIOD
    if test_period > 0:
        data_loader_val = make_data_loader(cfg, is_train=False, is_distributed=distributed, is_for_period=True)
    else:
        data_loader_val = None

    # 每迭代几次进行一次模型的保存
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD

    
    do_train(
        # 配置
        cfg,
        # 模型
        model,
        # 数据集下载
        data_loader,
        # 验证集数据下载 
        data_loader_val,
        # 优化器
        optimizer,
        # 动态调整优化器
        scheduler,
        # 节点模型
        checkpointer,
        # CUDA
        device,
        # 周期
        checkpoint_period,
        test_period,
        # 更新的参数
        arguments,
    )

    return model

def run_test(cfg, model, distributed):
    if distributed:
        model = model.module
    # 释放显存
    torch.cuda.empty_cache()  # TODO check if it helps
    dataset_names = cfg.DATASETS.TEST
    data_loaders_val = make_data_loader(cfg, is_train=False, is_distributed=distributed)
    for dataset_name, data_loader_val in zip(dataset_names, data_loaders_val):
        inference(
            model,
            data_loader_val,
            dataset_name=dataset_name,
            nms_thresh=cfg.TEST.NMS_THRESH,
            device=cfg.MODEL.DEVICE,
        )
        # 多线程锁
        synchronize()

def main():
    # 创建解析器
    parser = argparse.ArgumentParser(description="Tan")
    # 添加参数
    parser.add_argument(
        "--config-file",
        default="configs/2dtan_128x128_pool_k5l8_tacos.yaml",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    # local_rank代表当前进程使用的GPU编号 ，从0开始递增
    parser.add_argument("--local_rank", type=int, default=0)
    
    parser.add_argument(
        "--skip-test",
        # 如果提供 dest 参数，参数值就保存为命令行参数解析时返回的命名空间对象中名为该 dest 参数值的一个属性。
        dest="skip_test",
        help="Do not test the final model",
        # 默认为store,表示存参数的值  store_true/false  为保存相应的布尔值 触发时为真/假
        action="store_true",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        # 所有剩余的参数，均转化为一个列表赋值给此项    在运行时加上新的以后，要在default.py文件里也加上
        nargs=argparse.REMAINDER,
    )
    # 解析参数
    args = parser.parse_args()
    # WORLD_SIZE由torch.distributed.launch.py产生，具体数值为nproc_per_node*node(这里为1)
    # 因为在执行程序时指定了用分布式，所以此处已经得到了具体要跑几个节点
    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    # distributed为true
    args.distributed = num_gpus > 1

    if args.distributed:
        # 这里设定每一个进程使用的GPU都默认为0的
        torch.cuda.set_device(args.local_rank)
        # 初始化process Group  因为开了n个进程，所以要初始化n次
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        synchronize()
    # 读取配置文件 ，是将在运行代码时输入的参数替换掉default.py中默认参数
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir:
        mkdir(output_dir)
    # 设置日志打印  tan  log名  output_dir 日志保存路径  get_rank()只让0号进程做日志
    logger = setup_logger("tan", output_dir, get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    # 打印出在运行代码是配置的内容，显示一个命名空间 namespace
    logger.info(args)

    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        # 打印出正在运行的数据集的配置文件
        logger.info(config_str)
        # 打印出被数据集配置文件更新以后的默认配置文件
    logger.info("Running with config:\n{}".format(cfg))
    # 拼接保存最终配置文件的保存路径
    output_config_path = os.path.join(cfg.OUTPUT_DIR, 'config.yml')
    logger.info("Saving config into: {}".format(output_config_path))
    # save overloaded model config in the output directory
    # 把配置文件写到文件中 .yml文件
    save_config(cfg, output_config_path)
    # 测试用
    # logger.info("查看是否是分布式训练".format(args.distributed))
    # logger.info("查看使用的local_rank为".format(args.local_rank))
    # logger.info("查看使用的get_rank()为".format(get_rank()))
    # 开始训练
    model = train(cfg, args.local_rank, args.distributed)
    # 如果没有指定跳过测试为true,则再跑一次测试
    # 因为没有在运行代码时指定，所以没有触发，所以skip_test为false
    if not args.skip_test:
        run_test(cfg, model, args.distributed)

if __name__ == "__main__":
    main()
