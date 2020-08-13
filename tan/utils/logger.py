import logging
import os
import sys
# 创建日志打印输出
def setup_logger(name, save_dir, distributed_rank, filename="log.txt"):
    logger = logging.getLogger(name)
    # 日志级别设为debug
    logger.setLevel(logging.DEBUG)
    # don't log results for the non-master process
    # 只有0号进程设置打印详细日志，其他的只打印debug日志
    if distributed_rank > 0:
        return logger
    # 创建了Logger对象，接下来我们需要创建handler对象，
    # handler对象是用来配置处理log时用的格式，级别等等特性的，
    # 我们可以根据需求创建各种不同的handler，比如将log记录保存在文件的FileHandler，
    # 将log输出到屏幕的StreamHandler，支持日志根据时间回滚的TimedRotatingFileHandler，
    # 根据文件大小回滚的RotatingFileHandler等等

    # 输出到屏幕上的handler
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # 保存到文件中handler
    if save_dir:
        fh = logging.FileHandler(os.path.join(save_dir, filename))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger
