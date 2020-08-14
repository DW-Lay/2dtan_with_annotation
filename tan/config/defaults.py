import os
# 默认配置文件
from yacs.config import CfgNode as CN
# -----------------------------------------------------------------------------
# Convention about Training / Test specific parameters
# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------
_C = CN()

_C.MODEL = CN()
_C.MODEL.DEVICE = "cuda"
_C.MODEL.ARCHITECTURE = "TAN"
_C.MODEL.WEIGHT = ""   # 模型的权重，    若为空，则直接开始训练

# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
_C.INPUT = CN()
_C.INPUT.NUM_PRE_CLIPS = 256  # 应该能够输入的clip的最大个数
_C.INPUT.PRE_QUERY_SIZE = 300 # 文本中词向量的长度

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASETS = CN()
# List of the dataset names for training, as present in paths_catalog.py
_C.DATASETS.TRAIN = () # 会被具体配置文件的具体值覆盖
# List of the dataset names for testing, as present in paths_catalog.py
_C.DATASETS.TEST = ()

# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
# Number of data loading threads  数据加载启用的线程数 也会在每个文件的具体配置信息中重新配置
_C.DATALOADER.NUM_WORKERS = 4

# -----------------------------------------------------------------------------
# Models
# -----------------------------------------------------------------------------
_C.MODEL.TAN = CN()
# 模型中生成的N个clip特征，下一步在此基础上进行二维图构建
_C.MODEL.TAN.NUM_CLIPS = 128  

_C.MODEL.TAN.FEATPOOL = CN()
# 特征池化操作的相关参数
_C.MODEL.TAN.FEATPOOL.INPUT_SIZE = 4096
_C.MODEL.TAN.FEATPOOL.HIDDEN_SIZE = 512
_C.MODEL.TAN.FEATPOOL.KERNEL_SIZE = 2

_C.MODEL.TAN.FEAT2D = CN()
# 池化参数
_C.MODEL.TAN.FEAT2D.POOLING_COUNTS = [15,8,8,8]

# 整合视频和文本特征
_C.MODEL.TAN.INTEGRATOR = CN()
_C.MODEL.TAN.INTEGRATOR.QUERY_HIDDEN_SIZE = 512
_C.MODEL.TAN.INTEGRATOR.LSTM = CN()
_C.MODEL.TAN.INTEGRATOR.LSTM.NUM_LAYERS = 3
_C.MODEL.TAN.INTEGRATOR.LSTM.BIDIRECTIONAL = False

# 预测得分过程中最后卷积输出的参数
_C.MODEL.TAN.PREDICTOR = CN() 
_C.MODEL.TAN.PREDICTOR.HIDDEN_SIZE = 512
_C.MODEL.TAN.PREDICTOR.KERNEL_SIZE = 5
_C.MODEL.TAN.PREDICTOR.NUM_STACK_LAYERS = 8

# 计算损失函数时的参数
_C.MODEL.TAN.LOSS = CN()
_C.MODEL.TAN.LOSS.MIN_IOU = 0.3
_C.MODEL.TAN.LOSS.MAX_IOU = 0.7

# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()
# 最大迭代次数
_C.SOLVER.MAX_EPOCH = 12
# 初始学习率
_C.SOLVER.LR = 0.01
# 保存模型参数的周期
_C.SOLVER.CHECKPOINT_PERIOD = 1
# 测试周期    
_C.SOLVER.TEST_PERIOD = 1
# 训练的batch_size
_C.SOLVER.BATCH_SIZE = 32
# 自动调整学习率，epoch到8变一次，到11变一次
_C.SOLVER.MILESTONES = (8, 11)

# ---------------------------------------------------------------------------- #
# Specific test options
# ---------------------------------------------------------------------------- #
_C.TEST = CN()
# 测试的batch_size
_C.TEST.BATCH_SIZE = 64
# 非极大值抑制阈值  相似度大于0.4就去掉
_C.TEST.NMS_THRESH = 0.4  
 
# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
_C.OUTPUT_DIR = "."
# os.path.dirname(__file__) 返回当前脚本运行的目录，完整路径或者空目录
_C.PATHS_CATALOG = os.path.join(os.path.dirname(__file__), "paths_catalog.py")
