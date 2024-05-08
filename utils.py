# /usr/bin/env python
# coding=utf-8
"""utils"""
import logging
import os
import shutil
import json
from pathlib import Path

import torch

Label2IdxSub = {"B-H": 1, "I-H": 2, "O": 0}
Label2IdxObj = {"B-T": 1, "I-T": 2, "O": 0}


class Params:
    """参数定义
    """

    def __init__(self, ex_index=1, corpus_type='CMeIE'):
        self.root_path = Path(os.path.abspath(os.path.dirname(__file__)))   # 根目录

        # self.data_dir = self.root_path / f'data/{corpus_type}'              # 数据集目录
        self.data_dir = self.root_path / f'data/{corpus_type}'  # 数据集目录

        self.ex_dir = self.root_path / f'experiments/ex{ex_index}'          # 实验结果输出目录
        self.model_dir = self.root_path / f'model/ex{ex_index}'             # 模型目录

        # self.bert_model_dir = self.root_path / 'bert_base_cased'    # bert模型目录
        self.bert_model_dir = self.root_path / 'chinese_bert_wwm_ext'  # bert模型目录

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')      # 设备 cuda 或 cpu
        self.n_gpu = torch.cuda.device_count()                              # gpu数量
        self.max_seq_length = 128                                           # 最长序列长度
        self.data_cache = False                                             #
        self.train_batch_size = 16                                          # 训练批次大小，WebNLG 为 6，其余数据集为64
        self.val_batch_size = 32                                            # 验证批次大小
        self.test_batch_size = 64                                           # 测试批次大小
        # PRST parameters
        self.seq_tag_size = len(Label2IdxSub)                               # B-I-O标签长度
        # load label2id
        self.rel2idx = json.load(open(self.data_dir/'rel2id.json', 'r', encoding='utf-8'))[-1]    # 加载rel2idx
        self.rel_num = len(self.rel2idx)                                                          # 关系种类

        # early stop strategy
        # 早停策略
        self.min_epoch_num = 20     # 最小的训练轮数。在达到这个轮数之前，不会进行早停检查，默认值为20。
        self.patience = 0.00001       # 早停策略的容忍度。如果验证集上的性能没有提升超过这个阈值，就会计算一次早停指标，默认值为0.00001
        self.patience_num = 20      # 早停策略的容忍次数。如果在连续的这个次数内都没有达到早停指标，就会提前结束训练，默认值为20。

        # learning rate
        # 学习率
        self.fin_tuning_lr = 1e-4                   # 微调阶段的学习率
        self.downs_en_lr = 1e-3                     # 下采样和实体消融阶段的学习率
        self.clip_grad = 2.                         # 梯度裁剪的阈值，用于控制梯度的大小，防止梯度爆炸
        self.drop_prob = 0.3                        # dropout的概率，用于控制神经网络中的dropout层的丢弃概率
        self.weight_decay_rate = 0.01               # 权重衰减的比例，用于控制权重衰减的程度，防止过拟合
        self.warmup_prop = 0.1                      # 学习率预热的比例，在训练的早期阶段逐渐增加学习率，以帮助模型更好地收敛
        self.gradient_accumulation_steps = 2        # 梯度累积的步数，将多个小批次的梯度累积到一起进行参数更新，可以减少显存的使用

    def load(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def save(self, json_path):
        """保存配置到json文件
        """
        params = {}
        with open(json_path, 'w') as f:
            for k, v in self.__dict__.items():
                if isinstance(v, (str, int, float, bool)):
                    params[k] = v
            json.dump(params, f, indent=4)


class RunningAverage:
    """A simple class that maintains the running average of a quantity

    Example:
    ```
    loss_avg = RunningAverage()
    loss_avg.update(2)
    loss_avg.update(4)
    loss_avg() = 3
    ```
    """

    def __init__(self):
        self.steps = 0
        self.total = 0

    def update(self, val):
        self.total += val
        self.steps += 1

    def __call__(self):
        return self.total / float(self.steps)


def set_logger(save=False, log_path=None):
    """Set the logger to log info in terminal and file `log_path`.

    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.

    Example:
    ```
    logging.info("Starting training...")
    ```

    Args:
        log_path: (string) where to log
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if save and not os.path.exists(os.path.dirname(log_path)):
        os.makedirs(os.path.dirname(log_path))

    if not logger.handlers:
        if save:
            # Logging to a file
            file_handler = logging.FileHandler(log_path)
            file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
            logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)


def save_checkpoint(state, is_best, checkpoint):
    """Saves model and training parameters at checkpoint + 'last.pth.tar'. If is_best==True, also saves
    checkpoint + 'best.pth.tar'

    Args:
        state: (dict) contains the entire model, may contain other keys such as epoch, optimizer
        is_best: (bool) True if it is the best model seen till now
        checkpoint: (string) folder where parameters are to be saved
    """
    filepath = os.path.join(checkpoint, 'last.pth.tar')
    if not os.path.exists(checkpoint):
        print("Checkpoint Directory does not exist! Making directory {}".format(checkpoint))
        os.makedirs(checkpoint)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'best.pth.tar'))


def load_checkpoint(checkpoint, optimizer=True):
    """Loads entire model from file_path. If optimizer is True, loads
    optimizer assuming it is present in checkpoint.

    Args:
        checkpoint: (string) filename which needs to be loaded
        optimizer: (bool) resume optimizer from checkpoint
    """
    if not os.path.exists(checkpoint):
        raise ValueError("File doesn't exist {}".format(checkpoint))
    checkpoint = torch.load(checkpoint, map_location=torch.device('cpu'))

    if optimizer:
        return checkpoint['model'], checkpoint['optim']
    return checkpoint['model']
