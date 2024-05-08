# /usr/bin/env python
# coding=utf-8
"""Dataloader"""

import os
import json
import sys

os.chdir(sys.path[0])

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, Dataset
from transformers import BertTokenizer

from dataloader_utils import read_examples, convert_examples_to_features


class FeatureDataset(Dataset):
    """Pytorch Dataset for InputFeatures
    """

    def __init__(self, features):
        self.features = features

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index):
        return self.features[index]


class CustomDataLoader(object):
    def __init__(self, params):
        self.params = params

        self.train_batch_size = params.train_batch_size
        self.val_batch_size = params.val_batch_size
        self.test_batch_size = params.test_batch_size

        self.data_dir = params.data_dir
        self.max_seq_length = params.max_seq_length

        # self.tokenizer = BertTokenizer(vocab_file=os.path.join(params.bert_model_dir, 'vocab.txt'),do_lower_case=False)
        # self.tokenizer = BertTokenizer(vocab_file='C:/Users/Administrator/Desktop/联合抽取/PRGC/PRGC-main/bert-base-cased/vocab.txt',do_lower_case=False)
        self.tokenizer = BertTokenizer.from_pretrained('D:/yangsirui/PRGC-main/chinese_bert_wwm_ext/vocab.txt')

        self.data_cache = params.data_cache

    @staticmethod
    def collate_fn_train(features):
        """将InputFeatures转换为Tensor
        Args:
            features (List[InputFeatures])
        Returns:
            tensors (List[Tensors])
        """
        input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
        seq_tags = torch.tensor([f.seq_tag for f in features], dtype=torch.long)
        poten_relations = torch.tensor([f.relation for f in features], dtype=torch.long)
        corres_tags = torch.tensor([f.corres_tag for f in features], dtype=torch.long)
        rel_tags = torch.tensor([f.rel_tag for f in features], dtype=torch.long)
        tensors = [input_ids, attention_mask, seq_tags, poten_relations, corres_tags, rel_tags]
        return tensors

    @staticmethod
    def collate_fn_test(features):
        """将InputFeatures转换为Tensor
        Args:
            features (List[InputFeatures])
        Returns:
            tensors (List[Tensors])
        """
        input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
        triples = [f.triples for f in features]
        input_tokens = [f.input_tokens for f in features]
        tensors = [input_ids, attention_mask, triples, input_tokens]
        return tensors

    # 直接读取输入特征 或 将输入样本转化成输入特征
    def get_features(self, data_sign, ex_params):
        """convert InputExamples to InputFeatures
        :param data_sign: 'train', 'val' or 'test'
        """
        # 打印加载数据的提示信息
        print("=*=" * 10)
        print("Loading {} data...".format(data_sign))
        # get features
        # 根据数据标识和最大序列长度构建缓存路径
        cache_path = os.path.join(self.data_dir, "{}.cache.{}".format(data_sign, str(self.max_seq_length)))
        # 如果缓存路径存在且启用了数据缓存（data_cache），则从缓存中加载特征
        if os.path.exists(cache_path) and self.data_cache:
            features = torch.load(cache_path)
        else:
            # get relation to idx
            # 读取关系到索引的映射关系（rel2idx）
            with open(self.data_dir / f'rel2id.json', 'r', encoding='utf-8') as f_re:
                rel2idx = json.load(f_re)[-1]
            # get examples
            # 根据数据标识从数据目录中读取输入示例（InputExamples）
            if data_sign in ("train", "val", "test", "pseudo", 'EPO', 'SEO', 'SOO', 'Normal', '1', '2', '3', '4', '5'):
                examples = read_examples(self.data_dir, data_sign=data_sign, rel2idx=rel2idx)
            else:
                raise ValueError("please notice that the data can only be train/val/test!!")
            # 根据参数和输入样本将其转换为输入特征（InputFeatures）
            features = convert_examples_to_features(self.params, examples, self.tokenizer, rel2idx, data_sign,
                                                    ex_params)
            # save data
            # 如果启用了数据缓存（data_cache），则将特征保存到缓存路径
            if self.data_cache:
                torch.save(features, cache_path)
        return features

    # 加载数据集中的数据
    def get_dataloader(self, data_sign="train", ex_params=None):
        """construct dataloader
        :param data_sign: 'train', 'val' or 'test'
        """
        # InputExamples to InputFeatures
        # 根据 data_sign 得到特征，并封装成FeatureDataset
        features = self.get_features(data_sign=data_sign, ex_params=ex_params)
        dataset = FeatureDataset(features)
        print(f"{len(features)} {data_sign} data loaded!")
        print("=*=" * 10)
        # construct dataloader
        # RandomSampler(dataset) or SequentialSampler(dataset)
        # 根据data_sign参数选择数据采样器（datasampler）
        # 在训练集中使用随机采样器（RandomSampler），在验证集和测试集中使用顺序采样器（SequentialSampler）
        # 根据数据采样器、批次大小和数据处理函数（collate_fn）构建数据加载器（dataloader）。
        if data_sign == "train":
            datasampler = RandomSampler(dataset)
            dataloader = DataLoader(dataset, sampler=datasampler, batch_size=self.train_batch_size,
                                    collate_fn=self.collate_fn_train)
        elif data_sign == "val":
            datasampler = SequentialSampler(dataset)
            dataloader = DataLoader(dataset, sampler=datasampler, batch_size=self.val_batch_size,
                                    collate_fn=self.collate_fn_test)
        elif data_sign in ("test", "pseudo", 'EPO', 'SEO', 'SOO', 'Normal', '1', '2', '3', '4', '5'):
            datasampler = SequentialSampler(dataset)
            dataloader = DataLoader(dataset, sampler=datasampler, batch_size=self.test_batch_size,
                                    collate_fn=self.collate_fn_test)
        # 如果data_sign参数不是'train'、'val'或'test'，则抛出ValueError异常。
        else:
            raise ValueError("please notice that the data can only be train/val/test !!")
        return dataloader


if __name__ == '__main__':
    from utils import Params

    # params = Params(corpus_type='NYT')
    params = Params(corpus_type='CMeIE')

    ex_params = {
        'ensure_relpre': True,
        # 下面两个是新加的
        'ensure_rel': True,
        'num_negs': 4
    }
    dataloader = CustomDataLoader(params)
    feats = dataloader.get_features(ex_params=ex_params, data_sign='val')
    print(feats[7].input_tokens)
    print(feats[7].input_ids)
    print(feats[7].corres_tag)
    print(feats[7].seq_tag)
    print(feats[7].relation)
    print(feats[7].rel_tag)
