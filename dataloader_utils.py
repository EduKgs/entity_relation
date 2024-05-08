# !/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import random
from multiprocessing import Pool
import functools
import numpy as np
from collections import defaultdict
from itertools import chain

from utils import Label2IdxSub, Label2IdxObj
import tokenization

global false, null, true
false = null = true = ''

class InputExample(object):
    """a single set of samples of data
    """

    def __init__(self, text, en_pair_list, re_list, rel2ens):
        self.text = text
        self.en_pair_list = en_pair_list
        self.re_list = re_list
        self.rel2ens = rel2ens


class InputFeatures(object):
    """
    Desc:
        a single set of features of data
    """

    def __init__(self,
                 input_tokens,
                 input_ids,
                 attention_mask,
                 seq_tag=None,
                 corres_tag=None,
                 relation=None,
                 triples=None,
                 rel_tag=None
                 ):
        self.input_tokens = input_tokens
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.seq_tag = seq_tag
        self.corres_tag = corres_tag
        self.relation = relation
        self.triples = triples
        self.rel_tag = rel_tag

# 从json文件中加载样本
def read_examples(data_dir, data_sign, rel2idx):
    """load data to InputExamples
    """
    examples = []
    # read src data
    # 根据数据标签读取json文件
    with open(data_dir / f'{data_sign}_triples.json', "r", encoding='utf-8') as f:
        """
        # 原本英文
        data = json.load(f)
        for sample in data:
            text = sample['text']  # 文本
            rel2ens = defaultdict(list)  # 关系id-实体对 字典
            en_pair_list = []  # 实体对列表
            re_list = []  # 关系列表
            # 对于三元组列表中的每个三元组
            # e.g.["Annandale-on-Hudson", "/location/location/contains","Bard College"]
            for triple in sample['triple_list']:
                en_pair_list.append([triple[0], triple[-1]])      # [主语，宾语]
                re_list.append(rel2idx[triple[1]])                # 关系id
                rel2ens[rel2idx[triple[1]]].append((triple[0], triple[-1]))        # 关系id：（主语，宾语）
        
        """

        # CMeIE 和 DuIE2.0
        lines = f.readlines()
        for i, line in enumerate(lines):
            sample = eval(line)
            # 依次读取每个样本
            text = sample['text']           # 文本
            rel2ens = defaultdict(list)     # 关系id-实体对 字典
            en_pair_list = []               # 实体对列表
            re_list = []                    # 关系列表
            # 对于三元组列表中的每个三元组
            for triple in sample['spo_list']:
                en_pair_list.append([triple['subject'], triple['object']['@value']])  # [主语，宾语]
                re_list.append(rel2idx[triple['predicate']])  # 关系id
                rel2ens[rel2idx[triple['predicate']]].append((triple['subject'], triple['object']['@value']))  # 关系id：（主语，宾语）
            # if i <= 3:
                # print(en_pair_list)
                # print(re_list)
                # print(rel2ens)
            example = InputExample(text=text, en_pair_list=en_pair_list, re_list=re_list, rel2ens=rel2ens)
            examples.append(example)

        """
        
        # agr
        data = json.load(f)
        for sample in data:
            text = sample['text']  # 文本
            rel2ens = defaultdict(list)  # 关系id-实体对 字典
            en_pair_list = []  # 实体对列表
            re_list = []  # 关系列表
            # 对于三元组列表中的每个三元组
            for triple in sample['triples']:
                if triple['relation'] in rel2idx:
                    en_pair_list.append([triple['subject'], triple['object']])      # [主语，宾语]
                    re_list.append(rel2idx[triple['relation']])                # 关系id
                    rel2ens[rel2idx[triple['relation']]].append((triple['subject'], triple['object']))        # 关系id：（主语，宾语）
            example = InputExample(text=text, en_pair_list=en_pair_list, re_list=re_list, rel2ens=rel2ens)
            examples.append(example)
        """

    print("InputExamples:", len(examples))
    return examples


# 找到target在source中的起始位置
def find_head_idx(source, target):
    target_len = len(target)
    for i in range(len(source)):
        if source[i: i + target_len] == target:
            return i
    return -1


# 得到主语和宾语在text_tokens中的起始位置（下标）
def _get_so_head(en_pair, tokenizer, text_tokens):
    # 将实体对中的主语和宾语进行分词
    # sub = tokenizer.tokenize(en_pair[0])
    # obj = tokenizer.tokenize(en_pair[1])
    sub = tokenization.BasicTokenizer().tokenize(en_pair[0])
    obj = tokenization.BasicTokenizer().tokenize(en_pair[1])
    # 主语在text_tokens中的起始位置
    sub_head = find_head_idx(source=text_tokens, target=sub)
    # 如果主语和宾语相同
    if sub == obj:
        # 在主语之后的文本标记列表中找到宾语的头部位置
        obj_head = find_head_idx(source=text_tokens[sub_head + len(sub):], target=obj)
        # 如果在主语之后的文本中找到了宾语的头部位置，则将其更新为在整个文本中的位置
        if obj_head != -1:
            obj_head += sub_head + len(sub)
        # 否则，将宾语的头部位置设置为与主语相同的头部位置
        else:
            obj_head = sub_head
    # 如果主语和宾语不同，直接在文本中找到宾语的头部位置
    else:
        obj_head = find_head_idx(source=text_tokens, target=obj)
    return sub_head, obj_head, sub, obj


# 对每个样本调用该方法转换成特征
def convert(example, max_text_len, tokenizer, rel2idx, data_sign, ex_params):
    """convert function
    """
    # 文本分词
    # text_tokens = tokenizer.tokenize(example.text)
    text_tokens = tokenization.BasicTokenizer().tokenize(example.text)
    # cut off
    # 对文本tokens超出长度的部分进行截断
    if len(text_tokens) > max_text_len:             # max_text_len：100
        text_tokens = text_tokens[:max_text_len]
    # token to id
    # 将文本tokens转换成id
    input_ids = tokenizer.convert_tokens_to_ids(text_tokens)
    # 创建一个长度与输入id相同的列表，用于表示注意力掩码。初始值为1，表示所有的标记都应该被注意
    attention_mask = [1] * len(input_ids)
    # zero-padding up to the sequence length
    # 如果输入ID的长度小于最大文本长度，将id末尾填充0，使其达到最大文本长度
    # 将注意力掩码列表末尾填充0，使其与输入ID列表长度一致。表示填充部分不应该被注意
    if len(input_ids) < max_text_len:
        pad_len = max_text_len - len(input_ids)
        # token_pad_id=0
        input_ids += [0] * pad_len
        attention_mask += [0] * pad_len

    # train data
    if data_sign == 'train':
        # construct tags of correspondence and relation
        # 构建该样本的全局矩阵标签和关系标签
        # 创建一个 max_text_len * max_text_len 大小的全局矩阵（全0）
        corres_tag = np.zeros((max_text_len, max_text_len))
        # 创建一个 rel2idx 大小的关系标签（全0），用于表示每个关系的存在与否
        rel_tag = len(rel2idx) * [0]
        # 遍历实体对列表和关系列表
        for en_pair, rel in zip(example.en_pair_list, example.re_list):
            # get sub and obj head
            # 获取主语和宾语的头部位置（在text_tokens中的下标）
            sub_head, obj_head, _, _ = _get_so_head(en_pair, tokenizer, text_tokens)
            # construct relation tag
            # 将该关系标签中对应位置设置为1
            rel_tag[rel] = 1
            # 将全局矩阵中对应 主语-宾语 位置设置为1
            if sub_head != -1 and obj_head != -1:
                corres_tag[sub_head][obj_head] = 1

        # sub_feats: List[InputFeatures]
        sub_feats = []
        # positive samples
        # 遍历关系字典中的每个关系和对应的实体对列表
        for rel, en_ll in example.rel2ens.items():
            # init
            # 创建长度为最大文本长度的列表，用于存储主语/宾语的标签。初始值为'O'，表示非主语/非宾语。
            # 一种关系，一对主语/宾语标签
            tags_sub = max_text_len * [Label2IdxSub['O']]
            tags_obj = max_text_len * [Label2IdxObj['O']]
            # 遍历实体对列表中的每个实体对
            for en in en_ll:
                # get sub and obj head
                # 得到主语和宾语的头部位置以及它们的token
                sub_head, obj_head, sub, obj = _get_so_head(en, tokenizer, text_tokens)
                if sub_head != -1 and obj_head != -1:
                    # 如果主语的头部位置加上主语的长度不超过最大文本长度
                    if sub_head + len(sub) <= max_text_len:
                        # 将主语的头部位置的标签设置为'B-H'，表示主语的开始
                        tags_sub[sub_head] = Label2IdxSub['B-H']
                        # 将主语的头部位置之后，到主语结束位置之前的标签设置为'I-H'，表示主语的中间部分
                        tags_sub[sub_head + 1:sub_head + len(sub)] = (len(sub) - 1) * [Label2IdxSub['I-H']]
                    # 宾语标签同理
                    if obj_head + len(obj) <= max_text_len:
                        tags_obj[obj_head] = Label2IdxObj['B-T']
                        tags_obj[obj_head + 1:obj_head + len(obj)] = (len(obj) - 1) * [Label2IdxObj['I-T']]
            # seq_tag： List[List[int]]
            seq_tag = [tags_sub, tags_obj]

            # sanity check
            # 进行断言检查，确保输入id、主语标签、宾语标签、注意力掩码的长度都等于最大文本长度。
            assert len(input_ids) == len(tags_sub) == len(tags_obj) == len(
                attention_mask) == max_text_len, f'length is not equal!!'
            sub_feats.append(InputFeatures(
                input_tokens=text_tokens,                  # 文本token
                input_ids=input_ids,                       # 文本id
                attention_mask=attention_mask,             # 注意力掩码
                corres_tag=corres_tag,                     # 全局矩阵
                seq_tag=seq_tag,                           # 该关系下的[主语，宾语]标签
                relation=rel,                              # 关系（一种）
                rel_tag=rel_tag                            # 包含该文本下所有关系的关系标签
            ))

        # relation judgement ablation
        # 如果 'ensure_rel' 为False，进行关系判断消融实验，生成负样本的特征
        if not ex_params['ensure_rel']:
            # negative samples
            # 找到不在样本中出现的关系，并将其存储在集合中，根据负样本数量（4），从集合中随机选择
            neg_rels = set(rel2idx.values()).difference(set(example.re_list))
            neg_rels = random.sample(neg_rels, k=ex_params['num_negs'])
            # 生成每个负关系下的负样本的特征
            for neg_rel in neg_rels:
                # init
                # 初始化主语宾语标签均为0
                seq_tag = max_text_len * [Label2IdxSub['O']]
                # sanity check
                assert len(input_ids) == len(seq_tag) == len(attention_mask) == max_text_len, f'length is not equal!!'
                # 正样本：seq_tag = [tags_sub, tags_obj]
                seq_tag = [seq_tag, seq_tag]
                sub_feats.append(InputFeatures(
                    input_tokens=text_tokens,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    corres_tag=corres_tag,
                    seq_tag=seq_tag,
                    relation=neg_rel,                      # 随机生成的负关系
                    rel_tag=rel_tag                        # 包含该文本下所有关系的关系标签
                ))

    # val and test data
    else:
        triples = []
        for rel, en in zip(example.re_list, example.en_pair_list):
            # get sub and obj head
            sub_head, obj_head, sub, obj = _get_so_head(en, tokenizer, text_tokens)
            if sub_head != -1 and obj_head != -1:
                h_chunk = ('H', sub_head, sub_head + len(sub))
                t_chunk = ('T', obj_head, obj_head + len(obj))
                triples.append((h_chunk, t_chunk, rel))
        sub_feats = [
            InputFeatures(
                input_tokens=text_tokens,
                input_ids=input_ids,
                attention_mask=attention_mask,
                triples=triples
            )
        ]

    # get sub-feats
    return sub_feats

# 将样本转换成特征
def convert_examples_to_features(params, examples, tokenizer, rel2idx, data_sign, ex_params):
    """convert examples to features.
    :param examples (List[InputExamples])
    """
    max_text_len = params.max_seq_length    # 100
    # multi-process
    # 创建一个进程池对象 p，并设置最大进程数为 10。这是为了并行处理数据转换的过程。
    with Pool(10) as p:
        # 使用 functools.partial 函数创建一个新的函数 convert_func，它是 convert 函数的一个部分应用。这样做是为了将一些参数固定在 convert 函数中，以便在并行处理中使用
        convert_func = functools.partial(convert, max_text_len=max_text_len, tokenizer=tokenizer, rel2idx=rel2idx,
                                         data_sign=data_sign, ex_params=ex_params)
        # 使用进程池 p 的 map 方法，将 convert_func 应用于 examples 中的每个元素，并返回结果列表
        features = p.map(func=convert_func, iterable=examples)
    # 将并行处理的结果合并为一个列表
    return list(chain(*features))