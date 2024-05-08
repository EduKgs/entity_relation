# /usr/bin/env python
# coding=utf-8
"""Evaluate the model"""
import json
import logging
import random
import argparse
import copy

from tqdm import tqdm
import os

import torch
import numpy as np
import pandas as pd

from metrics import tag_mapping_nearest, tag_mapping_corres
from utils import Label2IdxSub, Label2IdxObj
import utils
from dataloader import CustomDataLoader

# load args
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=2020, help="random seed for initialization")
parser.add_argument('--ex_index', type=str, default=23)
parser.add_argument('--corpus_type', type=str, default="CMeIE", help="NYT, WebNLG, NYT*, WebNLG*")
parser.add_argument('--device_id', type=int, default=1, help="GPU index")
parser.add_argument('--restore_file', default='best', help="name of the file containing weights to reload")
parser.add_argument('--mode', type=str, default='val', help="data type")
parser.add_argument('--corres_threshold', type=float, default=0.5, help="threshold of global correspondence")
parser.add_argument('--rel_threshold', type=float, default=0.5, help="threshold of relation judgement")
parser.add_argument('--ensure_corres', action='store_true', default=True, help="correspondence ablation")
parser.add_argument('--ensure_rel', action='store_true', default=True, help="relation judgement ablation")
parser.add_argument('--emb_fusion', type=str, default="concat", help="way to embedding")


def get_metrics(correct_num, predict_num, gold_num):
    p = correct_num / predict_num if predict_num > 0 else 0
    r = correct_num / gold_num if gold_num > 0 else 0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
    return {
        'correct_num': correct_num,
        'predict_num': predict_num,
        'gold_num': gold_num,
        'precision': p,
        'recall': r,
        'f1': f1
    }


def span2str(triples, tokens):
    def _concat(token_list):
        result = ''
        for idx, t in enumerate(token_list):
            if idx == 0:
                result = t
            elif t.startswith('##'):
                result += t.lstrip('##')
            else:
                result += ' ' + t
        return result

    output = []
    for triple in triples:
        rel = triple[-1]
        sub_tokens = tokens[triple[0][1]:triple[0][-1]]
        obj_tokens = tokens[triple[1][1]:triple[1][-1]]
        sub = _concat(sub_tokens)
        obj = _concat(obj_tokens)
        output.append((sub, obj, rel))
    return output

def evaluate(model, data_iterator, params, ex_params, mark='Val'):
    """Evaluate the model on `steps` batches."""
    # set model to evaluation mode
    model.eval()
    rel_num = params.rel_num

    predictions = []
    ground_truths = []
    correct_num, predict_num, gold_num = 0, 0, 0
    start_correct_num, end_correct_num = 0, 0

    for batch in tqdm(data_iterator, unit='Batch', ascii=True):
        # to device
        batch = tuple(t.to(params.device) if isinstance(t, torch.Tensor) else t for t in batch)
        input_ids, attention_mask, triples, input_tokens = batch
        print(input_ids)
        bs, seq_len = input_ids.size()

        # inference
        with torch.no_grad():
            pred_seqs, pre_corres, xi, pred_rels = model(input_ids, attention_mask=attention_mask,
                                                         ex_params=ex_params)
            # (sum(x_i), seq_len)
            pred_seqs = pred_seqs.detach().cpu().numpy()
            # (bs, seq_len, seq_len)
            pre_corres = pre_corres.detach().cpu().numpy()
        if ex_params['ensure_rel']:
            # (bs,)
            # 将xi从列表类型转换为NumPy数组。xi表示每个样本的x_i值
            xi = np.array(xi)
            # (sum(s_i),)
            # 将pred_rels从Tensor类型转换为NumPy数组，并将其从设备上移动到CPU上。pred_rels表示预测的关系索引
            pred_rels = pred_rels.detach().cpu().numpy()
            # decode by per batch
            # 通过每个批次进行解码，计算出每个批次的累积x_i值的列表xi_index
            xi_index = np.cumsum(xi).tolist()
            # (bs+1,)
            # 将0插入到xi_index列表的开头，形成一个长度为bs+1的列表
            xi_index.insert(0, 0)


        for idx in range(bs):
            if ex_params['ensure_rel']:
                pre_triples = tag_mapping_corres(predict_tags=pred_seqs[xi_index[idx]:xi_index[idx + 1]],
                                                 pre_corres=pre_corres[idx],
                                                 pre_rels=pred_rels[xi_index[idx]:xi_index[idx + 1]],
                                                 label2idx_sub=Label2IdxSub,
                                                 label2idx_obj=Label2IdxObj)
            else:
                pre_triples = tag_mapping_corres(predict_tags=pred_seqs[idx * rel_num:(idx + 1) * rel_num],
                                                 pre_corres=pre_corres[idx],
                                                 label2idx_sub=Label2IdxSub,
                                                 label2idx_obj=Label2IdxObj)

            gold_triples = span2str(triples[idx], input_tokens[idx])
            pre_triples = span2str(pre_triples, input_tokens[idx])
            ground_truths.append(list(set(gold_triples)))
            predictions.append(list(set(pre_triples)))

            gold = set(gold_triples)
            pre = set(pre_triples)
            correct = pre & gold
            correct_num += len(correct)
            predict_num += len(pre)
            gold_num += len(gold)

            god_set1 = copy.deepcopy(gold)
            # sub obj 首字母匹配 和 关系匹配
            for pre_triple in pre:
                start_sub = pre_triple[0][0]
                start_obj = pre_triple[1][0]
                rel = pre_triple[2]
                for gold_triple in god_set1.copy():
                    if gold_triple[0][0] == start_sub and gold_triple[1][0] == start_obj and gold_triple[2] == rel:
                        start_correct_num += 1
                        god_set1.remove(gold_triple)

            god_set2 = copy.deepcopy(gold)
            # sub obj 尾字母匹配 和 关系匹配
            for pre_triple in pre:
                end_sub = pre_triple[0][-1]
                end_obj = pre_triple[1][-1]
                rel = pre_triple[2]
                for gold_triple in god_set2.copy():
                    if gold_triple[0][-1] == end_sub and gold_triple[1][-1] == end_obj and gold_triple[2] == rel:
                        end_correct_num += 1
                        god_set2.remove(gold_triple)

    metrics_all = get_metrics(correct_num, predict_num, gold_num)
    metrics_start = get_metrics(start_correct_num, predict_num, gold_num)
    metrics_end = get_metrics(end_correct_num, predict_num, gold_num)

    # logging loss, f1 and report
    metrics_str_all = "; ".join("{}: {:05.5f}".format(k, v) for k, v in metrics_all.items())
    metrics_str_start = "; ".join("{}: {:05.5f}".format(k, v) for k, v in metrics_start.items())
    metrics_str_end = "; ".join("{}: {:05.5f}".format(k, v) for k, v in metrics_end.items())
    logging.info("- {} metrics(all):\n".format(mark) + metrics_str_all)
    logging.info("- {} metrics(start):\n".format(mark) + metrics_str_start)
    logging.info("- {} metrics(end):\n".format(mark) + metrics_str_end)
    return metrics_all, predictions, ground_truths


if __name__ == '__main__':
    args = parser.parse_args()
    params = utils.Params(ex_index=args.ex_index, corpus_type=args.corpus_type)
    ex_params = {
        'corres_threshold': args.corres_threshold,
        'rel_threshold': args.rel_threshold,
        'ensure_corres': args.ensure_corres,
        'ensure_rel': args.ensure_rel,
        'emb_fusion': args.emb_fusion
    }

    torch.cuda.set_device(args.device_id)
    print('current device:', torch.cuda.current_device())
    mode = args.mode
    # Set the random seed for reproducible experiments
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    params.seed = args.seed

    # Set the logger
    utils.set_logger()

    # get dataloader
    dataloader = CustomDataLoader(params)

    # Define the model
    logging.info('Loading the model...')
    logging.info(f'Path: {os.path.join(params.model_dir, args.restore_file)}.pth.tar')
    # Reload weights from the saved file
    model, optimizer = utils.load_checkpoint(os.path.join(params.model_dir, args.restore_file + '.pth.tar'))
    model.to(params.device)
    logging.info('- done.')

    logging.info("Loading the dataset...")
    loader = dataloader.get_dataloader(data_sign=mode, ex_params=ex_params)
    logging.info('-done')

    logging.info("Starting prediction...")
    _, predictions, ground_truths = evaluate(model, loader, params, ex_params, mark=mode)

    """
    with open(params.data_dir / f'{mode}_triples.json', 'r', encoding='utf-8') as f_src:
    
        src = json.load(f_src)
        df = pd.DataFrame(
            {
                'text': [sample['text'] for sample in src],
                'pre': predictions,
                'truth': ground_truths
            }
        )

        lines = f_src.readlines()
        df = pd.DataFrame(
            {
                'text': [eval(line)['text'] for line in lines],
                'pre': predictions,
                'truth': ground_truths
            }
        )
        df.to_csv(params.ex_dir / f'{mode}_result.csv')
    logging.info('-done')
    """
