# /usr/bin/env python
# coding=utf-8
"""train with valid"""
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

import torch
from transformers import BertConfig
import random
import logging
from tqdm import trange
import argparse

import utils
from optimization import BertAdam
from evaluate import evaluate
from dataloader import CustomDataLoader
from model import BertForRE, FGM, PGD

# load args
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=2020, help="random seed for initialization")   # 随机种子
parser.add_argument('--ex_index', type=str, default=1)  # 实验索引
parser.add_argument('--corpus_type', type=str, default="CMeIE", help="NYT, WebNLG, NYT*, WebNLG*")  # 数据库类型
parser.add_argument('--device_id', type=int, default=0, help="GPU index")   # GPU索引
# parser.add_argument('--epoch_num', required=True, type=int, default=1, help="number of epochs")    # 训练轮数 100
parser.add_argument('--epoch_num', type=int, default=30, help="number of epochs")    # 训练轮数 100
parser.add_argument('--multi_gpu', action='store_true', help="ensure multi-gpu training")   # 是否使用多GPU进行训练。如果提供了该参数，则设置为True。
parser.add_argument('--restore_file', default=None, help="name of the file containing weights to reload")   # 要重新加载的权重的文件名
parser.add_argument('--corres_threshold', type=float, default=0.5, help="threshold of global correspondence")   # 全局对应性的阈值
parser.add_argument('--rel_threshold', type=float, default=0.5, help="threshold of relation judgement")     # 关系判断的阈值
parser.add_argument('--ensure_corres', action='store_true', default=True, help="correspondence ablation")     # 是否进行对应性消融
parser.add_argument('--ensure_rel', action='store_true', default=True, help="relation judgement ablation")    # 是否进行关系判断消融
parser.add_argument('--emb_fusion', type=str, default="concat", help="way to embedding")        # 进行嵌入融合的方式，默认值为"concat"。
parser.add_argument('--num_negs', type=int, default=4,
                    help="number of negative sample when ablate relation judgement")    # 在进行关系判断消融时的负样本数量。默认值为4。

# 一个轮次的训练过程
def train(model, data_iterator, optimizer, params, ex_params):
    """Train the model one epoch
    """
    # set model to training mode
    # 将模型设置为训练模式
    model.train()

    loss_avg = utils.RunningAverage()
    loss_avg_seq = utils.RunningAverage()
    loss_avg_mat = utils.RunningAverage()
    loss_avg_rel = utils.RunningAverage()

    # Use tqdm for progress bar
    # one epoch
    # 通过使用tqdm库来显示训练进度条。
    t = trange(len(data_iterator), ascii=True)
    # 循环遍历数据迭代器中的每个批次(batch)
    for step, _ in enumerate(t):
        # fetch the next training batch
        batch = next(iter(data_iterator))
        batch = tuple(t.to(params.device) for t in batch)
        input_ids, attention_mask, seq_tags, relations, corres_tags, rel_tags = batch

        # compute model output and loss
        # 将输入数据传递给模型，计算模型的输出和损失（forword方法中）
        loss, loss_seq, loss_mat, loss_rel, rel_emb = model(input_ids, attention_mask=attention_mask, seq_tags=seq_tags,
                                                            potential_rels=relations, corres_tags=corres_tags,
                                                            rel_tags=rel_tags, ex_params=ex_params)
        # 如果使用多个GPU进行训练，需要对损失进行平均处理。
        if params.n_gpu > 1 and args.multi_gpu:
            loss = loss.mean()  # mean() to average on multi-gpu.
        # 如果梯度累积的步数大于1，需要对损失进行归一化。
        if params.gradient_accumulation_steps > 1:
            loss = loss / params.gradient_accumulation_steps
        # back-prop
        # 进行反向传播
        loss.backward()


        # FGM
        fgm = FGM(model, rel_emb)
        # 对抗训练
        fgm.attack()  # embedding被修改了
        # optimizer.zero_grad()     # 如果不想累加梯度，就把这里的注释取消
        loss_sum, _, _, _, _ = model(input_ids, attention_mask=attention_mask, seq_tags=seq_tags,
                                     potential_rels=relations, corres_tags=corres_tags, rel_tags=rel_tags,
                                     ex_params=ex_params)
        loss_sum.backward()  # 反向传播，在正常的grad基础上，累加对抗训练的梯度
        fgm.restore()  # 恢复Embedding的参数


        """
        # PGD
        pgd = PGD(model, rel_emb)
        K = 3
        pgd.backup_grad()  # 保存正常的grad
        # 对抗训练
        for t in range(K):
            pgd.attack(is_first_attack=(t == 0))  # 在embedding上添加对抗扰动, first attack时备份param.data
            if t != K - 1:
                optimizer.zero_grad()
            else:
                pgd.restore_grad()  # 恢复正常的grad
            loss_sum, _, _, _, _ = model(input_ids, attention_mask=attention_mask, seq_tags=seq_tags,
                                         potential_rels=relations, corres_tags=corres_tags, rel_tags=rel_tags,
                                         ex_params=ex_params)
            loss_sum.backward()  # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
        pgd.restore()  # 恢复embedding参数
        """

        # 步数达到梯度累计步数的整数倍时，进行参数更新
        if (step + 1) % params.gradient_accumulation_steps == 0:
            # performs updates using calculated gradients
            optimizer.step()
            model.zero_grad()
        # update the average loss
        # 更新平均损失(loss_avg)和其他损失(loss_avg_seq, loss_avg_mat, loss_avg_rel)。
        loss_avg.update(loss.item() * params.gradient_accumulation_steps)
        loss_avg_seq.update(loss_seq.item())
        loss_avg_mat.update(loss_mat.item())
        loss_avg_rel.update(loss_rel.item())
        # 最后，在进度条中显示平均损失和其他损失。
        # 右边第一个0为填充数，第二个5为数字个数为5位，第三个3为小数点有效数为3，最后一个f为数据类型为float类型。
        t.set_postfix(loss='{:05.5f}'.format(loss_avg()),
                      loss_seq='{:05.5f}'.format(loss_avg_seq()),
                      loss_mat='{:05.5f}'.format(loss_avg_mat()),
                      loss_rel='{:05.5f}'.format(loss_avg_rel()))
    logging.info('loss={:05.5f},'.format(loss_avg()))
    logging.info('loss_seq={:05.5f},'.format(loss_avg_seq()))
    logging.info('loss_mat={:05.5f},'.format(loss_avg_mat()))
    logging.info('loss_rel={:05.5f},'.format(loss_avg_rel()))


def train_and_evaluate(model, params, ex_params, restore_file=None):
    """Train the model and evaluate every epoch."""
    # Load training data and val data
    # 加载训练集和验证集
    dataloader = CustomDataLoader(params)
    train_loader = dataloader.get_dataloader(data_sign='train', ex_params=ex_params)
    val_loader = dataloader.get_dataloader(data_sign='val', ex_params=ex_params)

    # reload weights from restore_file if specified
    if restore_file is not None:
        restore_path = os.path.join(params.model_dir, args.restore_file + '.pth.tar')
        logging.info("Restoring parameters from {}".format(restore_path))
        # 读取checkpoint
        model, optimizer = utils.load_checkpoint(restore_path)

    model.to(params.device)
    # parallel model
    if params.n_gpu > 1 and args.multi_gpu:
        model = torch.nn.DataParallel(model)

    # Prepare optimizer
    # fine-tuning
    param_optimizer = list(model.named_parameters())
    # pretrain model param
    param_pre = [(n, p) for n, p in param_optimizer if 'bert' in n]
    # downstream model param
    param_downstream = [(n, p) for n, p in param_optimizer if 'bert' not in n]
    no_decay = ['bias', 'LayerNorm', 'layer_norm']
    optimizer_grouped_parameters = [
        # pretrain model param
        {'params': [p for n, p in param_pre if not any(nd in n for nd in no_decay)],
         'weight_decay': params.weight_decay_rate, 'lr': params.fin_tuning_lr
         },
        {'params': [p for n, p in param_pre if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0, 'lr': params.fin_tuning_lr
         },
        # downstream model
        {'params': [p for n, p in param_downstream if not any(nd in n for nd in no_decay)],
         'weight_decay': params.weight_decay_rate, 'lr': params.downs_en_lr
         },
        {'params': [p for n, p in param_downstream if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0, 'lr': params.downs_en_lr
         }
    ]
    num_train_optimization_steps = len(train_loader) // params.gradient_accumulation_steps * args.epoch_num
    optimizer = BertAdam(optimizer_grouped_parameters, warmup=params.warmup_prop, schedule="warmup_cosine",
                         t_total=num_train_optimization_steps, max_grad_norm=params.clip_grad)

    # patience stage
    best_val_f1 = 0.0
    # best_val_p = 0.000
    patience_counter = 0

    # 依次进行每个轮次训练
    for epoch in range(1, args.epoch_num + 1):
        # Run one epoch
        logging.info("Epoch {}/{}".format(epoch, args.epoch_num))

        # Train for one epoch on training set
        # 完成训练集一个轮次的训练
        train(model, train_loader, optimizer, params, ex_params)

        # Evaluate for one epoch on training set and validation set
        # train_metrics = evaluate(args, model, train_loader, params, mark='Train',
        #                          verbose=True)  # Dict['loss', 'f1']
        val_metrics, _, _ = evaluate(model, val_loader, params, ex_params, mark='Val')
        val_f1 = val_metrics['f1']
        improve_f1 = val_f1 - best_val_f1
        # val_p = val_metrics['precision']
        # improve_p = val_p - best_val_p

        # Save weights of the network
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
        optimizer_to_save = optimizer
        utils.save_checkpoint({'epoch': epoch + 1,
                               'model': model_to_save,
                               'optim': optimizer_to_save},
                              is_best=improve_f1 > 0,
                              checkpoint=params.model_dir)
        params.save(params.ex_dir / 'params.json')

        # stop training based params.patience
        # """
        if improve_f1 > 0:
            logging.info("- Found new best F1")
            best_val_f1 = val_f1
            if improve_f1 < params.patience:
                patience_counter += 1
            else:
                patience_counter = 0
        """
        if improve_p > 0:
            logging.info("- Found new best precision")
            best_val_p = val_p
            if improve_p < params.patience:
                patience_counter += 1
            else:
                patience_counter = 0
        else:
            patience_counter += 1
        """

        # Early stopping and logging best f1
        if (patience_counter > params.patience_num and epoch > params.min_epoch_num) or epoch == args.epoch_num:
            logging.info("Best val p: {:05.3f}".format(best_val_f1))
            break


if __name__ == '__main__':
    args = parser.parse_args()
    params = utils.Params(args.ex_index, args.corpus_type)
    ex_params = {
        'ensure_corres': args.ensure_corres,    # 是否进行对应性消融
        'ensure_rel': args.ensure_rel,          # 是否进行关系判断消融
        'num_negs': args.num_negs,              # 进行关系判断消融时的负样本数量，默认值为4。
        'emb_fusion': args.emb_fusion           # 嵌入融合的方式，默认值为"concat"。
    }

    # 根据命令行参数来配置GPU的使用和设置随机种子，以确保实验的一致性和可重复性。

    # 多GPU计算
    if args.multi_gpu:
        params.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        n_gpu = torch.cuda.device_count()
        params.n_gpu = n_gpu
    # 单GPU计算
    else:
        torch.cuda.set_device(args.device_id)
        print('current device:', torch.cuda.current_device())
        params.n_gpu = n_gpu = 1

    # Set the random seed for reproducible experiments
    # 设置随机种子
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    params.seed = args.seed
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    # Set the logger
    # 设置日志
    utils.set_logger(save=True, log_path=os.path.join(params.ex_dir, 'train.log'))
    logging.info(f"Model type:")
    logging.info("device: {}".format(params.device))

    logging.info('Load pre-train model weights...')
    # "bert_model": "C:/Users/Administrator/Desktop/实体链接/code/BLINK/blink/chinese_bert_wwm_ext"

    # bert_config = BertConfig.from_json_file(os.path.join(params.bert_model_dir, 'bert_config.json'))
    bert_config = BertConfig.from_json_file(os.path.join(params.bert_model_dir, 'config.json'))

    model = BertForRE.from_pretrained(config=bert_config,
                                      pretrained_model_name_or_path=params.bert_model_dir,
                                      params=params)
    logging.info('-done')

    # Train and evaluate the model
    # 训练并评估
    logging.info("Starting training for {} epoch(s)".format(args.epoch_num))
    train_and_evaluate(model, params, ex_params, args.restore_file)
