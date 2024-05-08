# /usr/bin/env python
# coding=utf-8
"""model"""
from collections import Counter

import torch
import torch.nn as nn

from transformers import BertPreTrainedModel, BertModel

# 多类别非线性分类器
# 使用了一个线性层和一个隐藏层到标签的线性层

# 这个神经网络的结构如下：
# 线性层（self.linear）：输入特征的维度为hidden_size，输出特征的维度为int(hidden_size / 2)。
# ReLU激活函数：对线性层的输出进行非线性变换。
# Dropout层（self.dropout）：以dropout_rate的概率对输入进行随机置零，用于正则化。
# 隐藏层到标签的线性层（self.hidden2tag）：输入特征的维度为int(hidden_size / 2)，输出特征的维度为tag_size。
class MultiNonLinearClassifier(nn.Module):
    # hidden_size（隐藏层大小）、tag_size（标签大小）、dropout_rate（dropout概率）
    def __init__(self,hidden_size, tag_size, dropout_rate):
        super(MultiNonLinearClassifier, self).__init__()
        self.tag_size = tag_size
        # 一个线性层
        self.linear = nn.Linear(hidden_size, int(hidden_size / 2))
        # 一个隐藏层到标签的线性层
        self.hidden2tag = nn.Linear(int(hidden_size / 2), self.tag_size)
        # dropout层
        self.dropout = nn.Dropout(dropout_rate)

    # 输入特征经过线性层后，通过ReLU激活函数进行非线性变换，并通过dropout层进行正则化
    # 最后，经过隐藏层到标签的线性层，输出分类结果。
    def forward(self, input_features):
        # 输入特征经过线性层
        features_tmp = self.linear(input_features)
        # 通过ReLU激活函数进行非线性变换
        features_tmp = nn.ReLU()(features_tmp)
        # 通过dropout层进行正则化
        features_tmp = self.dropout(features_tmp)
        # 经过隐藏层到标签的线性层，输出分类结果
        features_output = self.hidden2tag(features_tmp)
        return features_output


# 对于主语、宾语的标签分类器
class SequenceLabelForSO(nn.Module):
    def __init__(self, hidden_size, tag_size, dropout_rate):
        super(SequenceLabelForSO, self).__init__()
        self.tag_size = tag_size
        self.linear = nn.Linear(hidden_size, int(hidden_size / 2))
        self.hidden2tag_sub = nn.Linear(int(hidden_size / 2), self.tag_size)
        self.hidden2tag_obj = nn.Linear(int(hidden_size / 2), self.tag_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, input_features):
        """
        Args:
            input_features: (bs, seq_len, h)
        """
        # 输入特征经过线性层
        features_tmp = self.linear(input_features)
        # 通过ReLU激活函数进行非线性变换
        features_tmp = nn.ReLU()(features_tmp)
        # 通过dropout层进行正则化
        features_tmp = self.dropout(features_tmp)
        # 经过隐藏层到主语标签的线性层，输出主语分类结果
        sub_output = self.hidden2tag_sub(features_tmp)
        # 经过隐藏层到宾语标签的线性层，输出宾语分类结果
        obj_output = self.hidden2tag_obj(features_tmp)
        return sub_output, obj_output


class BertForRE(BertPreTrainedModel):
    def __init__(self, config, params):
        super().__init__(config)
        self.max_seq_len = params.max_seq_length        # 100
        self.seq_tag_size = params.seq_tag_size         # B-I-O标签长度  3
        self.rel_num = params.rel_num                   # 关系种类
        self.bidirectional = True

        # pretrain model
        self.bert = BertModel(config)
        # sequence tagging
        # 主语、宾语标记

        # """
        self.linear = nn.Linear(config.hidden_size * 4, config.hidden_size * 2)
        self.dropout = nn.Dropout(params.drop_prob)
        self.lstm_sub = nn.LSTM(input_size=768 * 2, hidden_size=config.hidden_size * 2, num_layers=1, batch_first=True,
                                bidirectional=self.bidirectional)
        # linear and sigmoid layers
        if self.bidirectional:
            self.sequence_tagging_sub = MultiNonLinearClassifier(config.hidden_size * 2, self.seq_tag_size, params.drop_prob)
        else:
            self.sequence_tagging_sub = MultiNonLinearClassifier(config.hidden_size, self.seq_tag_size, params.drop_prob)

        self.lstm_obj = nn.LSTM(input_size=768 * 2, hidden_size=config.hidden_size * 2, num_layers=1, batch_first=True,
                                bidirectional=self.bidirectional)
        # linear and sigmoid layers
        if self.bidirectional:
            self.sequence_tagging_obj = MultiNonLinearClassifier(config.hidden_size * 2, self.seq_tag_size, params.drop_prob)
        else:
            self.sequence_tagging_obj = MultiNonLinearClassifier(config.hidden_size, self.seq_tag_size, params.drop_prob)
        # """

        self.sequence_tagging_sub = MultiNonLinearClassifier(config.hidden_size * 2, self.seq_tag_size, params.drop_prob)
        self.sequence_tagging_obj = MultiNonLinearClassifier(config.hidden_size * 2, self.seq_tag_size, params.drop_prob)
        self.sequence_tagging_sum = SequenceLabelForSO(config.hidden_size, self.seq_tag_size, params.drop_prob)

        # global correspondence
        # 全局一致性
        self.global_corres = MultiNonLinearClassifier(config.hidden_size * 2, 1, params.drop_prob)
        # relation judgement
        # 关系分类
        """
        self.lstm = nn.LSTM(input_size=768, hidden_size=config.hidden_size, num_layers=1, batch_first=True, bidirectional=self.bidirectional)
        # linear and sigmoid layers
        if self.bidirectional:
            # self.rel_judgement = MultiNonLinearClassifier(config.hidden_size * 2, params.rel_num, params.drop_prob)
            self.rel_judgement = nn.Linear(config.hidden_size * 2, params.rel_num)
        else:
            # self.rel_judgement = MultiNonLinearClassifier(config.hidden_size, params.rel_num, params.drop_prob)
            self.rel_judgement = nn.Linear(config.hidden_size, params.rel_num)
        """
        self.rel_judgement = MultiNonLinearClassifier(config.hidden_size, params.rel_num, params.drop_prob)
        self.rel_embedding = nn.Embedding(params.rel_num, config.hidden_size)

        self.init_weights()

    @staticmethod
    def masked_avgpool(sent, mask):
        mask_ = mask.masked_fill(mask == 0, -1e9).float()
        score = torch.softmax(mask_, -1)
        return torch.matmul(score.unsqueeze(1), sent).squeeze(1)

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            seq_tags=None,
            potential_rels=None,
            corres_tags=None,
            rel_tags=None,
            ex_params=None
    ):
        """
        Args:
            input_ids: (batch_size, seq_len)
            attention_mask: (batch_size, seq_len)
            rel_tags: (bs, rel_num)
            potential_rels: (bs,), only in train stage.
            seq_tags: (bs, 2, seq_len)
            corres_tags: (bs, seq_len, seq_len)
            ex_params: experiment parameters
        """
        # get params for experiments
        # 获取键为'corres_threshold'和'rel_threshold'的值，如果不存在，则使用默认值0.5和0.1
        corres_threshold, rel_threshold = ex_params.get('corres_threshold', 0.5), ex_params.get('rel_threshold', 0.1)
        # ablation study
        ensure_corres, ensure_rel = ex_params['ensure_corres'], ex_params['ensure_rel']
        # pre-train model
        # 调用self.bert模型，将input_ids和attention_mask作为参数传递给模型，并设置output_hidden_states=True以获取隐藏状态，将返回的结果存储在outputs变量中
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )  # sequence_output, pooled_output, (hidden_states), (attentions)
        # 从outputs中获取第一个元素，即模型的输出序列，并将其存储在sequence_output变量中
        sequence_output = outputs[0]
        # 获取sequence_output的形状，分别将形状的三个维度赋给bs、seq_len和h变量
        bs, seq_len, h = sequence_output.size()

        if ensure_rel:

            # (bs, h)
            h_k_avg = self.masked_avgpool(sequence_output, attention_mask)
            # (bs, rel_num)
            rel_pred = self.rel_judgement(h_k_avg)

            """
            # (bs, seq_len, h)
            lstm_input = sequence_output
            lstm_output, (hidden_last,cn_last) = self.lstm(lstm_input)
            # print("lstm_output.shape", lstm_output.size())          # [16, 128, 1536]
            # print("hidden_last.shape", hidden_last.size())          # [2, 16, 768]
            # print("cn_last.shape", cn_last.size())                  # [2, 16, 768]
            if self.bidirectional:
                # 正向最后一层，最后一个时刻
                hidden_last_L = hidden_last[-2]
                # print("hidden_last_L.shape", hidden_last_L.size())      [16, 768]
                # 反向最后一层，最后一个时刻
                hidden_last_R = hidden_last[-1]
                # print("hidden_last_R.shape", hidden_last_R.size())      [16, 768]
                # 进行拼接
                hidden_last_out = torch.cat([hidden_last_L, hidden_last_R], dim=-1)
                # print('hidden_last_out', hidden_last_out.size())    [16, 1536]
            else:
                hidden_last_out = hidden_last[-1]
            # dropout and fully-connected layer
            out = self.dropout(hidden_last_out)
            # print("out.shape", out.size())      [16, 1536]
            rel_pred = self.rel_judgement(out)
            # print("rel_pred.shape", rel_pred.size())    [16, 44]
            """

            # before fuse relation representation
        if ensure_corres:
            # for every position $i$ in sequence, should concate $j$ to predict.
            sub_extend = sequence_output.unsqueeze(2).expand(-1, -1, seq_len, -1)  # (bs, s, s, h)
            obj_extend = sequence_output.unsqueeze(1).expand(-1, seq_len, -1, -1)  # (bs, s, s, h)
            # batch x seq_len x seq_len x 2*hidden
            corres_pred = torch.cat([sub_extend, obj_extend], 3)
            # (bs, seq_len, seq_len)
            corres_pred = self.global_corres(corres_pred).squeeze(-1)
            mask_tmp1 = attention_mask.unsqueeze(-1)
            mask_tmp2 = attention_mask.unsqueeze(1)
            corres_mask = mask_tmp1 * mask_tmp2

        # relation predict and data construction in inference stage
        xi, pred_rels = None, None
        if ensure_rel and seq_tags is None:
            # (bs, rel_num)
            # 使用torch.sigmoid对rel_pred进行sigmoid操作，并根据大于rel_threshold的结果生成一个独热编码的张量rel_pred_onehot。
            # 如果rel_pred大于rel_threshold，则对应位置为1，否则为0
            rel_pred_onehot = torch.where(torch.sigmoid(rel_pred) > rel_threshold,
                                          torch.ones(rel_pred.size(), device=rel_pred.device),
                                          torch.zeros(rel_pred.size(), device=rel_pred.device))
            # if potential relation is null
            # 如果潜在关系中没有任何关系被预测出来
            for idx, sample in enumerate(rel_pred_onehot):
                if 1 not in sample:
                    # (rel_num,)
                    # 找到rel_pred中概率最大的类别的索引max_index
                    max_index = torch.argmax(rel_pred[idx])
                    # 将对应位置的值设为1，表示该类别被预测为关系
                    sample[max_index] = 1
                    # 更新rel_pred_onehot中对应样本的预测结果
                    rel_pred_onehot[idx] = sample
            # 2*(sum(x_i),)
            # 使用torch.nonzero找到rel_pred_onehot中非零元素的索引，返回两个张量bs_idxs和pred_rels。
            # bs_idxs表示本批次内样本的索引，pred_rels表示该样本预测的关系索引
            bs_idxs, pred_rels = torch.nonzero(rel_pred_onehot, as_tuple=True)
            # get x_i
            # 计算每个样本的x_i值，即样本中预测的关系数量
            xi_dict = Counter(bs_idxs.tolist())
            xi = [xi_dict[idx] for idx in range(bs)]

            pos_seq_output = []
            pos_potential_rel = []
            pos_attention_mask = []
            # 根据bs_idxs和pred_rels，从sequence_output、attention_mask和potential_rels中提取出与预测关系对应的序列输出、注意力掩码和潜在关系索引
            for bs_idx, rel_idx in zip(bs_idxs, pred_rels):
                # (seq_len, h)
                pos_seq_output.append(sequence_output[bs_idx])
                pos_attention_mask.append(attention_mask[bs_idx])
                pos_potential_rel.append(rel_idx)
            # (sum(x_i), seq_len, h)
            sequence_output = torch.stack(pos_seq_output, dim=0)
            # (sum(x_i), seq_len)
            attention_mask = torch.stack(pos_attention_mask, dim=0)
            # (sum(x_i),)
            potential_rels = torch.stack(pos_potential_rel, dim=0)
        # ablation of relation judgement
        elif not ensure_rel and seq_tags is None:
            # construct test data
            # 将sequence_output重复（repeat）self.rel_num次，并改变形状（view）为(bs * self.rel_num, seq_len, h)
            sequence_output = sequence_output.repeat((1, self.rel_num, 1)).view(bs * self.rel_num, seq_len, h)
            attention_mask = attention_mask.repeat((1, self.rel_num)).view(bs * self.rel_num, seq_len)
            # 在设备（device）上创建一个张量，范围从0到self.rel_num，并将其重复（repeat）bs次。这个张量表示潜在关系的索引
            potential_rels = torch.arange(0, self.rel_num, device=input_ids.device).repeat(bs)

        # (bs/sum(x_i), h)
        # 将potential_rels作为输入传递给self.rel_embedding模型，以获取关系的嵌入向量
        rel_emb = self.rel_embedding(potential_rels)

        # relation embedding vector fusion
        # 对rel_emb进行维度扩展，将其维度从(bs/sum(x_i), h)扩展为(bs/sum(x_i), seq_len, h)
        rel_emb = rel_emb.unsqueeze(1).expand(-1, seq_len, h)
        if ex_params['emb_fusion'] == 'concat':
            """
            # (bs/sum(x_i), seq_len, 2*h)
            # 将sequence_output和rel_emb按照最后一个维度进行拼接
            decode_input = torch.cat([sequence_output, rel_emb], dim=-1)
            # (bs/sum(x_i), seq_len, tag_size)
            # 将decode_input作为输入传递给self.sequence_tagging_sub和sequence_tagging_obj模型，以进行序列标注，得到output_sub和output_obj
            output_sub = self.sequence_tagging_sub(decode_input)
            output_obj = self.sequence_tagging_obj(decode_input)
            """

            # """
            # (bs/sum(x_i), seq_len, 2*h)
            # 将sequence_output和rel_emb按照最后一个维度进行拼接
            decode_input = torch.cat([sequence_output, rel_emb], dim=-1)
            lstm_output_sub, (hidden_last_sub, cn_last_sub) = self.lstm_sub(decode_input)
            # print("lstm_output_sub.shape", lstm_output_sub.size())          # [16, 128, 1536 * 2]
            # print("hidden_last_sub.shape", hidden_last_sub.size())          # [2, 16, 1536]
            # print("cn_last_sub.shape", cn_last_sub.size())                  # [2, 16, 1536]
            # dropout and fully-connected layer
            out_sub = self.dropout(lstm_output_sub)
            # print("out_sub.shape", out_sub.size())                  # [16, 128, 1536 * 2]
            out_sub = self.linear(out_sub)
            # print("out_sub.shape", out_sub.size())                  # [16, 128, 1536]
            out_sub = nn.ReLU()(out_sub)
            out_sub = self.dropout(out_sub)
            output_sub = self.sequence_tagging_sub(out_sub)         # [16, 128, 3]
            # print("output_sub.shape", output_sub.size())

            lstm_output_obj, (hidden_last_obj, cn_last_obj) = self.lstm_obj(decode_input)
            # dropout and fully-connected layer
            out_obj = self.dropout(lstm_output_obj)
            out_obj = self.linear(out_obj)
            out_obj = nn.ReLU()(out_obj)
            out_obj = self.dropout(out_obj)
            output_obj = self.sequence_tagging_obj(out_obj)
            # """

        elif ex_params['emb_fusion'] == 'sum':
            # (bs/sum(x_i), seq_len, h)
            decode_input = sequence_output + rel_emb
            # (bs/sum(x_i), seq_len, tag_size)
            output_sub, output_obj = self.sequence_tagging_sum(decode_input)

        # train
        if seq_tags is not None:
            # calculate loss
            # 将attention_mask变量的形状改变为一维
            attention_mask = attention_mask.view(-1)
            # sequence label loss
            # 定义一个交叉熵损失函数，其中reduction='none'表示不进行降维
            loss_func = nn.CrossEntropyLoss(reduction='none')
            # 计算主体序列标注的损失，首先将output_sub和seq_tags的形状调整为二维，然后使用交叉熵损失函数计算损失，乘以attention_mask进行掩码，最后求和并除以attention_mask的和
            loss_seq_sub = (loss_func(output_sub.view(-1, self.seq_tag_size), seq_tags[:, 0, :].reshape(-1)) * attention_mask).sum() / attention_mask.sum()
            loss_seq_obj = (loss_func(output_obj.view(-1, self.seq_tag_size), seq_tags[:, 1, :].reshape(-1)) * attention_mask).sum() / attention_mask.sum()

            # loss_seq_sub = (loss_func(output_sub, seq_tags[:, 0, :].reshape(-1)) * attention_mask).sum() / attention_mask.sum()
            # loss_seq_obj = (loss_func(output_obj, seq_tags[:, 1, :].reshape(-1)) * attention_mask).sum() / attention_mask.sum()

            # 将主体和客体序列标注的损失求平均，得到总的序列标注损失
            loss_seq = (loss_seq_sub + loss_seq_obj) / 2
            # init
            loss_matrix, loss_rel = torch.tensor(0), torch.tensor(0)
            if ensure_corres:
                # 将变量的形状调整为二维，行数为bs
                corres_pred = corres_pred.view(bs, -1)
                corres_mask = corres_mask.view(bs, -1)
                corres_tags = corres_tags.view(bs, -1)
                loss_func = nn.BCEWithLogitsLoss(reduction='none')
                loss_matrix = (loss_func(corres_pred,
                                         corres_tags.float()) * corres_mask).sum() / corres_mask.sum()

            if ensure_rel:
                loss_func = nn.BCEWithLogitsLoss(reduction='mean')
                loss_rel = loss_func(rel_pred, rel_tags.float())

            loss = loss_seq + loss_matrix + loss_rel
            return loss, loss_seq, loss_matrix, loss_rel, rel_emb

        # inference
        else:
            # (sum(x_i), seq_len)
            # 使用torch.softmax对output_sub进行softmax操作，并使用torch.argmax找到每个位置上概率最大的类别，返回一个张量pred_seq_sub
            pred_seq_sub = torch.argmax(torch.softmax(output_sub, dim=-1), dim=-1)
            pred_seq_obj = torch.argmax(torch.softmax(output_obj, dim=-1), dim=-1)
            # (sum(x_i), 2, seq_len)
            # 使用torch.cat将pred_seq_sub.unsqueeze(1)和pred_seq_obj.unsqueeze(1)在维度1上进行拼接，形成一个新的张量pred_seqs。
            # 这个张量的形状为(sum(x_i), 2, seq_len)，其中sum(x_i)表示所有样本的序列长度之和
            pred_seqs = torch.cat([pred_seq_sub.unsqueeze(1), pred_seq_obj.unsqueeze(1)], dim=1)
            if ensure_corres:
                # 将corres_pred使用torch.sigmoid进行sigmoid操作，并与corres_mask相乘。corres_pred是一个表示关系对应度的张量，corres_mask是一个用于掩码操作的张量
                corres_pred = torch.sigmoid(corres_pred) * corres_mask
                # (bs, seq_len, seq_len)】
                # 使用torch.where根据corres_pred是否大于corres_threshold来生成一个独热编码的张量pred_corres_onehot。
                # 如果corres_pred大于corres_threshold，则对应位置为1，否则为0。这个张量的形状为(bs, seq_len, seq_len)，其中bs表示批量大小
                pred_corres_onehot = torch.where(corres_pred > corres_threshold,
                                                 torch.ones(corres_pred.size(), device=corres_pred.device),
                                                 torch.zeros(corres_pred.size(), device=corres_pred.device))
                return pred_seqs, pred_corres_onehot, xi, pred_rels
            return pred_seqs, xi, pred_rels

class FGM():
    def __init__(self, model_, rel_emb):
        self.model = model_
        self.backup = {}
        self.emb = rel_emb

    def attack(self, epsilon=1., emb_name='emb'):
        # emb_name这个参数要换成你模型中embedding的参数名
        # 例如，self.emb = nn.Embedding(5000, 100)
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)  # 默认为2范数
                if norm != 0:
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name='emb'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


class PGD():
    def __init__(self, model, rel_emb):
        self.model = model
        self.emb_backup = {}
        self.grad_backup = {}
        self.emb = rel_emb

    def attack(self, epsilon=1., alpha=0.3, emb_name='emb', is_first_attack=False):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                if is_first_attack:
                    self.emb_backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0:
                    r_at = alpha * param.grad / norm
                    param.data.add_(r_at)
                    param.data = self.project(name, param.data, epsilon)

    def restore(self, emb_name='emb'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.emb_backup
                param.data = self.emb_backup[name]
        self.emb_backup = {}

    def project(self, param_name, param_data, epsilon):
        r = param_data - self.emb_backup[param_name]
        if torch.norm(r) > epsilon:
            r = epsilon * r / torch.norm(r)
        return self.emb_backup[param_name] + r

    def backup_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.grad_backup[name] = param.grad.clone()

    def restore_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.grad = self.grad_backup[name]



if __name__ == '__main__':
    from transformers import BertConfig
    import utils
    import os

    params = utils.Params()
    # Prepare model
    bert_config = BertConfig.from_json_file(os.path.join(params.bert_model_dir, 'config.json'))
    model = BertForRE.from_pretrained(config=bert_config,
                                      pretrained_model_name_or_path=params.bert_model_dir,
                                      params=params)
    model.to(params.device)

    for n, _ in model.named_parameters():
        print(n)
