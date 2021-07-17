# _*_ coding:utf-8 _*_


"""
适用于中文BERT,RoBERTa

"""


import warnings

import numpy as np
import torch
import torch.nn as nn
from transformers import BertModel
from transformers import BertPreTrainedModel

from layers.encoders.transformers.bert.layernorm import ConditionalLayerNorm
from utils.data_util import batch_gather

warnings.filterwarnings("ignore")


class ERENet(BertPreTrainedModel):
    """
    ERENet : entity relation jointed extraction
    """

    def __init__(self, config, classes_num,subject_class_num):
        super(ERENet, self).__init__(config, classes_num,subject_class_num)

        print('spo_transformers')
        self.classes_num = classes_num
        self.subject_class_num=subject_class_num
        # BERT model
        self.bert = BertModel(config)
        self.subject_type_embedding = nn.Embedding(num_embeddings=subject_class_num, embedding_dim=256,padding_idx=0)
        self.LayerNorm = ConditionalLayerNorm(config.hidden_size*2+256,config.hidden_size, eps=config.layer_norm_eps)
        # pointer net work
        self.po_dense = nn.Linear(config.hidden_size, self.classes_num * 2)
        self.subject_dense = nn.Linear(config.hidden_size, self.subject_class_num*2)
        self.loss_fct = nn.BCEWithLogitsLoss(reduction='none')

        self.init_weights()

    def forward(self, q_ids=None, passage_ids=None, segment_ids=None, token_type_ids=None, subject_ids=None,
                subject_labels=None,
                object_labels=None, eval_file=None,
                is_eval=False):
        mask = (passage_ids != 0).float()
        bert_encoder = self.bert(passage_ids, token_type_ids=segment_ids, attention_mask=mask)[0]
        if not is_eval:
            a=subject_ids[:, 1]
            sub_start_encoder = batch_gather(bert_encoder, subject_ids[:, 0])
            sub_end_encoder = batch_gather(bert_encoder, subject_ids[:, 1])
            subject_item = torch.cat([sub_start_encoder, sub_end_encoder], 1)
            subject_type_em=self.subject_type_embedding(subject_ids[:,2])
            subject = torch.cat([subject_item,subject_type_em],1)
            #bert_encoder:[batch_size,len(token_ids),embedding_size]
            #context_encoder :[batch_size,len(token_ids),embedding_size]
            context_encoder = self.LayerNorm(bert_encoder, subject)
            #sub_preds:[batch_size,len(token_ids),2] 但是事实上你要把它变成
            #[batch_size,len(token_ids),len(subject_type),2] 其中2这个纬度表示subject 的起始位置这
            sub_preds = self.subject_dense(bert_encoder).reshape(passage_ids.size(0),-1,self.subject_class_num,2)

            po_preds = self.po_dense(context_encoder).reshape(passage_ids.size(0), -1, self.classes_num, 2)
            # a=torch.where(subject_labels[0][:,:,0]>0.5)
            subject_loss = self.loss_fct(sub_preds, subject_labels)
            subject_loss = torch.sum(subject_loss.mean(3),2)
            subject_loss = torch.sum(subject_loss * mask.float()) / torch.sum(mask.float())

            po_loss = self.loss_fct(po_preds, object_labels)
            po_loss = torch.sum(po_loss.mean(3), 2)
            po_loss = torch.sum(po_loss * mask.float()) / torch.sum(mask.float())

            loss = subject_loss + po_loss

            return loss

        else:
            subject_preds = nn.Sigmoid()(self.subject_dense(bert_encoder).reshape(passage_ids.size(0),-1,self.subject_class_num,2))
            answer_list = list()
            for qid, sub_pred in zip(q_ids.cpu().numpy(),
                                     subject_preds.cpu().numpy()):
                context = eval_file[qid].bert_tokens
                subject_label_start = np.where(sub_pred[:,:,0] > 0.5)
                subject_label_end = np.where(sub_pred[:,:, 1] > 0.5)
                start,start_label=subject_label_start[0],subject_label_start[1]
                end,end_label=subject_label_end[0],subject_label_end[1]
                subjects = []
                for i in range(len(start)):
                    #j存的是在end中的index,指的是
                    j = np.where(end>=start[i])[0]
                    # j = end[end >= i]
                    if start[i] == 0 or start[i] > len(context) - 2:
                        continue
                    k=0
                    #这种策略事实上是值得商榷的
                    if len(j) > 0:
                        while k<len(j):
                            if end[j[k]] > len(context) - 2:
                                break
                            end_index=j[k]
                            if end_label[end_index]==start_label[i]:
                                subjects.append((start[i],end[j[k]],start_label[i]))
                                break
                            else:
                                k+=1
                answer_list.append(subjects)
            qid_ids, bert_encoders, pass_ids, subject_ids, token_type_ids = [], [], [], [], []
            for i, subjects in enumerate(answer_list):
                if subjects:
                    qid = q_ids[i].unsqueeze(0).expand(len(subjects))
                    pass_tensor = passage_ids[i, :].unsqueeze(0).expand(len(subjects), passage_ids.size(1))
                    new_bert_encoder = bert_encoder[i, :, :].unsqueeze(0).expand(len(subjects), bert_encoder.size(1),
                                                                                 bert_encoder.size(2))

                    token_type_id = torch.zeros((len(subjects), passage_ids.size(1)), dtype=torch.long)
                    for index, (start,end,subject_type) in enumerate(subjects):
                        token_type_id[index, start:end + 1] = 1

                    qid_ids.append(qid)
                    pass_ids.append(pass_tensor)
                    subject_ids.append(torch.tensor(subjects, dtype=torch.long))
                    bert_encoders.append(new_bert_encoder)
                    token_type_ids.append(token_type_id)

            if len(qid_ids) == 0:
                subject_ids = torch.zeros(1, 2).long().to(bert_encoder.device)
                qid_tensor = torch.tensor([-1], dtype=torch.long).to(bert_encoder.device)
                po_tensor = torch.zeros(1, bert_encoder.size(1)).long().to(bert_encoder.device)
                return qid_tensor, subject_ids, po_tensor

            qids = torch.cat(qid_ids).to(bert_encoder.device)
            pass_ids = torch.cat(pass_ids).to(bert_encoder.device)
            bert_encoders = torch.cat(bert_encoders).to(bert_encoder.device)
            subject_ids = torch.cat(subject_ids).to(bert_encoder.device)

            flag = False
            split_heads = 1024

            bert_encoders_ = torch.split(bert_encoders, split_heads, dim=0)
            pass_ids_ = torch.split(pass_ids, split_heads, dim=0)
            subject_encoder_ = torch.split(subject_ids, split_heads, dim=0)

            po_preds = list()
            for i in range(len(bert_encoders_)):
                bert_encoders = bert_encoders_[i]
                pass_ids = pass_ids_[i]
                subject_encoder = subject_encoder_[i]

                if bert_encoders.size(0) == 1:
                    flag = True
                    bert_encoders = bert_encoders.expand(2, bert_encoders.size(1), bert_encoders.size(2))
                    subject_encoder = subject_encoder.expand(2, subject_encoder.size(1))
                sub_start_encoder = batch_gather(bert_encoders, subject_encoder[:, 0])
                sub_end_encoder = batch_gather(bert_encoders, subject_encoder[:, 1])
                subject_item = torch.cat([sub_start_encoder, sub_end_encoder], 1)

                subject_type_em=self.subject_type_embedding(subject_encoder[:, 2])
                subject = torch.cat([subject_item,subject_type_em],1)
                context_encoder = self.LayerNorm(bert_encoders, subject)

                po_pred = self.po_dense(context_encoder).reshape(subject_encoder.size(0), -1, self.classes_num, 2)

                if flag:
                    po_pred = po_pred[1, :, :, :].unsqueeze(0)

                po_preds.append(po_pred)

            po_tensor = torch.cat(po_preds).to(qids.device)
            po_tensor = nn.Sigmoid()(po_tensor)
            return qids, subject_ids, po_tensor
