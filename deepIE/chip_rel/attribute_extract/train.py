# _*_ coding:utf-8 _*_
import codecs
import json
import logging
import sys
import time
from warnings import simplefilter

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

import models.attribute_net.attribute_net as attr_net
from layers.encoders.transformers.bert.bert_optimization import BertAdam

simplefilter(action='ignore', category=FutureWarning)
logger = logging.getLogger(__name__)


class Trainer(object):

    def __init__(self, args, data_loaders, examples, spo_conf, tokenizer):

        self.args = args
        self.tokenizer = tokenizer
        self.device = torch.device("cuda:{}".format(args.device_id) if torch.cuda.is_available() else "cpu")
        self.n_gpu = torch.cuda.device_count()

        self.id2rel = {item: key for key, item in spo_conf.items()}
        self.rel2id = spo_conf

        if self.n_gpu > 0:
            torch.cuda.manual_seed_all(args.seed)
        self.model = attr_net.AttributeExtractNet.from_pretrained(args.bert_model, classes_num=len(spo_conf))

        self.model.to(self.device)
        if args.train_mode != "train":
            self.resume(args)
        logging.info('total gpu num is {}'.format(self.n_gpu))
        if self.n_gpu > 1:
            self.model = nn.DataParallel(self.model.cuda(), device_ids=[0, 1])

        train_dataloader, dev_dataloader, test_dataloader = data_loaders
        train_eval, dev_eval, test_eval = examples
        self.eval_file_choice = {
            "train": train_eval,
            "dev": dev_eval,
            "test": test_eval
        }
        self.data_loader_choice = {
            "train": train_dataloader,
            "dev": dev_dataloader,
            "test": test_dataloader
        }
        # todo 稍后要改成新的优化器，并加入梯度截断
        self.optimizer = self.set_optimizer(args, self.model,
                                            train_steps=(int(
                                                len(train_eval) / args.train_batch_size) + 1) * args.epoch_num)

    def set_optimizer(self, args, model, train_steps=None):
        param_optimizer = list(model.named_parameters())
        param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=args.learning_rate,
                             warmup=args.warmup_proportion,
                             t_total=train_steps)
        return optimizer

    def train(self, args):

        best_f1 = 0.0
        patience_stop = 0
        self.model.train()
        step_gap = 20
        for epoch in range(int(args.epoch_num)):

            global_loss = 0.0

            for step, batch in tqdm(enumerate(self.data_loader_choice[u"train"]), mininterval=5,
                                    desc=u'training at epoch : %d ' % epoch, leave=False, file=sys.stdout):

                loss = self.forward(batch)

                if step % step_gap == 0:
                    global_loss += loss
                    current_loss = global_loss / step_gap
                    print(
                        u"step {} / {} of epoch {}, train/loss: {}".format(step, len(self.data_loader_choice["train"]),
                                                                           epoch, current_loss))
                    global_loss = 0.0

            res_dev = self.eval_data_set("dev")
            if res_dev['f1'] >= best_f1:
                best_f1 = res_dev['f1']
                logging.info("** ** * Saving fine-tuned model ** ** * ")
                model_to_save = self.model.module if hasattr(self.model,
                                                             'module') else self.model  # Only save the model it-self
                output_model_file = args.output + "/pytorch_model.bin"
                torch.save(model_to_save.state_dict(), str(output_model_file))
                patience_stop = 0
            else:
                patience_stop += 1
            if patience_stop >= args.patience_stop:
                return

    def resume(self, args):
        resume_model_file = args.output + "/pytorch_model.bin"
        logging.info("=> loading checkpoint '{}'".format(resume_model_file))
        checkpoint = torch.load(resume_model_file, map_location='cpu')
        self.model.load_state_dict(checkpoint)

    def forward(self, batch, chosen=u'train', eval=False, answer_dict=None):

        batch = tuple(t.to(self.device) for t in batch)
        if not eval:
            input_ids, token_type_ids, segment_ids, object_labels = batch
            loss = self.model(passage_id=input_ids, token_type_id=token_type_ids, segment_id=segment_ids,
                              object_labels=object_labels)
            if self.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu.

            loss.backward()
            loss = loss.item()
            self.optimizer.step()
            self.optimizer.zero_grad()
            return loss
        else:
            p_ids, input_ids, token_type_ids, segment_ids = batch
            eval_file = self.eval_file_choice[chosen]
            po_pred = self.model(passage_id=input_ids,
                                 token_type_id=token_type_ids,
                                 segment_id=segment_ids,
                                 is_eval=eval)
            ans_dict = self.convert_spo_contour(p_ids, po_pred, eval_file,
                                                answer_dict)
            return ans_dict

    def eval_data_set(self, chosen="dev"):

        self.model.eval()

        data_loader = self.data_loader_choice[chosen]
        eval_file = self.eval_file_choice[chosen]
        answer_dict = {i: [[], []] for i in range(len(eval_file))}

        last_time = time.time()
        with torch.no_grad():
            for _, batch in tqdm(enumerate(data_loader), mininterval=5, leave=False, file=sys.stdout):
                self.forward(batch, chosen, eval=True, answer_dict=answer_dict)
        used_time = time.time() - last_time
        logging.info('chosen {} took : {} sec'.format(chosen, used_time))

        # self.convert2result(eval_file, answer_dict)

        res = self.evaluate(eval_file, answer_dict, chosen)
        self.model.train()
        return res

    def predict_data_set(self, chosen="dev"):

        self.model.eval()

        data_loader = self.data_loader_choice[chosen]
        eval_file = self.eval_file_choice[chosen]
        answer_dict = {i: [[], []] for i in range(len(eval_file))}

        last_time = time.time()
        with torch.no_grad():
            for _, batch in tqdm(enumerate(data_loader), mininterval=5, leave=False, file=sys.stdout):
                self.forward(batch, chosen, eval=True, answer_dict=answer_dict)
        used_time = time.time() - last_time
        logging.info('chosen {} took : {} sec'.format(chosen, used_time))

        # self.convert2result(eval_file, answer_dict)

        with codecs.open(self.args.res_path, 'w', 'utf-8') as f:
            for key, ans_list in answer_dict.items():
                out_put = {}
                out_put['text'] = eval_file[int(key)].raw_text
                spo_tuple_lst = ans_list[1]
                spo_lst = []
                for (s, p, o) in spo_tuple_lst:
                    spo_lst.append({"predicate": p, "subject": s, "object": {"@value": o}})
                out_put['spo_list'] = spo_lst

                json_str = json.dumps(out_put, ensure_ascii=False)
                f.write(json_str)
                f.write('\n')

    def evaluate(self, eval_file, answer_dict, chosen):

        ent_em, ent_pred_num, ent_gold_num = 0.0, 0.0, 0.0
        spo_em, spo_pred_num, spo_gold_num = 0.0, 0.0, 0.0

        for key in answer_dict.keys():
            context = eval_file[key].context

            triple_pred = answer_dict[key][1]
            triple_gold = answer_dict[key][0]

            # if set(triple_pred) != set(triple_gold):
            #     print()
            #     print(context)
            #     print(triple_pred)
            #     print(triple_gold)

            spo_em += len(set(triple_pred) & set(triple_gold))
            spo_pred_num += len(set(triple_pred))
            spo_gold_num += len(set(triple_gold))

        p = spo_em / spo_pred_num if spo_pred_num != 0 else 0
        r = spo_em / spo_gold_num if spo_gold_num != 0 else 0
        f = 2 * p * r / (p + r) if p + r != 0 else 0

        ent_precision = 100.0 * ent_em / ent_pred_num if ent_pred_num > 0 else 0.
        ent_recall = 100.0 * ent_em / ent_gold_num if ent_gold_num > 0 else 0.
        ent_f1 = 2 * ent_recall * ent_precision / (ent_recall + ent_precision) if (
                                                                                          ent_recall + ent_precision) != 0 else 0.0

        print('============================================')
        print("{}/entity_em: {},\tentity_pred_num&entity_gold_num: {}\t{} ".format(chosen, ent_em, ent_pred_num,
                                                                                   ent_gold_num))
        print(
            "{}/entity_f1: {}, \tentity_precision: {},\tentity_recall: {} ".format(chosen, ent_f1, ent_precision,
                                                                                   ent_recall))
        print('============================================')
        print("{}/em: {},\tpre&gold: {}\t{} ".format(chosen, spo_em, spo_pred_num, spo_gold_num))
        print("{}/f1: {}, \tPrecision: {},\tRecall: {} ".format(chosen, f * 100, p * 100,
                                                                r * 100))
        return {'f1': f, "recall": r, "precision": p}

    def convert2result(self, eval_file, answer_dict):
        for qid in answer_dict.keys():
            spoes = answer_dict[qid][2]

            context = eval_file[qid].context
            tok_to_orig_start_index = eval_file[qid].tok_to_orig_start_index
            tok_to_orig_end_index = eval_file[qid].tok_to_orig_end_index

            po_predict = []
            for s, po in spoes.items():
                po.sort(key=lambda x: x[2])
                sub_ent = context[tok_to_orig_start_index[s[0] - 1]:tok_to_orig_end_index[s[1] - 1] + 1]
                for (o1, o2, p) in po:
                    obj_ent = context[tok_to_orig_start_index[o1 - 1]:tok_to_orig_end_index[o2 - 1] + 1]
                    predicate = self.id2rel[p]

                    # TODO:到时候选择
                    # if sub_ent.replace(' ','') in context:
                    #     sub_ent = sub_ent.replace(' ', '')
                    # if obj_ent.replace(' ','') in context:
                    #     obj_ent = obj_ent.replace(' ', '')
                    po_predict.append((sub_ent, predicate, obj_ent))
            answer_dict[qid][1].extend(po_predict)

    def convert_spo_contour(self, qids, po_preds, eval_file, answer_dict):

        for qid, po_pred in zip(qids.data.cpu().numpy(), po_preds.data.cpu().numpy()):
            example = eval_file[qid.item()]

            text_id = example.text_id
            tokens = example.bert_tokens
            gold_spo=example.gold_answer
            context = example.context
            sub_index = example.choice_sub
            sub_name = context[sub_index[0]:sub_index[1] + 1]
            start = np.where(po_pred[:, :, 0] > 0.5)
            end = np.where(po_pred[:, :, 1] > 0.5)

            po_index_lst = []

            for _start, predicate1 in zip(*start):
                if _start > len(tokens) - 2 or _start == 0:
                    continue
                for _end, predicate2 in zip(*end):
                    if _start <= _end <= len(tokens) - 2 and predicate1 == predicate2:
                        po_index_lst.append((_start, _end, predicate1))
                        break
            spo_lst = []
            for po in po_index_lst:
                start, end, p = po
                obj_name = context[start - 1:end]
                predicate = self.id2rel[p]
                spo_lst.append((sub_name, predicate, obj_name))

            if text_id not in answer_dict:
                raise ValueError('text_id error in answer_dict ')
            else:
                answer_dict[text_id][1].extend(spo_lst)
                if len(answer_dict[text_id][0])>1:
                    continue
                gold_spo=[(gold[0],gold[1],gold[2]) for gold in gold_spo]
                answer_dict[text_id][0].extend(gold_spo)

