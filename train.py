import time
import datetime
import json
import logging
import os
import pytorch_lightning as pl
import random
import sys
import torch
from argparse import ArgumentParser
from collections import Counter, defaultdict
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, AdamW, get_linear_schedule_with_warmup
from utils import episode_collate_fn as collate_fn
from model import FewShotBaseModel as BaseModel
from model import SpanEncoder, DotAttention, AvgAttention, TokenEncoder
from utils import truecase_episode_loader, sent_loader, get_io_spans, LabelEncoder, \
    EpisodeDataset, EpisodeSamplingDataset, EpisodeSpanDataset


class ProtoSpan(BaseModel):
    def __init__(self, backbone_name, pretrained_encoder, reinit, types, span_rep_dim, dropout, dropout_pos,
                 span_output,  max_span_len, use_atten, lr, bert_lr, wd, train_step, seed, metric='L2', cosine_s=20):
        super(ProtoSpan, self).__init__()
        self.save_hyperparameters()

        pl.seed_everything(self.hparams.seed)
        self.encoder = SpanEncoder(backbone_name, span_rep_dim, max_span_len, dropout, dropout_pos, span_output,
                                   pretrained_encoder, reinit)
        if self.hparams.use_atten:
            self.atten = DotAttention()
        else:
            self.atten = AvgAttention()
        self.loss_fn = nn.NLLLoss(ignore_index=-1)
        self.encode_time, self.comp_time = 0., 0.

    def forward(self, batch, token_embedding=None):
        if token_embedding is None:
            token_embedding = self.encoder(batch['token_id'], batch['atten_mask'])

        support_mask = (batch['sent_flag'] == 0).bool()
        query_mask = (batch['sent_flag'] == 1).bool()
        atten_mask = batch['atten_mask'].bool()
        word_mask = batch['word_mask'].bool()
        label = batch['label']
        global_label = batch['global_label']
        full_span = batch['full_span']
        subword_len = batch['subword_len']

        batch_id = range(batch['batch_size'])

        encode = []
        for bid in batch_id:
            batch_mask = batch['batch_id'] == bid

            batch_support_idx = (batch_mask & support_mask).nonzero(as_tuple=True)[0]
            support_emb = token_embedding.index_select(0, batch_support_idx)
            support_label = label.index_select(0, batch_support_idx)
            support_global_label = global_label.index_select(0, batch_support_idx)
            support_word_mask = word_mask.index_select(0, batch_support_idx)
            support_full_span = full_span.index_select(0, batch_support_idx)
            support_subword_len = subword_len.index_select(0, batch_support_idx)

            batch_query_idx = (batch_mask & query_mask).nonzero(as_tuple=True)[0]
            query_emb = token_embedding.index_select(0, batch_query_idx)
            query_label = label.index_select(0, batch_query_idx)
            query_global_label = global_label.index_select(0, batch_query_idx)
            query_word_mask = word_mask.index_select(0, batch_query_idx)
            query_full_span = full_span.index_select(0, batch_query_idx)
            query_subword_len = subword_len.index_select(0, batch_query_idx)

            support_sample_span_mask, support_full_span_rep = [], []
            support_full_span_global_label, support_full_span_label = [], []
            for seq_span_emb, seq_word_mask, seq_full_span in zip(support_emb, support_word_mask, support_full_span):
                seq_full_mask = seq_full_span[:, :, 0] >= 0
                gather_start, gather_end = torch.nonzero(seq_full_mask, as_tuple=True)
                seq_full_span_rep = self.encoder.span_extractor(seq_span_emb, seq_word_mask, gather_start, gather_end)
                seq_full_span = seq_full_span[seq_full_mask]
                seq_sample_span_mask = seq_full_span[:, 2].bool()
                seq_full_span_label = seq_full_span[:, 0]
                seq_full_span_global_label = seq_full_span[:, 1]

                support_sample_span_mask.append(seq_sample_span_mask)
                support_full_span_rep.append(seq_full_span_rep)
                support_full_span_label.append(seq_full_span_label)
                support_full_span_global_label.append(seq_full_span_global_label)

            support_sample_span_mask = torch.cat(support_sample_span_mask, 0)
            support_full_span_rep = torch.cat(support_full_span_rep, 0)
            support_full_span_label = torch.cat(support_full_span_label, 0)
            support_full_span_global_label = torch.cat(support_full_span_global_label, 0)

            query_sample_span_mask, query_full_span_rep = [], []
            query_full_span_label, query_full_span_global_label = [], []
            query_full_span_i, query_full_span_j, query_full_span_k = [], [], []
            query_full_span_i_mat = torch.arange(0, query_full_span.size(0)).view(-1, 1, 1).expand_as(
                query_full_span[:, :, :, 0])
            for seq_span_emb, seq_word_mask, seq_full_span, seq_full_span_i_mat in zip(query_emb, query_word_mask,
                                                                                       query_full_span,
                                                                                       query_full_span_i_mat):
                seq_full_mask = seq_full_span[:, :, 0] >= 0
                gather_start, gather_end = torch.nonzero(seq_full_mask, as_tuple=True)
                seq_full_span_rep = self.encoder.span_extractor(seq_span_emb, seq_word_mask, gather_start, gather_end)
                seq_full_span = seq_full_span[seq_full_mask]
                seq_sample_span_mask = seq_full_span[:, 2].bool()
                seq_full_span_label = seq_full_span[:, 0]
                seq_full_span_global_label = seq_full_span[:, 1]
                seq_full_span_i = seq_full_span_i_mat[seq_full_mask]
                seq_full_span_j = gather_start
                seq_full_span_k = gather_end

                query_sample_span_mask.append(seq_sample_span_mask)
                query_full_span_rep.append(seq_full_span_rep)
                query_full_span_label.append(seq_full_span_label)
                query_full_span_global_label.append(seq_full_span_global_label)
                query_full_span_i.append(seq_full_span_i)
                query_full_span_j.append(seq_full_span_j)
                query_full_span_k.append(seq_full_span_k)

            query_sample_span_mask = torch.cat(query_sample_span_mask, 0)
            query_full_span_rep = torch.cat(query_full_span_rep, 0)
            query_full_span_label = torch.cat(query_full_span_label, 0)
            query_full_span_global_label = torch.cat(query_full_span_global_label, 0)
            query_full_span_i = torch.cat(query_full_span_i, 0)
            query_full_span_j = torch.cat(query_full_span_j, 0)
            query_full_span_k = torch.cat(query_full_span_k, 0)

            support_sample_span_rep = support_full_span_rep[support_sample_span_mask]
            support_sample_span_label = support_full_span_label[support_sample_span_mask]
            support_sample_span_global_label = support_full_span_global_label[support_sample_span_mask]
            query_sample_span_rep = query_full_span_rep[query_sample_span_mask]
            query_sample_span_label = query_full_span_label[query_sample_span_mask]
            query_sample_span_global_label = query_full_span_global_label[query_sample_span_mask]

            encode.append({
                'support_label': support_label,
                'support_global_label': support_global_label,
                'support_word_mask': support_word_mask,
                'support_sample_span_label': support_sample_span_label,
                'support_full_span_label': support_full_span_label,
                'support_sample_span_global_label': support_sample_span_global_label,
                'support_full_span_global_label': support_full_span_global_label,
                'support_sample_span_rep': support_sample_span_rep,
                'support_full_span_rep': support_full_span_rep,
                'support_subword_len': support_subword_len,
                'query_label': query_label,
                'query_global_label': query_global_label,
                'query_word_mask': query_word_mask,
                'query_sample_span_label': query_sample_span_label,
                'query_full_span_label': query_full_span_label,
                'query_sample_span_global_label': query_sample_span_global_label,
                'query_full_span_global_label': query_full_span_global_label,
                'query_full_span_i': query_full_span_i,
                'query_full_span_j': query_full_span_j,
                'query_full_span_k': query_full_span_k,
                'query_sample_span_rep': query_sample_span_rep,
                'query_full_span_rep': query_full_span_rep,
                'query_subword_len': query_subword_len,
            })

        return encode

    def training_step(self, batch, batch_idx):
        assert self.training
        output = self(batch)
        batch_id = range(batch['batch_size'])
        loss_sims, loss_targets = [], []
        for bid in batch_id:
            full_support_span_rep = output[bid]['support_full_span_rep']
            full_support_span_label = output[bid]['support_full_span_label']
            full_support_span_global_label = output[bid]['support_full_span_global_label']

            full_query_span_rep = output[bid]['query_full_span_rep']
            full_query_span_label = output[bid]['query_full_span_label']
            full_query_span_global_label = output[bid]['query_full_span_global_label']

            sample_query_span_rep = output[bid]['query_sample_span_rep']
            sample_query_span_label = output[bid]['query_sample_span_label']
            sample_query_span_global_label = output[bid]['query_sample_span_global_label']

            sample_support_span_rep = output[bid]['support_sample_span_rep']
            sample_support_span_label = output[bid]['support_sample_span_label']
            sample_support_span_global_label = output[bid]['support_sample_span_global_label']

            query_sim = []
            tag_list = [(0, 'O')] + [(lid + 1, l) for lid, l in enumerate(batch['types'][bid])]
            for lid, l in tag_list:
                in_class_emb = full_support_span_rep[full_support_span_label == lid]
                atten_proto, atten_weight = self.atten(sample_query_span_rep, in_class_emb, in_class_emb)
                in_class_sim = -(sample_query_span_rep - atten_proto).pow(2).sum(1)
                query_sim.append(in_class_sim)

            query_sim = torch.stack(query_sim, dim=-1)
            query_logit = F.log_softmax(query_sim, dim=-1)

            loss_sims.append(query_logit)
            loss_targets.append(sample_query_span_label)

            if getattr(self, 'dual_loss', True):
                support_sim = []
                tag_list = [(0, 'O')] + [(lid + 1, l) for lid, l in enumerate(batch['types'][bid])]
                for lid, l in tag_list:
                    in_class_emb = full_query_span_rep[full_query_span_label == lid]
                    atten_proto, atten_weight = self.atten(sample_support_span_rep, in_class_emb, in_class_emb)
                    if self.hparams.metric == 'L2':
                        in_class_sim = -(sample_support_span_rep - atten_proto).pow(2).sum(1)
                    elif self.hparams.metric == 'cosine':
                        in_class_sim = F.normalize(sample_support_span_rep, dim=-1).matmul(
                            F.normalize(atten_proto, dim=-1)) * self.hparams.cosine_s
                    else:
                        raise Exception('Unknown metric')
                    support_sim.append(in_class_sim)

                support_sim = torch.stack(support_sim, dim=-1)
                support_logit = F.log_softmax(support_sim, dim=-1)

                loss_sims.append(support_logit)
                loss_targets.append(sample_support_span_label)

        loss_sims = torch.cat(loss_sims, 0)
        loss_targets = torch.cat(loss_targets, 0)
        loss = self.loss_fn(loss_sims, loss_targets)

        self.log('train/loss', loss, batch_size=batch['batch_size'])
        return loss

    # wired but not to break old model
    def set_dual_loss_mode(self):
        self.dual_loss = True

    def unset_dual_loss_mode(self):
        self.dual_loss = False

    def eval_step(self, batch, output=None):
        assert not self.training

        def decode(cand):
            def conflict_judge(line_x, line_y):
                if line_x[0] == line_y[0]:
                    return True
                if line_x[0] < line_y[0]:
                    if line_x[1] >= line_y[0]:
                        return True
                if line_x[0] > line_y[0]:
                    if line_x[0] <= line_y[1]:
                        return True
                return False

            filter_list = []
            for elem in sorted(cand, key=lambda x: -x[3]):
                flag = False
                current = (elem[0], elem[1])
                for prior in filter_list:
                    flag = conflict_judge(current, (prior[0], prior[1]))
                    if flag:
                        break
                if not flag:
                    filter_list.append(elem)

            return filter_list
        
        start_time = time.time()
        if output is None:
            output = self(batch)
        end_time = time.time()
        self.encode_time += end_time - start_time
        
        batch_id = range(batch['batch_size'])
        query_preds, query_target, query_global_target = [], [], []
        for bid in batch_id:
            start_time = time.time()
            query_label = output[bid]['query_label']
            query_global_label = output[bid]['query_global_label']
            query_subword_len = output[bid]['query_subword_len']
            query_word_mask = output[bid]['query_word_mask']

            full_support_span_rep = output[bid]['support_full_span_rep']
            full_support_span_label = output[bid]['support_full_span_label']
            full_query_span_rep = output[bid]['query_full_span_rep']
            full_query_span_label = output[bid]['query_full_span_label']
            full_query_span_i = output[bid]['query_full_span_i']
            full_query_span_j = output[bid]['query_full_span_j']
            full_query_span_k = output[bid]['query_full_span_k']

            query_sim = []
            tag_list = [(0, 'O')] + [(lid + 1, l) for lid, l in enumerate(batch['types'][bid])]
            for lid, l in tag_list:
                in_class_emb = full_support_span_rep[full_support_span_label == lid]
                atten_proto, atten_weight = self.atten(full_query_span_rep, in_class_emb, in_class_emb)
                if self.hparams.metric == 'L2':
                    in_class_sim = -(full_query_span_rep - atten_proto).pow(2).sum(1)
                elif self.hparams.metric == 'cosine':
                    in_class_sim = F.normalize(full_query_span_rep, dim=-1).matmul(
                        F.normalize(atten_proto, dim=-1)) * self.hparams.cosine_s
                else:
                    raise Exception('Unknown metric')
                query_sim.append(in_class_sim)

            query_sim = torch.stack(query_sim, dim=-1)
            query_logit = F.log_softmax(query_sim, dim=-1)
            pred_logit, pred_label = torch.max(query_logit, dim=-1)
            
            decoded_pred = torch.ones_like(query_label, device=torch.device('cpu')) * -1
            sent_spans = defaultdict(list)
            decoded_cand_buf = {}
            for logit, label, i, j, k in zip(pred_logit.tolist(), pred_label.tolist(),
                                             full_query_span_i.tolist(), full_query_span_j.tolist(),
                                             full_query_span_k.tolist()):
                if label != 0:
                    sent_spans[i].append((j, k, label, logit))

            for idx, subword_len in enumerate(query_subword_len.tolist()):
                decoded_pred[idx, 1:subword_len + 1] = 0
            for i, span_cand in sent_spans.items():
                decoded_cand = decode(span_cand)
                decoded_cand_buf[i] = decoded_cand
                for j, k, pred, logit in decoded_cand:
                    decoded_pred[i, j:k + 1] = pred
            query_word_label = query_label.masked_fill(~query_word_mask, -1)
            query_word_global_label = query_global_label.masked_fill(~query_word_mask, -1)
            # decoded_pred = decoded_pred.to(device=query_word_label.device)
            query_preds.append(decoded_pred)
            query_target.append(query_word_label)
            query_global_target.append(query_word_global_label)
            end_time = time.time()
            self.comp_time += end_time - start_time
            
        output = {
            'pred': query_preds, 'target': query_target, 'global_target': query_global_target,
            'types': batch['types'], 'query_id': batch['query_id']
        }
        batch_jsons = [json.loads(json_str) for json_str in batch['jsons']]
        batch_preds, batch_targets = self.assemble_sentence(output)
        return batch_preds, batch_targets, batch_jsons

    def configure_optimizers(self):
        all_parameters = list(self.named_parameters())
        bert_parameters = [(n, p) for n, p in all_parameters if n.startswith('encoder.bert')]
        else_parameters = [(n, p) for n, p in all_parameters if not n.startswith('encoder.bert')]

        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        grouped_param = [
            {'params': [p for n, p in bert_parameters if not any(nd in n for nd in no_decay)],
             'weight_decay': self.hparams.wd, 'lr': self.hparams.bert_lr},
            {'params': [p for n, p in bert_parameters if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0, 'lr': self.hparams.bert_lr},
            {'params': [p for n, p in else_parameters if not any(nd in n for nd in no_decay)],
             'weight_decay': self.hparams.wd, 'lr': self.hparams.lr},
            {'params': [p for n, p in else_parameters if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0, 'lr': self.hparams.lr},
        ]
        optimizer = AdamW(grouped_param, lr=self.hparams.lr)
        warmup_steps = int(self.hparams.train_step * 0.1)
        logging.info(f'total step: {self.hparams.train_step} warmup step: {warmup_steps}')
        scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, self.hparams.train_step)

        return [
            {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'step',
                    'frequency': 1,
                }
            }
        ]


class ProtoToken(BaseModel):
    def __init__(self, backbone_name, pretrained_encoder, reinit, types, token_rep_dim, dropout, dropout_pos,
                 token_output, use_atten, lr, bert_lr, wd, train_step, seed):
        super(ProtoToken, self).__init__()
        self.save_hyperparameters()

        pl.seed_everything(self.hparams.seed)
        self.encoder = TokenEncoder(backbone_name, token_rep_dim, dropout, dropout_pos, token_output,
                                    pretrained_encoder, reinit)
        if self.hparams.use_atten:
            self.atten = DotAttention()
        else:
            self.atten = AvgAttention()
        self.loss_fn = nn.NLLLoss(ignore_index=-1)

    def forward(self, batch):
        token_embedding = self.encoder(batch['id'], batch['atten_mask'])

        support_mask = (batch['sent_flag'] == 0).bool()
        query_mask = (batch['sent_flag'] == 1).bool()
        word_mask = batch['word_mask'].bool()
        label = batch['label']

        batch_id = range(batch['batch_size'])
        loss_sims = []
        loss_targets = []
        query_preds = []
        query_target = []
        for bid in batch_id:
            batch_mask = batch['batch_id'] == bid

            batch_support_idx = (batch_mask & support_mask).nonzero(as_tuple=True)[0]
            support_emb = token_embedding.index_select(0, batch_support_idx)
            support_label = label.index_select(0, batch_support_idx)
            support_word_mask = word_mask.index_select(0, batch_support_idx)

            batch_query_idx = (batch_mask & query_mask).nonzero(as_tuple=True)[0]
            query_emb = token_embedding.index_select(0, batch_query_idx)
            query_label = label.index_select(0, batch_query_idx)
            query_word_mask = word_mask.index_select(0, batch_query_idx)

            query_sim = []
            tag_list = [(0, 'O')] + [(lid + 1, l) for lid, l in enumerate(batch['types'][bid])]
            for lid, l in tag_list:
                in_class_emb = support_emb[(support_label == lid) & support_word_mask]
                atten_proto, atten_weight = self.atten(query_emb, in_class_emb, in_class_emb)
                in_class_sim = -(query_emb - atten_proto).pow(2).sum(-1)
                query_sim.append(in_class_sim)

            query_sim = torch.stack(query_sim, dim=-1)
            query_word_label = query_label.masked_fill(~query_word_mask, -1)

            loss_sims.append(query_sim.view(-1, query_sim.size(-1)))
            loss_targets.append(query_word_label.view(-1))

            _, pred = torch.max(query_sim, dim=2)
            query_preds.append(pred)
            query_target.append(query_word_label)


        loss_sims = torch.cat(loss_sims, 0)
        loss_targets = torch.cat(loss_targets, 0)
        loss = self.loss_fn(loss_sims, loss_targets)

        return {
            'loss': loss, 'pred': query_preds, 'target': query_target, 'types': batch['types'],
            'query_id': batch['query_id']
        }

    def training_step(self, batch, batch_idx):
        assert self.training
        output = self(batch)
        self.log('train/loss', output['loss'])
        return output['loss']

    def configure_optimizers(self):
        all_parameters = list(self.named_parameters())
        bert_parameters = [(n, p) for n, p in all_parameters if n.startswith('encoder.bert')]
        else_parameters = [(n, p) for n, p in all_parameters if not n.startswith('encoder.bert')]

        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        grouped_param = [
            {'params': [p for n, p in bert_parameters if not any(nd in n for nd in no_decay)],
             'weight_decay': self.hparams.wd, 'lr': self.hparams.bert_lr},
            {'params': [p for n, p in bert_parameters if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0, 'lr': self.hparams.bert_lr},
            {'params': [p for n, p in else_parameters if not any(nd in n for nd in no_decay)],
             'weight_decay': self.hparams.wd, 'lr': self.hparams.lr},
            {'params': [p for n, p in else_parameters if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0, 'lr': self.hparams.lr},
        ]
        optimizer = AdamW(grouped_param, lr=self.hparams.lr)
        warmup_steps = int(self.hparams.train_step * 0.1)
        scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, self.hparams.train_step)

        return [
            {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'step',
                    'frequency': 1,
                }
            }
        ]

if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--mode', default='inter',
                        help='training mode, must be in [inter, intra]')
    parser.add_argument('--N', default=5, type=int,
                        help='N way')
    parser.add_argument('--K', default=5, type=int,
                        help='K shot')
    parser.add_argument('--batch_size', default=1, type=int,
                        help='batch size')
    parser.add_argument('--grad_acc', default=1, type=int,
                        help='grad acc step')
    parser.add_argument('--train_step', default=12000, type=int,
                        help='num of step in training')
    parser.add_argument('--val_interval', default=1000, type=int,
                        help='num of interval between validation')
    parser.add_argument('--model', default='span',
                        help='model name, must be in [span, token]')
    parser.add_argument('--hidden', default=768, type=int,
                        help='hidden size')
    parser.add_argument('--dropout', default=0, type=float,
                        help='dropout ration for extractor')
    parser.add_argument('--dropout_pos', default='before_linear',
                        help='dropout positions')
    parser.add_argument('--span_output', default='layernorm',
                        help='span_layner positions')
    parser.add_argument('--backbone_name', default='bert-base-uncased',
                        help='bert/roberta encoder name')
    parser.add_argument('--pretrained_encoder', default='',
                        help='bert tagger encoder ckpt path')
    parser.add_argument('--reinit', action='store_true',
                        help='reinit span extractor even with pretrained encoder')
    parser.add_argument('--max_length', default=96, type=int,
                        help='max length')
    parser.add_argument('--neg_rate', default=5.6, type=float,
                        help='span negative sampling ration')
    parser.add_argument('--max_span_len', default=10, type=int,
                        help='maximum span length')
    parser.add_argument('--lr', default=5e-4, type=float,
                        help='learning rate')
    parser.add_argument('--bert_lr', default=2e-5, type=float,
                        help='learning rate')
    parser.add_argument('--wd', default=0, type=float,
                        help='weight decay rate')
    parser.add_argument('--seed', default=0, type=int,
                        help='random seed')
    parser.add_argument('--use_atten', action='store_true',
                        help='use atten to generate proto')
    parser.add_argument('--debug_only', action='store_true',
                        help='only debug')
    parser.add_argument('--output_dir_overwrite', action='store_true',
                        help='output to specified dir')
    parser.add_argument('--output_dir', default='',
                        help='output dir')
    parser.add_argument('--wandb_proj', default='FewNERD',
                        help='wandb project')
    parser.add_argument('--metric', default='L2',
                        help='similarity metric, must be in [L2, cosine]')
    parser.add_argument('--cosface_s', default=20, type=float,
                        help='cosface scale')
    args = parser.parse_args()

    if not args.debug_only:
        if not args.output_dir_overwrite:
            filename = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            if 'SLURM_JOB_ID' in os.environ:
                checkpoint_dir = os.path.join('checkpoint', f"{filename}-{os.environ['SLURM_JOB_ID']}")
            else:
                checkpoint_dir = os.path.join('checkpoint', filename)
            assert not os.path.isdir(checkpoint_dir), checkpoint_dir
            os.mkdir(checkpoint_dir)
        else:
            checkpoint_dir = args.output_dir
            assert not os.path.isdir(checkpoint_dir), checkpoint_dir
            os.mkdir(checkpoint_dir)

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s;%(levelname)s;%(message)s",
                                  "%Y-%m-%d %H:%M:%S")
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    root.addHandler(handler)

    if not args.debug_only:
        handler = logging.FileHandler(os.path.join(checkpoint_dir, f'shell_output.log'))
        handler.setFormatter(formatter)
        root.addHandler(handler)

    logging.info('## Cmd line ##')
    logging.info(f"python {' '.join(sys.argv)}")

    logging.info(
        f'model:{args.model}-{args.backbone_name} pretrained encoder:{args.pretrained_encoder} reinit:{args.reinit}'
        f' hidden:{args.hidden} dropout:{args.dropout}  dropout position:{args.dropout_pos}'
        f' span output:{args.span_output} use atten:{args.use_atten}  metric:{args.metric} cosine_s {args.cosface_s}')
    logging.info(f'fewshot:{args.mode}-{args.N}-{args.K}-{args.max_length}')
    logging.info(f'train:{args.train_step} val:{args.val_interval} bs:{args.batch_size} grad acc:{args.grad_acc}')
    if args.model == 'span':
        logging.info(f'span max len:{args.max_span_len} neg:{args.neg_rate}')
    logging.info(f'lr:{args.lr} bert lr:{args.bert_lr} wd:{args.wd}')
    logging.info(f'seed:{args.seed}')

    worker_num = min(os.cpu_count(), 4)
    pl.seed_everything(args.seed)
    train_dataset = EpisodeSamplingDataset(args.N, args.K, args.K, sent_loader(args.mode, 'train'), args.backbone_name,
                                           args.max_length)
    types = train_dataset.types
    if args.model == 'span':
        train_dataset = EpisodeSpanDataset(train_dataset, args.neg_rate, max_len=args.max_span_len,
                                           length_limited_full_span=True)

    effective_batch_size = args.batch_size * args.grad_acc
    assert len(train_dataset) >= args.train_step * effective_batch_size
    logging.info(f'train dataset:{args.mode}-{args.N}-{args.K} max_length:{args.max_length}')

    train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=worker_num,
                                   collate_fn=collate_fn, shuffle=True)
    logging.info(f'training loader:{len(train_data_loader)} batch:{args.batch_size} worker:{worker_num}')

    dev_dataset = EpisodeDataset(truecase_episode_loader(args.mode, 'dev', args.N, args.K), args.backbone_name,
                                 args.max_length)
    if args.model == 'span':
        dev_dataset = EpisodeSpanDataset(dev_dataset, -1, max_len=args.max_span_len, length_limited_full_span=True)
    logging.info(f'dev dataset:{args.mode}-{args.N}-{args.K} max_length:{args.max_length}')
    dev_data_loader = DataLoader(dev_dataset, batch_size=effective_batch_size, num_workers=worker_num,
                                 collate_fn=collate_fn)
    logging.info(f'dev loader:{len(dev_data_loader)} batch:{args.batch_size} worker:{worker_num}')

    test_dataset = EpisodeDataset(truecase_episode_loader(args.mode, 'test', args.N, args.K), args.backbone_name,
                                  args.max_length)
    if args.model == 'span':
        test_dataset = EpisodeSpanDataset(test_dataset, -1, max_len=args.max_span_len, length_limited_full_span=True)
    logging.info(f'test dataset:{args.mode}-{args.N}-{args.K} max_length:{args.max_length}')
    test_data_loader = DataLoader(test_dataset, batch_size=effective_batch_size, num_workers=worker_num,
                                  collate_fn=collate_fn)
    logging.info(f'test loader:{len(test_data_loader)} batch:{args.batch_size} worker:{worker_num}')

    if args.model == 'span':
        model = ProtoSpan(args.backbone_name, args.pretrained_encoder, args.reinit, types, args.hidden, args.dropout,
                          args.dropout_pos, args.span_output, args.max_span_len, args.use_atten,
                          args.lr, args.bert_lr, args.wd, args.train_step, args.seed, args.metric, args.cosface_s)
    elif args.model == 'token':
        model = ProtoToken(args.backbone_name, args.pretrained_encoder, args.reinit, types, args.hidden, args.dropout,
                          args.dropout_pos, args.span_output, args.use_atten,
                           args.lr, args.bert_lr, args.wd, args.train_step, args.seed)
    else:
        raise Exception('Unknown model type')

    if not args.debug_only:
        checkpoint_callback = pl.callbacks.model_checkpoint.ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename="{epoch}-{step}",
            monitor='valid/f1',
            mode='max',
        )
        lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='step')

        logger = pl.loggers.wandb.WandbLogger(project=args.wandb_proj)
        logger.experiment.config.update(args)
        logger.experiment.config['dir_timestamp'] = checkpoint_dir

        trainer = pl.Trainer(
            logger=logger,
            gpus=1,
            precision=16,
            max_steps=args.train_step,
            max_epochs=100,
            num_sanity_val_steps=0,
            val_check_interval=args.val_interval * args.grad_acc,
            accumulate_grad_batches=args.grad_acc,
            callbacks=[checkpoint_callback, lr_monitor]
        )
    else:
        trainer = pl.Trainer(
            logger=False,
            enable_checkpointing=False,
            gpus=1,
            precision=16,
            max_steps=args.train_step,
            max_epochs=100,
            num_sanity_val_steps=20,
            val_check_interval=args.val_interval * args.grad_acc,
            accumulate_grad_batches=args.grad_acc,
            callbacks=[]
        )

    trainer.fit(model, train_dataloaders=train_data_loader, val_dataloaders=dev_data_loader)
    if not args.debug_only:
        trainer.test(model, ckpt_path='best', dataloaders=test_data_loader)
        torch.save(model.encoder.state_dict(), os.path.join(checkpoint_dir, f'encoder.state'))
    else:
        trainer.test(model, dataloaders=test_data_loader)

