import datetime
import json
import logging
import os
import time
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
from model import nn_metric
from utils import episode_loader, sent_loader, get_io_spans, LabelEncoder,\
    EpisodeDataset, EpisodeSamplingDataset


class LockedDropout(nn.Module):
    def __init__(self, p=0.5):
        super(LockedDropout, self).__init__()
        self.p = p

    def forward(self, x):
        if not self.training:
            return x
        batch_size, seq_length, feat_size = x.size()
        m = x.new(batch_size, 1, feat_size).bernoulli_(1 - self.p)
        assert not m.requires_grad
        mask = m / (1 - self.p)
        mask = mask.expand_as(x)
        assert not mask.requires_grad
        return mask * x


class CosFace(nn.Module):
    def __init__(self, s=10, m=0.4):
        super(CosFace, self).__init__()
        self.s = s
        self.m = m
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-1)

    def forward(self, sims, targets):
        num_classes = sims.size(-1)
        valid_idx = (targets >= 0).nonzero(as_tuple=True)[0]
        valid_sims = sims.index_select(0, valid_idx)
        valid_targets = targets.index_select(0, valid_idx)
        label_one_hot = F.one_hot(valid_targets, num_classes=num_classes)
        margin_cos_sim = self.s * (valid_sims - label_one_hot * self.m)
        loss = self.loss_fn(margin_cos_sim, valid_targets)
        return loss


class Proto(BaseModel):
    def __init__(self, backbone_name, metric,
                 dropout_rate, locked_dropout,
                 bert_layer, bert_layer_agg,
                 cosface, cosface_s, cosface_m,
                 subword_proto, subword_eval,
                 lr, wd, train_step, seed):
        super(Proto, self).__init__()
        self.save_hyperparameters()

        pl.seed_everything(self.hparams.seed)
        self.bert = AutoModel.from_pretrained(self.hparams.backbone_name)
        if self.hparams.locked_dropout:
            self.drop = LockedDropout(self.hparams.dropout_rate)
        else:
            self.drop = nn.Dropout(self.hparams.dropout_rate)

        if self.hparams.cosface:
            self.loss_fn = CosFace(self.hparams.cosface_s, self.hparams.cosface_m)
        else:
            self.loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
        self.bert_layer = [int(i) for i in self.hparams.bert_layer.split(',')]
        self.encode_time, self.comp_time, self.total_time = 0., 0., 0.

    def forward(self, batch, outputs=None):
        start_time = time.time()
        assert outputs is None
        if outputs is None:
            outputs = self.bert(batch['token_id'], attention_mask=batch['atten_mask'],
                                output_hidden_states=True, return_dict=True)
        if len(self.bert_layer) == 1:
            token_embedding = self.drop(outputs['hidden_states'][self.bert_layer[0]])
        else:
            selected_layers = [outputs['hidden_states'][i] for i in self.bert_layer]
            if self.hparams.bert_layer_agg == 'sum':
                token_embedding = self.drop(torch.stack(selected_layers).sum(dim=0))
            elif self.hparams.bert_layer_agg == 'cat':
                token_embedding = self.drop(torch.cat(selected_layers, dim=-1))
            else:
                raise Exception('Unknown BERT Layer Aggregation')

        if self.hparams.metric == 'cosine':
            token_embedding = F.normalize(token_embedding, dim=-1)

        support_mask = (batch['sent_flag'] == 0).bool()
        query_mask = (batch['sent_flag'] == 1).bool()
        subword_mask = batch['atten_mask'].bool()
        word_mask = batch['word_mask'].bool()
        label = batch['label']
        end_time = time.time()
        self.encode_time += end_time - start_time

        batch_id = range(batch['batch_size'])
        loss = 0
        query_preds = []
        query_target = []
        for bid in batch_id:
            start_time = time.time()
            batch_mask = batch['batch_id'] == bid

            batch_support_idx = (batch_mask & support_mask).nonzero(as_tuple=True)[0]
            support_emb = token_embedding.index_select(0, batch_support_idx)
            support_label = label.index_select(0, batch_support_idx)
            support_subword_mask = subword_mask.index_select(0, batch_support_idx)
            support_word_mask = word_mask.index_select(0, batch_support_idx)

            batch_query_idx = (batch_mask & query_mask).nonzero(as_tuple=True)[0]
            query_emb = token_embedding.index_select(0, batch_query_idx)
            query_label = label.index_select(0, batch_query_idx)
            query_subword_mask = subword_mask.index_select(0, batch_query_idx)
            query_word_mask = word_mask.index_select(0, batch_query_idx)
            end_time = time.time()
            self.encode_time += end_time - start_time
            
            start_time = time.time()
            proto = []
            tag_list = [(0, 'O')] + [(lid + 1, l) for lid, l in enumerate(batch['types'][bid])]
            for lid, l in tag_list:
                if lid == -1:
                    continue
                if self.hparams.subword_proto:
                    in_class_sent_idx, in_class_token_idx = (support_label == lid).nonzero(as_tuple=True)
                else:
                    in_class_sent_idx, in_class_token_idx = ((support_label == lid) & support_word_mask).nonzero(
                        as_tuple=True)
                in_class_embedding = support_emb[in_class_sent_idx, in_class_token_idx, :]
                proto.append(torch.mean(in_class_embedding, 0))
            proto = torch.stack(proto)

            if self.hparams.metric == 'dot':
                sim = query_emb.inner(proto)
            elif self.hparams.metric == 'cosine':
                sim = query_emb.inner(F.normalize(proto, dim=-1))
            elif self.hparams.metric == 'L2':
                sim = -torch.cdist(query_emb, proto.unsqueeze(0)).pow(2)
            else:
                raise Exception('Unknown Distance Metric')

            if not self.hparams.subword_eval:
                query_label = query_label.masked_fill(~query_word_mask, -1)

            loss += self.loss_fn(sim.view(-1, proto.size(0)), query_label.view(-1))

            _, pred = torch.max(sim, dim=2)
            query_preds.append(pred)
            query_target.append(query_label)
            end_time = time.time()
            self.comp_time += end_time - start_time
        
        return {
            'loss': loss, 'pred': query_preds, 'target': query_target, 'types': batch['types'],
            'query_id': batch['query_id']
        }

    def training_step(self, batch, batch_idx):
        output = self(batch)
        self.log('train/loss', output['loss'], batch_size=batch['batch_size'])
        return output['loss']

    def eval_step(self, batch, output=None):
        if output is None:
            output = self(batch)
        batch_preds, batch_targets = self.assemble_sentence(output)
        batch_jsons = [json.loads(json_str) for json_str in batch['jsons']]
        return batch_preds, batch_targets, batch_jsons

    def configure_optimizers(self):
        all_parameters = list(self.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        grouped_param = [
            {'params': [p for n, p in all_parameters if not any(
                nd in n for nd in no_decay)], 'weight_decay': self.hparams.wd},
            {'params': [p for n, p in all_parameters if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
        optimizer = AdamW(grouped_param, lr=self.hparams.lr)
        warmup_steps = int(self.hparams.train_step * 0.1)
        logging.info(f'total step: {self.hparams.train_step} warmup step: {warmup_steps}')
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=self.hparams.train_step)
        logging.info(f'lr:{self.hparams.lr} wd:{self.hparams.wd}')

        return [
            {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'step',
                    'frequency': 1
                }
            }
        ]


class NNShot(BaseModel):
    def __init__(self, backbone_name, metric,
                 dropout_rate, locked_dropout,
                 bert_layer, bert_layer_agg,
                 cosface, cosface_s, cosface_m,
                 subword_proto, subword_eval,
                 lr, wd, train_step, seed):
        super(NNShot, self).__init__()
        self.save_hyperparameters()

        pl.seed_everything(self.hparams.seed)
        self.bert = AutoModel.from_pretrained(self.hparams.backbone_name)
        if self.hparams.locked_dropout:
            self.drop = LockedDropout(self.hparams.dropout_rate)
        else:
            self.drop = nn.Dropout(self.hparams.dropout_rate)

        if self.hparams.cosface:
            self.loss_fn = CosFace(self.hparams.cosface_s, self.hparams.cosface_m)
        else:
            self.loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
        self.bert_layer = [int(i) for i in self.hparams.bert_layer.split(',')]

    def forward(self, batch, outputs=None):
        if outputs is None:
            outputs = self.bert(batch['token_id'], attention_mask=batch['atten_mask'],
                                output_hidden_states=True, return_dict=True)
        if len(self.bert_layer) == 1:
            token_embedding = self.drop(outputs['hidden_states'][self.bert_layer[0]])
        else:
            selected_layers = [outputs['hidden_states'][i] for i in self.bert_layer]
            if self.hparams.bert_layer_agg == 'sum':
                token_embedding = self.drop(torch.stack(selected_layers).sum(dim=0))
            elif self.hparams.bert_layer_agg == 'cat':
                token_embedding = self.drop(torch.cat(selected_layers, dim=-1))
            else:
                raise Exception('Unknown BERT Layer Aggregation')

        if self.hparams.metric == 'cosine':
            token_embedding = F.normalize(token_embedding, dim=-1)

        support_mask = (batch['sent_flag'] == 0).bool()
        query_mask = (batch['sent_flag'] == 1).bool()
        subword_mask = batch['atten_mask'].bool()
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
            support_subword_mask = subword_mask.index_select(0, batch_support_idx)
            support_word_mask = word_mask.index_select(0, batch_support_idx)

            batch_query_idx = (batch_mask & query_mask).nonzero(as_tuple=True)[0]
            query_emb = token_embedding.index_select(0, batch_query_idx)
            query_label = label.index_select(0, batch_query_idx)
            query_subword_mask = subword_mask.index_select(0, batch_query_idx)
            query_word_mask = word_mask.index_select(0, batch_query_idx)

            query_sim = []
            tag_list = [(0, 'O')] + [(lid + 1, l) for lid, l in enumerate(batch['types'][bid])]
            for lid, l in tag_list:
                if lid == -1:
                    continue
                if self.hparams.subword_proto:
                    in_class_sent_idx, in_class_token_idx = (support_label == lid).nonzero(as_tuple=True)
                else:
                    in_class_sent_idx, in_class_token_idx = ((support_label == lid) & support_word_mask).nonzero(
                        as_tuple=True)
                in_class_support_emb = support_emb[in_class_sent_idx, in_class_token_idx]
                query_sim.append(nn_metric(self.hparams.metric, query_emb, in_class_support_emb))
            query_sim = torch.stack(query_sim, dim=-1)

            if not self.hparams.subword_eval:
                query_label = query_label.masked_fill(~query_word_mask, -1)
            loss_sims.append(query_sim.view(-1, query_sim.size(-1)))
            loss_targets.append(query_label.view(-1))

            _, pred = torch.max(query_sim, dim=2)
            query_preds.append(pred)
            query_target.append(query_label)

        loss_sims = torch.cat(loss_sims, 0)
        loss_targets = torch.cat(loss_targets, 0)
        loss = self.loss_fn(loss_sims, loss_targets)

        return {
            'loss': loss, 'pred': query_preds, 'target': query_target, 'types': batch['types'],
            'query_id': batch['query_id']
        }

    def training_step(self, batch, batch_idx):
        output = self(batch)
        self.log('train/loss', output['loss'], batch_size=batch['batch_size'])
        return output['loss']

    def eval_step(self, batch, output=None):
        if output is None:
            output = self(batch)
        batch_preds, batch_targets = self.assemble_sentence(output)
        batch_jsons = [json.loads(json_str) for json_str in batch['jsons']]
        return batch_preds, batch_targets, batch_jsons

    def configure_optimizers(self):
        all_parameters = list(self.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        grouped_param = [
            {'params': [p for n, p in all_parameters if not any(
                nd in n for nd in no_decay)], 'weight_decay': self.hparams.wd},
            {'params': [p for n, p in all_parameters if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
        optimizer = AdamW(grouped_param, lr=self.hparams.lr)
        warmup_steps = int(self.hparams.train_step * 0.1)
        logging.info(f'total step: {self.hparams.train_step} warmup step: {warmup_steps}')
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=self.hparams.train_step)
        logging.info(f'lr:{self.hparams.lr} wd:{self.hparams.wd}')

        return [
            {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'step',
                    'frequency': 1
                }
            }
        ]


class StructShot(BaseModel):
    def __init__(self, backbone_name, metric,
                 dropout_rate, locked_dropout,
                 bert_layer, bert_layer_agg,
                 cosface, cosface_s, cosface_m,
                 subword_proto, subword_eval,
                 tau, subword_crf,
                 lr, wd, train_step, seed):
        super(StructShot, self).__init__()
        self.save_hyperparameters()

        pl.seed_everything(self.hparams.seed)
        self.bert = AutoModel.from_pretrained(self.hparams.backbone_name)
        if self.hparams.locked_dropout:
            self.drop = LockedDropout(self.hparams.dropout_rate)
        else:
            self.drop = nn.Dropout(self.hparams.dropout_rate)

        if self.hparams.cosface:
            self.loss_fn = CosFace(self.hparams.cosface_s, self.hparams.cosface_m)
        else:
            self.loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
        self.bert_layer = [int(i) for i in self.hparams.bert_layer.split(',')]

    def set_abstract_transitions(self, loader):
        tag_lists = []
        for types, support, query in loader:
            for words, labels, addtional_info in support:
                tag_lists.append(labels)
            for words, labels, addtional_info in query:
                tag_lists.append(labels)

        s_o, s_i = 0, 0
        o_o, o_i = 0, 0
        i_o, i_i, x_y = 0, 0, 0
        i_e, o_e = 0, 0
        for tags in tag_lists:
            assert len(tags) >= 2, tags
            if tags[0] == 'O':
                s_o += 1
            else:
                s_i += 1
            if tags[-1] == 'O':
                o_e += 1
            else:
                i_e += 1
            for i in range(len(tags) - 1):
                p, n = tags[i], tags[i + 1]
                if p == 'O':
                    if n == 'O':
                        o_o += 1
                    else:
                        o_i += 1
                else:
                    if n == 'O':
                        i_o += 1
                    elif p != n:
                        x_y += 1
                    else:
                        i_i += 1
        i_e = max(i_e, 1) #i_e can be 0
        self.abstract_transitions = [
            s_o / (s_o + s_i), s_i / (s_o + s_i),
            o_e / (o_e + i_e), i_e / (o_e + i_e),
            o_o / (o_o + o_i), o_i / (o_o + o_i),
            i_o / (i_o + i_i + x_y), i_i / (i_o + i_i + x_y), x_y / (i_o + i_i + x_y),
        ]

    def project_target_transitions(self, n_tag):
        s_o, s_i, o_e, i_e, o_o, o_i, i_o, i_i, x_y = self.abstract_transitions

        start = torch.zeros(n_tag + 1)
        start[0] = s_o
        start[1:] = s_i / n_tag

        end = torch.zeros(n_tag + 1)
        end[0] = o_e
        end[1:] = i_e / n_tag

        transitions = torch.zeros(n_tag + 1, n_tag + 1)
        transitions[1:, 1:] = x_y / (n_tag - 1)
        transitions[1:, 1:].fill_diagonal_(i_i)
        transitions[0, 0] = o_o
        transitions[0, 1:] = o_i / n_tag
        transitions[1:, 0] = i_o

        start = F.normalize(start.pow(self.hparams.tau), dim=-1, p=1).log()
        end = F.normalize(end.pow(self.hparams.tau), dim=-1, p=1).log()
        transitions = F.normalize(transitions.pow(self.hparams.tau), dim=-1, p=1).log()
        return start, end, transitions

    @staticmethod
    def viterbi_decode(start, end, transitions, emissions):
        sent_len = emissions.size(0)

        score = start + emissions[0, :]
        history = []

        for i in range(1, sent_len):
            score, indices = torch.max(score.unsqueeze(1) + transitions + emissions[i].unsqueeze(0), dim=0)
            history.append(indices)

        score += end
        _, best_last_tag = score.max(dim=0)
        best_tags = [best_last_tag]

        for indices in reversed(history):
            best_last_tag = indices[best_tags[-1]]
            best_tags.append(best_last_tag)

        best_tags.reverse()
        return torch.stack(best_tags)

    def forward(self, batch, outputs=None):
        if outputs is None:
            outputs = self.bert(batch['token_id'], attention_mask=batch['atten_mask'],
                                output_hidden_states=True, return_dict=True)
        if len(self.bert_layer) == 1:
            token_embedding = self.drop(outputs['hidden_states'][self.bert_layer[0]])
        else:
            selected_layers = [outputs['hidden_states'][i] for i in self.bert_layer]
            if self.hparams.bert_layer_agg == 'sum':
                token_embedding = self.drop(torch.stack(selected_layers).sum(dim=0))
            elif self.hparams.bert_layer_agg == 'cat':
                token_embedding = self.drop(torch.cat(selected_layers, dim=-1))
            else:
                raise Exception('Unknown BERT Layer Aggregation')

        if self.hparams.metric == 'cosine':
            token_embedding = F.normalize(token_embedding, dim=-1)

        support_mask = (batch['sent_flag'] == 0).bool()
        query_mask = (batch['sent_flag'] == 1).bool()
        subword_mask = batch['atten_mask'].bool()
        word_mask = batch['word_mask'].bool()
        label = batch['label']

        batch_id = range(batch['batch_size'])
        loss_sims = []
        loss_targets = []
        query_valid_mask = []
        query_target = []
        query_emission = []
        for bid in batch_id:
            batch_mask = batch['batch_id'] == bid

            batch_support_idx = (batch_mask & support_mask).nonzero(as_tuple=True)[0]
            support_emb = token_embedding.index_select(0, batch_support_idx)
            support_label = label.index_select(0, batch_support_idx)
            support_subword_mask = subword_mask.index_select(0, batch_support_idx)
            support_word_mask = word_mask.index_select(0, batch_support_idx)

            batch_query_idx = (batch_mask & query_mask).nonzero(as_tuple=True)[0]
            query_emb = token_embedding.index_select(0, batch_query_idx)
            query_label = label.index_select(0, batch_query_idx)
            query_subword_mask = subword_mask.index_select(0, batch_query_idx)
            query_word_mask = word_mask.index_select(0, batch_query_idx)

            query_sim = []
            tag_list = [(0, 'O')] + [(lid + 1, l) for lid, l in enumerate(batch['types'][bid])]
            for lid, l in tag_list:
                if lid == -1:
                    continue
                if self.hparams.subword_proto:
                    in_class_sent_idx, in_class_token_idx = (support_label == lid).nonzero(as_tuple=True)
                else:
                    in_class_sent_idx, in_class_token_idx = ((support_label == lid) & support_word_mask).nonzero(
                        as_tuple=True)
                in_class_support_emb = support_emb[in_class_sent_idx, in_class_token_idx]
                query_sim.append(nn_metric(self.hparams.metric, query_emb, in_class_support_emb))
            query_sim = torch.stack(query_sim, dim=-1)

            loss_sims.append(query_sim.view(-1, query_sim.size(-1)))
            if not self.hparams.subword_eval:
                masked_query_label = query_label.masked_fill(~query_word_mask, -1)
                loss_targets.append(masked_query_label.view(-1))
                query_target.append(masked_query_label)
            else:
                loss_targets.append(query_label.view(-1))
                query_target.append(query_label)

            _, pred = torch.max(query_sim, dim=2)
            query_valid_mask.append(query_label >= 0)
            query_emission.append(query_sim)

        loss_sims = torch.cat(loss_sims, 0)
        loss_targets = torch.cat(loss_targets, 0)
        loss = self.loss_fn(loss_sims, loss_targets)

        return {
            'loss': loss,
            'emission': query_emission, 'target': query_target, 'valid_mask': query_valid_mask,
            'types': batch['types'], 'query_id': batch['query_id']
        }

    def training_step(self, batch, batch_idx):
        output = self(batch)
        self.log('train/loss', output['loss'], batch_size=batch['batch_size'])
        return output['loss']

    def eval_step(self, batch, output=None):
        if output is None:
            output = self(batch)
        batch_preds = []
        batch_targets = []
        assert len(output['emission']) == len(output['target']) == len(output['valid_mask']) \
               == len(output['types']) == len(output['query_id'])
        for emissions, targets, valid_masks, types, sent_ids in zip(output['emission'], output['target'],
                                                                    output['valid_mask'],
                                                                    output['types'], output['query_id']):
            label_encoder = LabelEncoder(types)
            episode_preds = defaultdict(list)
            episode_targets = defaultdict(list)
            assert len(emissions) == len(targets) == len(sent_ids)
            start, end, transitions = self.project_target_transitions(len(types))
            for chunk_emission, chunk_target, chunk_mask, sent_id in zip(emissions, targets, valid_masks, sent_ids):
                chunk_emission = chunk_emission.cpu()
                chunk_target = chunk_target.cpu()
                chunk_mask = chunk_mask.cpu()
                logits = F.softmax(chunk_emission, dim=1).log()
                if self.hparams.subword_crf:
                    valid_emission = torch.index_select(logits, 0, chunk_mask.nonzero().squeeze())
                    valid_target = torch.index_select(chunk_target, 0, chunk_mask.nonzero().squeeze())
                    if valid_emission.size(0) == 0:
                        continue
                    chunk_pred = self.viterbi_decode(start, end, transitions, valid_emission)
                    episode_preds[sent_id].extend(chunk_pred[valid_target >= 0].tolist())
                    episode_targets[sent_id].extend(chunk_target[chunk_target >= 0].tolist())
                else:
                    valid_emission = torch.index_select(logits, 0, (chunk_target >= 0).nonzero().squeeze())
                    if valid_emission.size(0) == 0:
                        continue
                    chunk_pred = self.viterbi_decode(start, end, transitions, valid_emission)
                    episode_preds[sent_id].extend(chunk_pred.tolist())
                    episode_targets[sent_id].extend(chunk_target[chunk_target >= 0].tolist())

            episode_keys = sorted(list(episode_preds.keys()))
            episode_preds = [label_encoder.get(episode_preds[k]) for k in episode_keys]
            episode_targets = [label_encoder.get(episode_targets[k]) for k in episode_keys]

            batch_preds.append(episode_preds)
            batch_targets.append(episode_targets)
        batch_jsons = [json.loads(json_str) for json_str in batch['jsons']]
        return batch_preds, batch_targets, batch_jsons

    def configure_optimizers(self):
        all_parameters = list(self.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        grouped_param = [
            {'params': [p for n, p in all_parameters if not any(
                nd in n for nd in no_decay)], 'weight_decay': self.hparams.wd},
            {'params': [p for n, p in all_parameters if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
        optimizer = AdamW(grouped_param, lr=self.hparams.lr)
        warmup_steps = int(self.hparams.train_step * 0.1)
        logging.info(f'total step: {self.hparams.train_step} warmup step: {warmup_steps}')
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=self.hparams.train_step)
        logging.info(f'lr:{self.hparams.lr} wd:{self.hparams.wd}')

        return [
            {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'step',
                    'frequency': 1
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
    parser.add_argument('--batch_size', default=4, type=int,
                        help='batch size')
    parser.add_argument('--train_step', default=12000, type=int,
                        help='num of step in training')
    parser.add_argument('--val_interval', default=1000, type=int,
                        help='num of interval between validation')
    parser.add_argument('--model', default='proto',
                        help='model name, must be in [proto, nnshot, structshot]')
    parser.add_argument('--backbone_name', default='bert-base-uncased',
                        help='bert/roberta encoder name')
    parser.add_argument('--metric', default='L2',
                        help='similarity metric, must be in [L2, dot, cosine]')
    parser.add_argument('--max_length', default=96, type=int,
                        help='max length')
    parser.add_argument('--lr', default=2e-5, type=float,
                        help='learning rate')
    parser.add_argument('--wd', default=0, type=float,
                        help='weight decay rate')
    parser.add_argument('--dropout', default=0, type=float,
                        help='dropout rate')
    parser.add_argument('--locked_dropout', action='store_true',
                        help='use locked_dropout')
    parser.add_argument('--cosface', action='store_true',
                        help='use cosface')
    parser.add_argument('--cosface_s', default=20, type=float,
                        help='cosface scale')
    parser.add_argument('--cosface_m', default=0, type=float,
                        help='cosface margin')
    parser.add_argument('--subword_proto', action='store_true',
                        help='use subword level label to build prototype')
    parser.add_argument('--subword_eval', action='store_true',
                        help='use subword level label to eval')
    parser.add_argument('--bert_layer', default='-1',
                        help='which bert layer to extract token embedding')
    parser.add_argument('--bert_layer_agg', default='sum',
                        help='how to aggregate bert_layer, must be in [sum, cat]')
    parser.add_argument('--seed', default=0, type=int,
                        help='random seed')
    parser.add_argument('--tau', default=0.32, type=float,
                        help='structshot tau')
    parser.add_argument('--subword_crf', action='store_false',
                        help='use subword level crf to decode')
    parser.add_argument('--test_only', action='store_true',
                        help='only test')
    parser.add_argument('--test_checkpoint', default='',
                        help='checkpoint path loaded for testing')
    parser.add_argument('--debug_only', action='store_true',
                        help='only debug')
    parser.add_argument('--output_dir_overwrite', action='store_true',
                        help='output to specified dir')
    parser.add_argument('--output_dir', default='',
                        help='output dir')
    parser.add_argument('--wandb_proj', default='FewNERD',
                        help='wandb project')
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

    logging.info(f'model:{args.model}-{args.backbone_name}-{args.metric}-{args.seed}')
    logging.info(f'tau:{args.tau}-subword_crf:{args.subword_crf}')
    logging.info(f'fewshot:{args.mode}-{args.N}-{args.K}-{args.max_length}')
    logging.info(f'train:{args.train_step}-{args.val_interval}-{args.batch_size}')
    logging.info(f'optimizer:{args.lr}-{args.wd}')
    logging.info(f'dropout:{args.dropout}-{args.locked_dropout}')
    logging.info(f'loss:{args.cosface}-{args.cosface_s}-{args.cosface_m}')
    logging.info(f'subword:{args.subword_proto}-{args.subword_eval}')
    logging.info(f'bert_layer:{args.bert_layer}-{args.bert_layer_agg}')

    if args.model == 'proto':
        model = Proto(args.backbone_name, args.metric,
                      args.dropout, args.locked_dropout,
                      args.bert_layer, args.bert_layer_agg,
                      args.cosface, args.cosface_s, args.cosface_m,
                      args.subword_proto, args.subword_eval,
                      args.lr, args.wd,
                      args.train_step, args.seed)
    elif args.model == 'nnshot':
        model = NNShot(args.backbone_name, args.metric,
                       args.dropout, args.locked_dropout,
                       args.bert_layer, args.bert_layer_agg,
                       args.cosface, args.cosface_s, args.cosface_m,
                       args.subword_proto, args.subword_eval,
                       args.lr, args.wd,
                       args.train_step, args.seed)
    elif args.model == 'structshot':
        model = StructShot(args.backbone_name, args.metric,
                           args.dropout, args.locked_dropout,
                           args.bert_layer, args.bert_layer_agg,
                           args.cosface, args.cosface_s, args.cosface_m,
                           args.subword_proto, args.subword_eval,
                           args.tau, args.subword_crf,
                           args.lr, args.wd,
                           args.train_step, args.seed)
        model.set_abstract_transitions(episode_loader(args.mode, 'train', args.N, args.K))
        logging.info(model.abstract_transitions)
    else:
        raise Exception('Unknown model type')

    worker_num = min(os.cpu_count(), 4)
    pl.seed_everything(args.seed)
    train_dataset = EpisodeSamplingDataset(args.N, args.K, args.K, sent_loader(args.mode, 'train'), args.backbone_name,
                                           args.max_length)
    assert len(train_dataset) >= args.train_step * args.batch_size
    logging.info(f'train dataset:{args.mode}-{args.N}-{args.K} max_length:{args.max_length}')
    train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=worker_num,
                                   collate_fn=collate_fn,
                                   shuffle=True)
    logging.info(f'training loader:{len(train_data_loader)} batch:{args.batch_size} worker:{worker_num}')

    dev_dataset = EpisodeDataset(episode_loader(args.mode, 'dev', args.N, args.K), args.backbone_name, args.max_length)
    logging.info(f'dev dataset:{args.mode}-{args.N}-{args.K} max_length:{args.max_length}')
    dev_data_loader = DataLoader(dev_dataset, batch_size=args.batch_size, num_workers=worker_num, collate_fn=collate_fn)
    logging.info(f'dev loader:{len(dev_data_loader)} batch:{args.batch_size} worker:{worker_num}')

    test_dataset = EpisodeDataset(episode_loader(args.mode, 'test', args.N, args.K), args.backbone_name,
                                  args.max_length)
    logging.info(f'test dataset:{args.mode}-{args.N}-{args.K} max_length:{args.max_length}')
    test_data_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=worker_num,
                                  collate_fn=collate_fn)
    logging.info(f'test loader:{len(test_data_loader)} batch:{args.batch_size} worker:{worker_num}')

    if not args.test_only:
        if not args.debug_only:
            checkpoint_callback = pl.callbacks.model_checkpoint.ModelCheckpoint(
                dirpath=checkpoint_dir,
                filename="{epoch}-{step}",
                monitor='valid/f1',
                mode='max',
            )

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
                val_check_interval=args.val_interval,
                callbacks=[checkpoint_callback]
            )
        else:
            trainer = pl.Trainer(
                logger=False,
                enable_checkpointing=False,
                gpus=1,
                precision=16,
                max_steps=args.train_step,
                max_epochs=100,
                num_sanity_val_steps=2,
                val_check_interval=args.val_interval,
                callbacks=[]
            )

        trainer.fit(model, train_dataloaders=train_data_loader, val_dataloaders=dev_data_loader)
        if not args.debug_only:
            trainer.test(model, ckpt_path='best', dataloaders=test_data_loader)
    else:
        trainer = pl.Trainer(
            logger=False,
            enable_checkpointing=False,
            gpus=1,
            precision=16,
            max_steps=args.train_step,
            max_epochs=100,
            num_sanity_val_steps=0,
            val_check_interval=args.val_interval,
            callbacks=[]
        )

        if args.test_checkpoint != '':
            trainer.test(model, ckpt_path=args.test_checkpoint, dataloaders=test_data_loader)
        else:
            trainer.test(model, ckpt_path=None, dataloaders=test_data_loader)
