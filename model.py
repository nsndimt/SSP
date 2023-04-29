import datetime
import json
import logging
import math

import numpy as np
import os
import pytorch_lightning as pl
import random
import sys
import torch
from collections import defaultdict
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm
from transformers import AutoModel, AdamW, get_linear_schedule_with_warmup
from utils import get_io_spans, get_bio_spans, LabelEncoder


def all_avg_eval(test_preds, test_targets, print_res=False):
    pred_cnt = 0  # pred entity cnt
    label_cnt = 0  # true label entity cnt
    correct_cnt = 0  # correct predicted entity cnt

    fp_cnt = 0  # misclassify O as I-
    fn_cnt = 0  # misclassify I- as O
    total_token_cnt = 0  # total token cnt

    within_cnt = 0  # span correct but of wrong fine-grained type
    outer_cnt = 0  # span correct but of wrong coarse-grained type
    total_span_cnt = 0  # span correct
    for episode_preds, episode_targets in zip(test_preds, test_targets):
        assert len(episode_preds) == len(episode_targets)
        for sent_pred, sent_target in zip(episode_preds, episode_targets):
            assert len(sent_pred) == len(sent_target) > 0
            pred_spans, _ = get_io_spans(sent_pred)
            target_spans, _ = get_io_spans(sent_target)

            pred_cnt += len(pred_spans)
            label_cnt += len(target_spans)

            for pi, pj, pl in pred_spans:
                for ti, tj, tl in target_spans:
                    if pi == ti and pj == tj and pl == tl:
                        correct_cnt += 1
                        total_span_cnt += 1
                    elif pi == ti and pj == tj:
                        total_span_cnt += 1

                        if '-' in pl and '-' in tl:
                            pc, pf = pl.split('-')
                            tc, tf = tl.split('-')
                            if pc == tc:
                                within_cnt += 1
                            else:
                                outer_cnt += 1

            for p, t in zip(sent_pred, sent_target):
                assert p != '##NULL##' and t != '##NULL##'
                if p == 'O' and t != 'O':
                    fn_cnt += 1
                elif p != 'O' and t == 'O':
                    fp_cnt += 1
                total_token_cnt += 1

    precision = correct_cnt / (pred_cnt + 1e-6)
    recall = correct_cnt / (label_cnt + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)

    fp_error = fp_cnt / (total_token_cnt + 1e-6)
    fn_error = fn_cnt / (total_token_cnt + 1e-6)

    within_error = within_cnt / (total_span_cnt + 1e-6)
    outer_error = outer_cnt / (total_span_cnt + 1e-6)

    if print_res:
        logging.info('precision: {0:3.4f}, recall: {1:3.4f}, f1: {2:3.4f}'.format(precision, recall, f1))
        logging.info('fp: {0:3.4f}, fn: {1:3.4f}, within: {2:3.4f}, outer: {3:3.4f}'.format(
            fp_error, fn_error, within_error, outer_error))

    return precision, recall, f1, fp_error, fn_error, within_error, outer_error


def episode_avg_eval(test_preds, test_targets, print_res=False):
    episode_f1 = []
    episode_precision = []
    episode_recall = []
    for episode_preds, episode_targets in zip(test_preds, test_targets):
        assert len(episode_preds) == len(episode_targets)
        pred_cnt = 0
        label_cnt = 0
        correct_cnt = 0
        for sent_pred, sent_target in zip(episode_preds, episode_targets):
            assert len(sent_pred) == len(sent_target) > 0
            pred_spans, _ = get_io_spans(sent_pred)
            target_spans, _ = get_bio_spans(sent_target)

            pred_cnt += len(pred_spans)
            label_cnt += len(target_spans)

            for pi, pj, pl in pred_spans:
                for ti, tj, tl in target_spans:
                    if pi == ti and pj == tj and pl == tl:
                        correct_cnt += 1

        precision = correct_cnt / (pred_cnt + 1e-6)
        recall = correct_cnt / (label_cnt + 1e-6)
        f1 = 2 * precision * recall / (precision + recall + 1e-6)

        episode_f1.append(f1)
        episode_precision.append(precision)
        episode_recall.append(recall)

    episode_avg_f1 = sum(episode_f1) / len(episode_f1)
    episode_avg_precision = sum(episode_precision) / len(episode_precision)
    episode_avg_recall = sum(episode_recall) / len(episode_recall)

    if print_res:
        logging.info('precision: {0:3.4f}, recall: {1:3.4f}, f1: {2:3.4f}'.format(
            episode_avg_precision, episode_avg_recall, episode_avg_f1))
    return episode_avg_f1, episode_avg_precision, episode_avg_recall


class FewShotBaseModel(pl.LightningModule):
    def __init__(self):
        super(FewShotBaseModel, self).__init__()
        self.snips_mode = False

    # wired but not to break old model
    def set_snips_mode(self):
        self.snips_mode = True

    def unset_snips_mode(self):
        self.snips_mode = False

    @staticmethod
    def assemble_sentence(output):
        batch_preds = []
        batch_targets = []
        assert len(output['pred']) == len(output['target']) == len(output['types']) == len(output['query_id'])
        for preds, targets, types, sent_ids in zip(output['pred'], output['target'], output['types'],
                                                   output['query_id']):
            label_encoder = LabelEncoder(types)
            episode_preds = defaultdict(list)
            episode_targets = defaultdict(list)
            assert len(preds) == len(targets) == len(sent_ids)
            for chunk_pred, chunk_target, sent_id in zip(preds, targets, sent_ids):
                chunk_pred = chunk_pred.cpu()
                chunk_target = chunk_target.cpu()

                episode_preds[sent_id].extend(chunk_pred[chunk_target >= 0].tolist())
                episode_targets[sent_id].extend(chunk_target[chunk_target >= 0].tolist())

            episode_keys = sorted(list(episode_preds.keys()))
            episode_preds = [label_encoder.get(episode_preds[k]) for k in episode_keys]
            episode_targets = [label_encoder.get(episode_targets[k]) for k in episode_keys]

            batch_preds.append(episode_preds)
            batch_targets.append(episode_targets)

        return batch_preds, batch_targets

    def eval_step(self, batch):
        output = self(batch)
        batch_jsons = [json.loads(json_str) for json_str in batch['jsons']]
        batch_preds, batch_targets = self.assemble_sentence(output)
        return batch_preds, batch_targets, batch_jsons

    def eval_epoch_end(self, eval_step_outputs):
        test_preds, test_targets, test_jsons = [], [], []
        for batch_preds, batch_targets, batch_jsons in eval_step_outputs:
            assert len(batch_preds) == len(batch_targets) == len(batch_jsons)
            for episode_preds, episode_targets, episode_json in zip(batch_preds, batch_targets, batch_jsons):
                assert len(episode_preds) == len(episode_targets) == len(episode_json['query'])
                for sent_pred, sent_target, query_sent in zip(episode_preds, episode_targets, episode_json['query']):
                    words, labels, additional_info = query_sent
                    if 'zero_subword_remove' not in additional_info:
                        assert sent_target == labels, (sent_target, labels)
                        assert len(words) == len(labels) == len(sent_pred), (words, labels, sent_pred)
                    else:
                        for idx,l in additional_info['zero_subword_remove']:
                            sent_pred.insert(idx, 'O')
                            sent_target.insert(idx, l)
                        assert sent_target == labels, (sent_target, labels)
                        assert len(words) == len(labels) == len(sent_pred), (words, labels, sent_pred)

            test_preds.extend(batch_preds)
            test_targets.extend(batch_targets)
            test_jsons.extend(batch_jsons)
        if not self.snips_mode:
            precision, recall, f1, fp_error, fn_error, within_error, outer_error =\
                all_avg_eval(test_preds, test_targets, True)
            return precision, recall, f1, fp_error, fn_error, within_error, outer_error
        else:
            bio_test_targets = []
            for episode_json in test_jsons:
                bio_episode_targets= []
                for query_sent in episode_json['query']:
                    words, labels, additional_info = query_sent
                    bio_labels = additional_info['bio_labels']
                    bio_episode_targets.append(bio_labels)
                bio_test_targets.append(bio_episode_targets)
            episode_avg_f1, episode_avg_precision, episode_avg_recall =\
                episode_avg_eval(test_preds, bio_test_targets, True)
            return episode_avg_f1, episode_avg_precision, episode_avg_recall

    def validation_step(self, batch, batch_idx):
        return self.eval_step(batch)

    def validation_epoch_end(self, valid_step_outputs):
        logging.info('## Validation Result ##')
        if not self.snips_mode:
            precision, recall, f1, fp_error, fn_error, within_error, outer_error =\
                self.eval_epoch_end(valid_step_outputs)
            self.log('valid/precision', precision)
            self.log('valid/recall', recall)
            self.log('valid/f1', f1)
            self.log('valid/fp_error', fp_error)
            self.log('valid/fn_error', fn_error)
            self.log('valid/within_error', within_error)
            self.log('valid/outer_error', outer_error)
        else:
            episode_avg_f1, episode_avg_precision, episode_avg_recall = self.eval_epoch_end(valid_step_outputs)
            self.log('valid/precision', episode_avg_precision)
            self.log('valid/recall', episode_avg_recall)
            self.log('valid/f1', episode_avg_f1)

    def test_step(self, batch, batch_idx):
        return self.eval_step(batch)

    def test_epoch_end(self, test_step_outputs):
        logging.info('## Test Result ##')
        if not self.snips_mode:
            precision, recall, f1, fp_error, fn_error, within_error, outer_error =\
                self.eval_epoch_end(test_step_outputs)
            self.log('test/precision', precision)
            self.log('test/recall', recall)
            self.log('test/f1', f1)
            self.log('test/fp_error', fp_error)
            self.log('test/fn_error', fn_error)
            self.log('test/within_error', within_error)
            self.log('test/outer_error', outer_error)
        else:
            episode_avg_f1, episode_avg_precision, episode_avg_recall = self.eval_epoch_end(test_step_outputs)
            self.log('test/precision', episode_avg_precision)
            self.log('test/recall', episode_avg_recall)
            self.log('test/f1', episode_avg_f1)


class ClassificationBaseModel(FewShotBaseModel):
    def __init__(self):
        super(ClassificationBaseModel, self).__init__()

    def assemble_sentence(self, output):
        batch_preds = defaultdict(list)
        batch_targets = defaultdict(list)

        assert len(output['pred']) == len(output['target']) == len(output['sent_id'])
        preds, targets, sent_ids = output['pred'], output['target'], output['sent_id']
        for chunk_pred, chunk_target, sent_id in zip(preds, targets, sent_ids):
            chunk_pred = chunk_pred.cpu()
            chunk_target = chunk_target.cpu()

            batch_preds[sent_id].extend(chunk_pred[chunk_target >= 0].tolist())
            batch_targets[sent_id].extend(chunk_target[chunk_target >= 0].tolist())

        episode_keys = sorted(list(batch_preds.keys()))
        batch_preds = [self.label_encoder.get(batch_preds[k]) for k in episode_keys]
        batch_targets = [self.label_encoder.get(batch_targets[k]) for k in episode_keys]

        return batch_preds, batch_targets

    def eval_step(self, batch):
        output = self(batch)
        batch_jsons = [json.loads(json_str) for json_str in batch['jsons']]
        batch_preds, batch_targets = self.assemble_sentence(output)
        return batch_preds, batch_targets, batch_jsons

    def eval_epoch_end(self, eval_step_outputs):
        test_preds, test_targets, test_jsons = [], [], []
        for batch_preds, batch_targets, batch_jsons in eval_step_outputs:
            assert len(batch_preds) == len(batch_targets) == len(batch_jsons)
            for sent_pred, sent_target, sent_json in zip(batch_preds, batch_targets, batch_jsons):
                words, labels, additional_info = sent_json
                if 'zero_subword_remove' not in additional_info:
                    assert sent_target == labels, (sent_target, labels, additional_info)
                    assert len(words) == len(labels) == len(sent_pred), (words, labels, sent_pred)
                else:
                    for idx,l in additional_info['zero_subword_remove']:
                        sent_pred.insert(idx, 'O')
                        sent_target.insert(idx, l)
                    assert sent_target == labels, (sent_target, labels)
                    assert len(words) == len(labels) == len(sent_pred), (words, labels, sent_pred)

            test_preds.append(batch_preds)
            test_targets.append(batch_targets)
            test_jsons.append(batch_jsons)

        if not self.snips_mode:
            #global average each episode is a batch
            precision, recall, f1, fp_error, fn_error, within_error, outer_error = \
                all_avg_eval(test_preds, test_targets, True)
            return precision, recall, f1, fp_error, fn_error, within_error, outer_error
        else:
            bio_test_targets = []
            for batch_jsons in test_jsons:
                for words, labels, additional_info in batch_jsons:
                    bio_labels = additional_info['bio_labels']
                    bio_test_targets.append(bio_labels)
            flat_test_preds = [sent_pred for batch_preds in test_preds for sent_pred in batch_preds]
            episode_avg_f1, episode_avg_precision, episode_avg_recall = \
                episode_avg_eval([flat_test_preds], [bio_test_targets], True)
            return episode_avg_f1, episode_avg_precision, episode_avg_recall


class DotAttention(nn.Module):
    def __init__(self):
        super(DotAttention, self).__init__()

    def forward(self, query, key, value):
        if query.size(0) < 20480 and key.size(0) < 20480 and value.size(0) < 20480:
            atten_weight = (query.matmul(key.T)).softmax(dim=-1)
            x = atten_weight.matmul(value)
            return x, atten_weight
        else:
            assert not self.training, 'should only be called when we test very large N'
            x = []
            for query_chunk in torch.split(query, 2048, dim=0):
                atten_weight = (query_chunk.matmul(key.T)).softmax(dim=-1)
                x.append(atten_weight.matmul(value))
            x = torch.cat(x, 0)
            return x, None


class AvgAttention(nn.Module):
    def __init__(self):
        super(AvgAttention, self).__init__()

    def forward(self, query, key, value):
        x = torch.mean(value.view(-1, value.size(-1)), axis=0)
        return x, None


class L2Scale(nn.Module):
    def __init__(self):
        super(L2Scale, self).__init__()

    def forward(self, x):
        return F.normalize(x, p=2.0, dim=-1)


class SpanExtractor(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0, dropout_pos='before_linear', output='layernorm', max_len=10):
        super(SpanExtractor, self).__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim

        self.dropout_pos = dropout_pos
        self.proj = nn.Linear(input_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)
        if output == 'layernorm':
            self.output = nn.LayerNorm(output_dim)
        elif output == 'simple-layernorm':
            self.output = nn.LayerNorm(output_dim, elementwise_affine=False)
        elif output == 'gelu':
            self.output = nn.GELU()
        elif output == 'tanh+layernorm':
            self.output = nn.Sequential(nn.Tanh(), nn.LayerNorm(output_dim))
        elif output == 'gelu+layernorm':
            self.output = nn.Sequential(nn.GELU(), nn.LayerNorm(output_dim))
        elif output == 'tanh':
            self.output = nn.Tanh()
        elif output == 'raw':
            self.output = nn.Identity()
        elif output == 'raw_norm':
            self.output = L2Scale()
        elif output == 'batchnorm':
            self.output = nn.BatchNorm1d(output_dim)
        else:
            raise Exception('Unknown output layer')

        self.max_len = max_len
        self.subword_len_emb = nn.Embedding(max_len + 1, input_dim * 2)
        nn.init.zeros_(self.subword_len_emb.weight)
        self.word_len_emb = nn.Embedding(max_len + 1, input_dim * 2)
        nn.init.zeros_(self.word_len_emb.weight)

    def forward(self, word_repr, word_mask, gather_start, gather_end):
        assert word_repr.ndim == 2
        start = word_repr.index_select(0, gather_start)
        end = word_repr.index_select(0, gather_end)

        subword_len = torch.clamp(gather_end - gather_start + 1, 0, self.max_len)
        subword_len_emb = self.subword_len_emb(subword_len)

        word_cumsum = torch.cumsum(word_mask, -1)
        word_len = torch.clamp(word_cumsum[gather_end] - word_cumsum[gather_start] + 1, 0, self.max_len)
        word_len_emb = self.word_len_emb(word_len)

        span_rep = torch.cat([start, end], dim=-1) + subword_len_emb + word_len_emb

        if self.dropout_pos == 'before_linear':
            span_rep = self.dropout(span_rep)
            span_rep = self.proj(span_rep)
            span_rep = self.output(span_rep)
        elif self.dropout_pos == 'before_layernorm':
            span_rep = self.dropout(span_rep)
            span_rep = self.proj(span_rep)
            span_rep = self.output(span_rep)
        elif self.dropout_pos == 'final':
            span_rep = self.proj(span_rep)
            span_rep = self.output(span_rep)
            span_rep = self.dropout(span_rep)

        return span_rep


class L2SquareClassifier(nn.Module):
    def __init__(self, num_class, embed_dim):
        super(L2SquareClassifier, self).__init__()
        self.class_center = nn.parameter.Parameter(torch.empty(num_class, embed_dim))
        nn.init.kaiming_uniform_(self.class_center, a=math.sqrt(5))
        self.output = nn.LogSoftmax(dim=-1)

    def forward(self, token_embedding):
        dis = 2 * token_embedding.matmul(self.class_center.T) - token_embedding.pow(2).sum(-1).unsqueeze(-1) - \
              self.class_center.pow(2).sum(-1)
        return self.output(dis)


class DotClassifier(nn.Module):
    def __init__(self, num_class, embed_dim):
        super(DotClassifier, self).__init__()
        self.dot = nn.Linear(embed_dim, num_class)
        self.output = nn.LogSoftmax(dim=-1)

    def forward(self, token_embedding):
        return self.output(self.dot(token_embedding))


class SpanEncoder(nn.Module):
    def __init__(self, backbone_name, span_rep_dim, max_len, dropout, dropout_pos, span_output, pretrained_ckpt='',
                 reinit=False):
        super(SpanEncoder, self).__init__()
        self.bert = AutoModel.from_pretrained(backbone_name)
        if not reinit:
            self.span_extractor = SpanExtractor(self.bert.config.hidden_size, span_rep_dim, dropout, dropout_pos,
                                                span_output, max_len)
            if pretrained_ckpt != '':
                res = self.load_state_dict(torch.load(pretrained_ckpt))
                assert len(res.missing_keys) == 0 and len(res.unexpected_keys) == 0
        else:
            if pretrained_ckpt != '':
                res = self.load_state_dict(torch.load(pretrained_ckpt), strict=False)
                assert len(res.missing_keys) == 0 and all([k.startswith('span_extractor') for k in res.unexpected_keys])
            self.span_extractor = SpanExtractor(self.bert.config.hidden_size, span_rep_dim, dropout, dropout_pos,
                                                span_output, max_len)

    def forward(self, token_id, atten_mask):
        outputs = self.bert(token_id, attention_mask=atten_mask, output_hidden_states=True, return_dict=True)
        bert_embedding = outputs['hidden_states'][-1]
        return bert_embedding


class TokenExtractor(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0, dropout_pos='before_linear', output='layernorm', ):
        super(TokenExtractor, self).__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim

        self.dropout_pos = dropout_pos
        self.proj = nn.Linear(input_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        if output == 'layernorm':
            self.output = nn.LayerNorm(output_dim)
        elif output == 'simple-layernorm':
            self.output = nn.LayerNorm(output_dim, elementwise_affine=False)
        elif output == 'gelu':
            self.output = nn.GELU()
        elif output == 'tanh':
            self.output = nn.Tanh()
        elif output == 'tanh+layernorm':
            self.output = nn.Sequential(nn.Tanh(), nn.LayerNorm(output_dim))
        elif output == 'gelu+layernorm':
            self.output = nn.Sequential(nn.GELU(), nn.LayerNorm(output_dim))
        elif output == 'raw':
            self.output = nn.Identity()
        elif output == 'raw_norm':
            self.output = L2Scale()
        elif output == 'batchnorm':
            self.output = nn.BatchNorm1d(output_dim)
        else:
            raise Exception('Unknown output layer')

    def forward(self, token_embedding):
        if self.dropout_pos == 'before_linear':
            token_embedding = self.dropout(token_embedding)
            token_embedding = self.proj(token_embedding)
            token_embedding = self.output(token_embedding)
        elif self.dropout_pos == 'before_layernorm':
            token_embedding = self.dropout(token_embedding)
            token_embedding = self.proj(token_embedding)
            token_embedding = self.output(token_embedding)
        elif self.dropout_pos == 'final':
            token_embedding = self.proj(token_embedding)
            token_embedding = self.output(token_embedding)
            token_embedding = self.dropout(token_embedding)

        return token_embedding


class TokenEncoder(nn.Module):
    def __init__(self, backbone_name, token_rep_dim, dropout, dropout_pos, span_output, pretrained_ckpt='',
                 reinit=False):
        super(TokenEncoder, self).__init__()
        self.bert = AutoModel.from_pretrained(backbone_name)
        if not reinit:
            self.token_extractor = TokenExtractor(self.bert.config.hidden_size, token_rep_dim, dropout, dropout_pos,
                                                span_output)
            if pretrained_ckpt != '':
                res = self.load_state_dict(torch.load(pretrained_ckpt))
                assert len(res.missing_keys) == 0 and len(res.unexpected_keys) == 0
        else:
            if pretrained_ckpt != '':
                res = self.load_state_dict(torch.load(pretrained_ckpt), strict=False)
                assert len(res.missing_keys) == 0 and all([k.startswith('token_extractor') for k in res.unexpected_keys])
            self.token_extractor = TokenExtractor(self.bert.config.hidden_size, token_rep_dim, dropout, dropout_pos,
                                                 span_output)

    def forward(self, token_id, atten_mask):
        outputs = self.bert(token_id, attention_mask=atten_mask, output_hidden_states=True, return_dict=True)
        bert_embedding = outputs['hidden_states'][-1]
        return bert_embedding


def nn_metric(metric, query_emb, in_class_support_emb):
    if query_emb.size(0)*query_emb.size(1) < 20480 and in_class_support_emb.size(0) < 20480:
        if metric == 'dot':
            in_class_sim = query_emb.inner(in_class_support_emb)
        elif metric == 'cosine':
            in_class_sim = query_emb.inner(in_class_support_emb)
        elif metric == 'L2':
            in_class_sim = -torch.cdist(query_emb, in_class_support_emb.unsqueeze(0)).pow(2)
        else:
            raise Exception('Unknown Distance Metric')
        return in_class_sim.max(dim=2)[0]
    else:
        chunk_nn_sim = []
        for query_chunk in torch.split(query_emb.view(-1, query_emb.size(-1)), 2048, dim=0):
            if metric == 'dot':
                in_class_sim = query_chunk.inner(in_class_support_emb)
            elif metric == 'cosine':
                in_class_sim = query_chunk.inner(in_class_support_emb)
            elif metric == 'L2':
                in_class_sim = -torch.cdist(query_chunk, in_class_support_emb).pow(2)
            else:
                raise Exception('Unknown Distance Metric')
            chunk_max_v = in_class_sim.max(dim=1)[0]
            chunk_nn_sim.append(chunk_max_v)
        chunk_nn_sim = torch.cat(chunk_nn_sim, 0).view(query_emb.size(0), query_emb.size(1))
        return chunk_nn_sim