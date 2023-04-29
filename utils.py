import datetime
import json
import logging
import numpy as np
import os
import pytorch_lightning as pl
import random
import string
import sys
import torch
from collections import Counter, defaultdict
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import ConcatDataset
from tqdm import tqdm
from transformers import AutoTokenizer
from copy import deepcopy


def sent_loader(task, split):
    buf = []
    with open(f'Few-NERD/data/{task}/{split}.txt') as f:
        for line in f:
            line = line.strip()
            if line:
                word, label = line.split('\t')
                buf.append((word, label))
            else:
                words, labels = zip(*buf)
                yield words, labels, {}
                buf = []
    #previous will lose last sent because there is not empty line
    #but will not rerun all affected expriment
    if buf:
        words, labels = zip(*buf)
        yield words, labels, {}


def episode_loader(task, split, n, k):
    with open(f'Few-NERD/data/episode-data/{task}/{split}_{n}_{k}.jsonl') as f:
        for line in f:
            line = json.loads(line.strip())
            query = line['query']
            query = [(words, labels, {}) for words, labels in zip(query['word'], query['label'])]
            support = line['support']
            support = [(words, labels, {}) for words, labels in zip(support['word'], support['label'])]
            types = line['types']
            yield types, support, query


def truecase_episode_loader(mode, split, N, K):
    text_2_sent = {}
    for words, labels, addtional_info in sent_loader(mode, split):
        sent_text = ' '.join([w.lower() for w in words])
        text_2_sent[sent_text] = (words, labels)

    for types, support, query in episode_loader(mode, split, N, K):
        truecase_support, truecase_query = [], []
        for words, labels, addtional_info in support:
            sent_text = ' '.join([w for w in words])
            assert sent_text in text_2_sent
            truecase_words, truecase_labels = text_2_sent[sent_text]
            assert len(truecase_words) == len(words)
            assert all([tw.lower() == w for tw, w in zip(truecase_words, words)])
            truecase_support.append((truecase_words, labels, {}))

        for words, labels, addtional_info in query:
            sent_text = ' '.join([w for w in words])
            assert sent_text in text_2_sent
            truecase_words, truecase_labels = text_2_sent[sent_text]
            assert len(truecase_words) == len(words)
            assert all([tw.lower() == w for tw, w in zip(truecase_words, words)])
            truecase_query.append((truecase_words, labels, {}))

        yield types, truecase_support, truecase_query


def get_io_spans(labels):
    spans = []
    span_count = Counter()

    i = 0
    while i < len(labels):
        if type(labels[i]) == str:
            if labels[i] != 'O' and labels[i] != '##NULL##':
                start = i
                current_label = labels[i]
                i += 1
                while i < len(labels) and labels[i] == current_label:
                    i += 1
                spans.append((start, i - 1, current_label))
                span_count[current_label] += 1
            else:
                i += 1
        elif type(labels[i]) == int:
            assert labels[i] >= -1
            if labels[i] != 0 and labels[i] != -1:
                start = i
                current_label = labels[i]
                i += 1
                while i < len(labels) and labels[i] == current_label:
                    i += 1
                spans.append((start, i - 1, current_label))
                span_count[current_label] += 1
            else:
                i += 1
        elif type(labels[i]) == tuple:
            assert labels[i][0] >= -1
            if labels[i][0] != 0 and labels[i][0] != -1:
                start = i
                current_label = labels[i]
                i += 1
                while i < len(labels) and labels[i][0] == current_label[0]:
                    i += 1
                spans.append((start, i - 1, current_label))
                span_count[current_label[0]] += 1
            else:
                i += 1
        else:
            raise Exception('Unknown label type')

    return spans, span_count


class LabelEncoder:
    def __init__(self, labels):
        self.idx_to_item = {-1: '##NULL##', 0: 'O'}
        self.item_to_idx = {'##NULL##': -1, 'O': 0}
        for i, l in enumerate(labels):
            self.add(l, i + 1)

    def add(self, item, idx):
        assert item not in self.item_to_idx
        self.item_to_idx[item] = idx
        self.idx_to_item[idx] = item

    def get(self, idx):
        if type(idx) in [list, tuple]:
            return [self.get(i) for i in idx]
        else:
            return self.idx_to_item[idx]

    def index(self, item):
        if type(item) in [list, tuple]:
            return [self.index(i) for i in item]
        else:
            return self.item_to_idx[item]

    def __len__(self):
        return len(self.item_to_idx)

    def __str__(self):
        return str(self.item_to_idx)


class Sentence:
    def __init__(self, words, labels, **additional_info):
        self.words = words
        self.labels = labels
        self.additional_info = additional_info
        self.spans, self.span_count = get_io_spans(labels)

    def sent_tokenize_split(self, bert_tokenizer, max_length):
        token_ids = [bert_tokenizer.cls_token_id]
        token_labels = ['##NULL##']
        word_masks = [0]
        atten_masks = [1]

        for i, (w, l) in enumerate(zip(self.words, self.labels)):
            subword = bert_tokenizer.tokenize(w)
            assert len(subword) <= max_length - 2, subword
            # todo: filter too long word, find subword seq >= 96 in ontonote or roberta, too old old, no fix
            # if len(subword) >= 10:
            #     print(w,'long subword', len(subword))
            if len(subword) == 0:
                # for FEW-NERD and bert-base-(un)cased, all zero subword words are O and not in test set
                # so remove it should not make a difference for test and is OK
                # for SNIPS, one zero subword word in test are non-O(playlist), we need to use bio tag at eval so add
                # zero_subword_remove to make log removal position and padding O afterwards
                if 'zero_subword_remove' not in self.additional_info:
                    self.additional_info['zero_subword_remove'] = [(i,l)]
                else:
                    self.additional_info['zero_subword_remove'].append((i, l))
                print('empty subword', [ord(c) for c in w], l, self.additional_info)
                continue

            #todo: should be max_length - 1 since cls is also counted but this bug exist too long so no fix
            if len(token_ids) + len(subword) > max_length - 1:
                token_ids.append(bert_tokenizer.sep_token_id)
                token_labels.append('##NULL##')
                word_masks.append(0)
                atten_masks.append(1)

                # if sum(word_masks) <= 5:
                #     print(str(self),' '.join(subword_arr), 'little word sentence')

                assert len(token_ids) == len(token_labels) == len(word_masks) == len(atten_masks)
                yield token_ids, token_labels, word_masks, atten_masks

                token_ids = [bert_tokenizer.cls_token_id]
                token_labels = ['##NULL##']
                word_masks = [0]
                atten_masks = [1]

                #old code will drop the first word, we only rerun the evaluation due to time constraint
                token_ids.extend(bert_tokenizer.convert_tokens_to_ids(subword))
                token_labels.extend([l] * len(subword))
                word_masks.extend([1] + [0] * (len(subword) - 1))
                atten_masks.extend([1] * len(subword))
            else:
                token_ids.extend(bert_tokenizer.convert_tokens_to_ids(subword))
                token_labels.extend([l] * len(subword))
                word_masks.extend([1] + [0] * (len(subword) - 1))
                atten_masks.extend([1] * len(subword))

        if len(token_ids) > 0:
            token_ids.append(bert_tokenizer.sep_token_id)
            token_labels.append('##NULL##')
            word_masks.append(0)
            atten_masks.append(1)

            # if sum(word_masks) <= 5:
            #     print(str(self),' '.join(subword_arr), 'little word sentence')

            assert len(token_ids) == len(token_labels) == len(word_masks) == len(atten_masks)
            yield token_ids, token_labels, word_masks, atten_masks

    def split_tokenize(self, bert_tokenizer, max_length):
        if not hasattr(self, 'samples'):
            self.samples = []
            for ret in self.sent_tokenize_split(bert_tokenizer, max_length):
                self.samples.append(ret)
            return self.samples
        else:
            return self.samples

    def json_repr(self):
        return self.words, self.labels, self.additional_info

    def __str__(self):
        rep = []
        for w, l in zip(self.words, self.labels):
            if l == 'O':
                rep.append(w)
            else:
                rep.append(f'{w}[{l}]')
        return ' '.join(rep)


class SentenceDataset(Dataset):
    def __init__(self, loader, backbone_name, max_length):
        self.bert_tokenizer = AutoTokenizer.from_pretrained(backbone_name, use_fast=False)
        self.max_length = max_length

        self.types = set()
        self.samples = []
        for words, labels, additional_info in tqdm(loader):
            self.types.update(labels)
            self.samples.append(Sentence(words, labels, **additional_info))
        self.types.remove('O')
        self.types = sorted(list(self.types))
        self.label_encoder = LabelEncoder(self.types)

    def __getitem__(self, index):
        dataset = {'token_id': [], 'label': [], 'atten_mask': [], 'word_mask': [], 'types': self.types}

        sent = self.samples[index]
        token_ids, token_labels, word_masks, atten_masks = zip(
            *sent.split_tokenize(self.bert_tokenizer, self.max_length))
        dataset['token_id'].extend(token_ids)
        dataset['label'].extend(self.label_encoder.index(token_labels))
        dataset['word_mask'].extend(word_masks)
        dataset['atten_mask'].extend(atten_masks)
        # total_num = 0
        # for word_mask in word_masks:
        #     total_num += sum(word_mask)
        # if total_num+len(sent.additional_info.get('zero_subword_remove',[])) != len(sent.words):
        #     print(total_num, len(sent.words), sent.json_repr())
        #     raise Exception('Mismatch label token num')
        dataset['json'] = json.dumps(sent.json_repr())
        return dataset

    def __len__(self):
        return len(self.samples)


def concat_dataset_until(dataset, target_size):
    list_of_dataset = []
    list_size = 0
    while list_size < target_size:
        list_of_dataset.append(dataset)
        list_size += len(dataset)

    extended_dataset = ConcatDataset(list_of_dataset)
    return extended_dataset


class SentenceSpanDataset(Dataset):
    def __init__(self, sent_dataset, neg_rate, max_len, length_limited_full_span):
        self.sent_dataset = sent_dataset
        #         print(f'original dataset len:{self.sent_dataset_len}')
        self.neg_rate = neg_rate
        self.max_len = max_len
        assert self.neg_rate > 0
        self.length_limited_full_span = length_limited_full_span

    def __getitem__(self, index):
        dataset = self.sent_dataset[index]
        random.seed(index)
        assert 'span' not in dataset
        dataset['span'], dataset['word_len'], dataset['subword_len'] = [], [], []
        for token_labels, word_mask in zip(dataset['label'], dataset['word_mask']):
            valid_pos = [i for i, flag in enumerate(word_mask) if flag != 0]
            valid_label = [token_labels[idx] for idx in valid_pos]
            spans, span_count = get_io_spans(valid_label)
            valid_pos_span = [(start, end, label) for start, end, label in spans if end - start + 1 <= self.max_len]
            valid_reject_set = set([(start, end) for start, end, label in spans])

            valid_neg_span = []
            for i in range(len(valid_label)):
                for j in range(i, len(valid_label)):
                    if j - i + 1 <= self.max_len and (i, j) not in valid_reject_set:
                        valid_neg_span.append((i, j, 0))

            if len(valid_neg_span) > 0:
                neg_num = int(len(valid_label) * self.neg_rate) + 1
                sample_num = min(neg_num, len(valid_neg_span))
                sampled_valid_neg_span = random.sample(valid_neg_span, sample_num)
            else:
                sampled_valid_neg_span = valid_neg_span

            full_span = np.ones((len(word_mask), len(word_mask), 2), dtype=np.int64) * -1
            for i, j, l in valid_pos_span:
                word_i, word_j = valid_pos[i], valid_pos[j]
                full_span[word_i, word_j, 0] = l

            if self.length_limited_full_span:
                for i, j, l in valid_neg_span:
                    word_i, word_j = valid_pos[i], valid_pos[j]
                    assert l == 0
                    full_span[word_i, word_j, 0] = l
            else:
                for i in range(len(valid_label)):
                    for j in range(i, len(valid_label)):
                        if (i, j) not in valid_reject_set:
                            word_i, word_j = valid_pos[i], valid_pos[j]
                            full_span[word_i, word_j, 0] = 0

            # Sample mask
            full_span[:, :, 1] = 0
            for i, j, l in valid_pos_span + sampled_valid_neg_span:
                word_i, word_j = valid_pos[i], valid_pos[j]
                assert full_span[word_i, word_j, 0] == l
                full_span[word_i, word_j, 1] = 1

            dataset['span'].append(full_span)
            dataset['word_len'].append(len(valid_label))
            dataset['subword_len'].append(len(token_labels) - 2)
        return dataset

    def __len__(self):
        return len(self.sent_dataset)


class Episode:
    def __init__(self, query, support, types):
        self.types = sorted(types)

        self.query_counter = Counter()
        self.query = query
        for sent in self.query:
            self.query_counter += sent.span_count

        self.support_counter = Counter()
        self.support = support
        for sent in self.support:
            self.support_counter += sent.span_count

    def json_repr(self):
        support = [sent.json_repr() for sent in self.query]
        query = [sent.json_repr() for sent in self.query]
        return {'types':self.types, 'support':support, 'query':query}

    def __str__(self):
        rep = []
        rep.append('{0:<30}{1:<30}{2:<30}'.format('label', 'support', 'query'))
        for l in self.types:
            rep.append(f'{l:<30}{self.support_counter[l]:<30}{self.query_counter[l]:<30}')
        rep.append('{0:<30}{1:<30}{2:<30}'.format('total', len(self.support), len(self.query)))
        return '\n'.join(rep)

    def to_data(self, bert_tokenizer, max_length):
        label_encoder = LabelEncoder(self.types)

        dataset = {'token_id': [], 'label': [], 'atten_mask': [], 'word_mask': [], 'query_id': [], 'sent_flag': [],
                   'types': self.types}
        for sent in self.support:
            token_ids, token_labels, word_masks, atten_masks = zip(*sent.split_tokenize(bert_tokenizer, max_length))
            dataset['token_id'].extend(token_ids)
            dataset['label'].extend(label_encoder.index(token_labels))
            dataset['word_mask'].extend(word_masks)
            dataset['atten_mask'].extend(atten_masks)
            dataset['sent_flag'].extend([0 for j in range(len(token_ids))])

        last_sent_id = 0
        for sent in self.query:
            token_ids, token_labels, word_masks, atten_masks = zip(*sent.split_tokenize(bert_tokenizer, max_length))
            dataset['token_id'].extend(token_ids)
            dataset['label'].extend(label_encoder.index(token_labels))
            dataset['word_mask'].extend(word_masks)
            dataset['atten_mask'].extend(atten_masks)
            dataset['query_id'].extend([last_sent_id for j in range(len(token_ids))])
            dataset['sent_flag'].extend([1 for j in range(len(token_ids))])
            last_sent_id += 1
        dataset['json'] = json.dumps(self.json_repr())
        return dataset


class EpisodeDataset(Dataset):
    def __init__(self, loader, backbone_name, max_length):
        self.bert_tokenizer = AutoTokenizer.from_pretrained(backbone_name, use_fast=False)
        assert self.bert_tokenizer.pad_token_id == 0, 'padding token is not 0'
        self.max_length = max_length

        self.samples = []
        self.types = set()
        for types, support, query in tqdm(loader):
            self.add_sample(types, support, query)
            self.types.update(types)

        self.types = sorted(list(self.types))
        self.label_encoder = LabelEncoder(self.types)

    def add_sample(self, types, support, query):
        support_sent = [Sentence(words, labels, **additional_info) for words, labels, additional_info in support]
        query_sent = [Sentence(words, labels, **additional_info) for words, labels, additional_info in query]
        episode = Episode(query_sent, support_sent, types)
        self.samples.append(episode)

    def __getitem__(self, index):
        #         print(index, torch.utils.data.get_worker_info().id)
        episode = self.samples[index]
        assert len(episode.support) > 0 and len(episode.query) > 0, 'empty support or query'

        dataset = episode.to_data(self.bert_tokenizer, self.max_length)
        dataset['index'] = index
        dataset['global_label'] = []
        label_encoder = LabelEncoder(dataset['types'])
        for token_labels in dataset['label']:
            token_str_labels = label_encoder.get(token_labels)
            token_global_labels = self.label_encoder.index(token_str_labels)
            dataset['global_label'].append(token_global_labels)

        return dataset

    def __len__(self):
        return len(self.samples)


class EpisodeSamplingDataset(Dataset):
    def __init__(self, N, K, Q, loader, backbone_name, max_length):
        self.K = K
        self.N = N
        self.Q = Q

        self.bert_tokenizer = AutoTokenizer.from_pretrained(backbone_name, use_fast=False)
        assert self.bert_tokenizer.pad_token_id == 0, 'padding token is not 0'
        self.max_length = max_length

        self.samples = []
        self.sample_counter = Counter()
        self.class_to_idx = defaultdict(lambda: set())
        self.types = []

        for words, labels, additional_info in tqdm(loader):
            self.add_sample(Sentence(words, labels, **additional_info))

        self.types = sorted(list(self.class_to_idx.keys()))
        self.label_encoder = LabelEncoder(self.types)

        self.relax_k = 2

    def add_sample(self, sample):
        self.sample_counter += sample.span_count
        self.samples.append(sample)
        for k in sample.span_count:
            self.types.append(k)
            self.class_to_idx[k].add(len(self.samples) - 1)

    def valid(self, sample, set_counter, target_classes, k_shot):
        if len(sample.span_count) == 0:
            # NO EMPTY IN SAMPLING
            return False
        isvalid = False
        for c in sample.span_count:
            if c not in target_classes:
                return False
            if sample.span_count[c] + set_counter[c] > self.relax_k * k_shot:
                return False
            if set_counter[c] < k_shot:
                isvalid = True
        return isvalid

    @staticmethod
    def finish(set_counter, n_way, k_shot):
        if len(set_counter) < n_way:
            return False
        for k in set_counter:
            if set_counter[k] < k_shot:
                return False
        return True

    def gen_candidate_idx(self, target_classes):
        candidate_idx = set()
        # for c in self.types:
        #     if c in target_classes:
        #         candidate_idx = candidate_idx | self.class_to_idx[c]
        #     else:
        #         candidate_idx = candidate_idx - self.class_to_idx[c]
        # old code not fully elimate impossible sentence since it depends on self.types's order
        for c in self.types:
            if c in target_classes:
                candidate_idx = candidate_idx | self.class_to_idx[c]
        for c in self.types:
            if c not in target_classes:
                candidate_idx = candidate_idx - self.class_to_idx[c]
        return sorted(list(candidate_idx))

    def gen_episode(self):
        support_counter = Counter()
        support_idx = []
        query_counter = Counter()
        query_idx = []
        target_classes = random.sample(self.types, self.N)
        candidates = self.gen_candidate_idx(target_classes)
        while not candidates:
            target_classes = random.sample(self.types, self.N)
            candidates = self.gen_candidate_idx(target_classes)

        while not self.finish(support_counter, self.N, self.K):
            index = random.choice(candidates)
            if index not in support_idx:
                if self.valid(self.samples[index], support_counter, target_classes, self.K):
                    support_counter += self.samples[index].span_count
                    support_idx.append(index)

        while not self.finish(query_counter, self.N, self.Q):
            index = random.choice(candidates)
            if index not in query_idx and index not in support_idx:
                if self.valid(self.samples[index], query_counter, target_classes, self.Q):
                    query_counter += self.samples[index].span_count
                    query_idx.append(index)
        #         print(support_idx, query_idx)
        support_samples = [self.samples[idx] for idx in support_idx]
        query_samples = [self.samples[idx] for idx in query_idx]
        return target_classes, support_samples, query_samples

    def __getitem__(self, index):
        random.seed(index)
        #         print(index, torch.utils.data.get_worker_info().id)
        target_classes, support_samples, query_samples = self.gen_episode()

        episode = Episode(query_samples, support_samples, target_classes)
        assert len(episode.support) > 0 and len(episode.query) > 0, 'empty support or query'

        dataset = episode.to_data(self.bert_tokenizer, self.max_length)
        dataset['index'] = index
        dataset['global_label'] = []
        label_encoder = LabelEncoder(dataset['types'])
        for token_labels in dataset['label']:
            token_str_labels = label_encoder.get(token_labels)
            token_global_labels = self.label_encoder.index(token_str_labels)
            dataset['global_label'].append(token_global_labels)

        return dataset

    def __len__(self):
        return 1000000


class EpisodeSpanDataset(Dataset):
    def __init__(self, episode_dataset, neg_rate, max_len, length_limited_full_span):
        self.episode_dataset = episode_dataset
        self.neg_rate = neg_rate
        self.max_len = max_len
        self.length_limited_full_span = length_limited_full_span

    def __getitem__(self, index):
        dataset = self.episode_dataset[index]
        random.seed(index)
        assert 'span' not in dataset
        dataset['span'], dataset['word_len'], dataset['subword_len'] = [], [], []
        for token_labels, token_global_labels, word_mask in zip(dataset['label'], dataset['global_label'],
                                                                dataset['word_mask']):
            valid_pos = [i for i, flag in enumerate(word_mask) if flag != 0]
            valid_label = [(token_labels[idx], token_global_labels[idx]) for idx in valid_pos]
            spans, span_count = get_io_spans(valid_label)
            valid_pos_span = [(start, end, label, global_label) for start, end, (label, global_label) in spans
                              if end - start + 1 <= self.max_len]
            valid_reject_set = set([(start, end) for start, end, (label, global_label) in spans])

            valid_neg_span = []
            for i in range(len(valid_label)):
                for j in range(i, len(valid_label)):
                    if j - i + 1 <= self.max_len and (i, j) not in valid_reject_set:
                        valid_neg_span.append((i, j, 0, 0))

            if len(valid_neg_span) > 0 and self.neg_rate > 0:
                neg_num = int(len(valid_label) * self.neg_rate) + 1
                sample_num = min(neg_num, len(valid_neg_span))
                sampled_valid_neg_span = random.sample(valid_neg_span, sample_num)
            else:
                sampled_valid_neg_span = valid_neg_span

            full_span = np.ones((len(word_mask), len(word_mask), 3), dtype=np.int64) * -1
            for i, j, l, gl in valid_pos_span:
                word_i, word_j = valid_pos[i], valid_pos[j]
                full_span[word_i, word_j, 0] = l
                full_span[word_i, word_j, 1] = gl

            if self.length_limited_full_span:
                for i, j, l, gl in valid_neg_span:
                    word_i, word_j = valid_pos[i], valid_pos[j]
                    assert l == 0 and gl == 0
                    full_span[word_i, word_j, 0] = l
                    full_span[word_i, word_j, 1] = gl
            else:
                for i in range(len(valid_label)):
                    for j in range(i, len(valid_label)):
                        if (i, j) not in valid_reject_set:
                            word_i, word_j = valid_pos[i], valid_pos[j]
                            full_span[word_i, word_j, 0] = 0
                            full_span[word_i, word_j, 1] = 0

            # Sample mask
            full_span[:, :, 2] = 0
            for i, j, l, gl in valid_pos_span + sampled_valid_neg_span:
                word_i, word_j = valid_pos[i], valid_pos[j]
                assert full_span[word_i, word_j, 0] == l
                assert full_span[word_i, word_j, 1] == gl, (full_span[word_i, word_j, 1], l, gl)
                full_span[word_i, word_j, 2] = 1

            dataset['span'].append(full_span)
            dataset['word_len'].append(len(valid_label))
            dataset['subword_len'].append(len(token_labels) - 2)

        return dataset

    def __len__(self):
        return len(self.episode_dataset)


def padding(list_of_list, pad_sign):
    sizes = [len(outer_list) for outer_list in list_of_list]
    max_size = max(sizes)
    pad_sizes = [max_size - size for size in sizes]
    return [outer_list + [pad_sign] * pad_size for outer_list, pad_size in zip(list_of_list, pad_sizes)]


def label_padding(list_of_tensor, max_dim, pad_sign):
    padded_tensor_list = []
    for i, tensor in enumerate(list_of_tensor):
        padded_tensor = np.ones((max_dim, max_dim, tensor.shape[-1]), dtype=np.int64) * pad_sign
        padded_tensor[:tensor.shape[0], :tensor.shape[1]] = tensor
        padded_tensor_list.append(padded_tensor)
    return np.stack(padded_tensor_list)


def episode_collate_fn(data):
    dataset = {'token_id': [], 'label': [], 'global_label': [], 'atten_mask': [], 'word_mask': [], 'query_id': [],
               'sent_flag': [], 'batch_id': [], 'index': [], 'batch_size': len(data), 'types': [], 'jsons':[]}

    last_batch_id = 0
    for episode_data in data:
        dataset['token_id'].extend(episode_data['token_id'])
        dataset['label'].extend(episode_data['label'])
        dataset['global_label'].extend(episode_data['global_label'])
        dataset['atten_mask'].extend(episode_data['atten_mask'])
        dataset['word_mask'].extend(episode_data['word_mask'])
        dataset['sent_flag'].extend(episode_data['sent_flag'])
        dataset['query_id'].append(episode_data['query_id'])
        dataset['types'].append(episode_data['types'])
        dataset['jsons'].append(episode_data['json'])
        dataset['index'].append(episode_data['index'])
        dataset['batch_id'].extend([last_batch_id] * len(episode_data['sent_flag']))
        last_batch_id += 1

    dataset['token_id'] = torch.LongTensor(padding(dataset['token_id'], 0))
    dataset['label'] = torch.LongTensor(padding(dataset['label'], -1))
    dataset['global_label'] = torch.LongTensor(padding(dataset['global_label'], -1))
    dataset['atten_mask'] = torch.LongTensor(padding(dataset['atten_mask'], 0))
    dataset['word_mask'] = torch.LongTensor(padding(dataset['word_mask'], 0))
    dataset['sent_flag'] = torch.LongTensor(dataset['sent_flag'])
    dataset['batch_id'] = torch.LongTensor(dataset['batch_id'])

    has_span = 'span' in data[0]
    if has_span:
        assert all(['span' in episode_data for episode_data in data])
        seq_num, max_token_len = dataset['token_id'].size()
        full_span = []
        word_len = []
        subword_len = []

        for episode_data in data:
            full_span.extend(episode_data['span'])
            word_len.extend(episode_data['word_len'])
            subword_len.extend(episode_data['subword_len'])

        assert len(full_span) == dataset['token_id'].size(0)
        dataset['full_span'] = torch.from_numpy(label_padding(full_span, max_token_len, -1))
        dataset['word_len'] = torch.LongTensor(word_len)
        dataset['subword_len'] = torch.LongTensor(subword_len)

    return dataset


def sentence_collate_fn(data):
    dataset = {'token_id': [], 'label': [], 'atten_mask': [], 'word_mask': [], 'sent_id': [], 'batch_size': len(data),
               'jsons':[]}

    last_sent_id = 0
    for sent_data in data:
        dataset['token_id'].extend(sent_data['token_id'])
        dataset['label'].extend(sent_data['label'])
        dataset['atten_mask'].extend(sent_data['atten_mask'])
        dataset['word_mask'].extend(sent_data['word_mask'])
        dataset['sent_id'].extend([last_sent_id] * len(sent_data['token_id']))
        dataset['jsons'].append(sent_data['json'])
        last_sent_id += 1
    assert last_sent_id == len(data)

    dataset['token_id'] = torch.LongTensor(padding(dataset['token_id'], 0))
    dataset['label'] = torch.LongTensor(padding(dataset['label'], -1))
    dataset['atten_mask'] = torch.LongTensor(padding(dataset['atten_mask'], 0))
    dataset['word_mask'] = torch.LongTensor(padding(dataset['word_mask'], 0))

    has_span = 'span' in data[0]
    if has_span:
        assert all(['span' in sent_data for sent_data in data])
        batch_size, max_token_len = dataset['token_id'].size()

        full_span = []
        word_len = []
        subword_len = []

        for episode_data in data:
            full_span.extend(episode_data['span'])
            word_len.extend(episode_data['word_len'])
            subword_len.extend(episode_data['subword_len'])

        assert len(full_span) == dataset['token_id'].size(0)
        dataset['full_span'] = torch.from_numpy(label_padding(full_span, max_token_len, -1))
        dataset['word_len'] = torch.LongTensor(word_len)
        dataset['subword_len'] = torch.LongTensor(subword_len)

    return dataset


def validate_bio(ys):
    tags, labels = ['O'], ['O']
    for y in ys:
        if y == 'O':
            tags.append(y)
            labels.append(y)
        else:
            tags.append(y[0])
            labels.append(y[2:])
    for i in range(len(ys)):
        assert (tags[i], tags[i+1]) != ('O', 'I')
        if (tags[i], tags[i+1]) in [('I', 'I'), ('B', 'I')]:
            assert labels[i] == labels[i+1]
    return ys


def convert_bio_io(bio_ys):
    validate_bio(bio_ys)
    io_ys = deepcopy(bio_ys)
    for i, y in enumerate(io_ys):
        if y == 'O':
            continue
        else:
            tag, label = y[0], y[2:]
            assert '-' not in label
            io_ys[i] = label
    assert len(io_ys) == len(bio_ys)
    cmp_bio_io_span(io_ys, bio_ys)
    return io_ys


def cmp_bio_io_span(io_ys, bio_ys):
    io_spans, io_span_count = get_io_spans(io_ys)
    bio_spans, bio_span_count = get_bio_spans(bio_ys)
    if io_spans != bio_spans:
        for i,j,l in bio_spans:
            if (i,j,l) not in io_spans:
                print(f'({i},{j},{l}) missing')
        for i,j,l in io_spans:
            if (i,j,l) not in bio_spans:
                print(f'({i},{j},{l}) new')


def SNIPS_CV_loader(split, cv_i, K):
    split_2_types = {
        'AddToPlaylist': {'playlist', 'music_item', 'entity_name', 'playlist_owner', 'artist'},
        'RateBook': {'best_rating', 'rating_value', 'object_type', 'object_select', 'object_part_of_series_type',
                     'rating_unit', 'object_name'},
        'SearchScreeningEvent': {'timeRange', 'movie_type', 'object_type', 'object_location_type', 'spatial_relation',
                                 'movie_name', 'location_name'},
        'BookRestaurant': {'party_size_number', 'party_size_description', 'sort', 'spatial_relation', 'state',
                           'timeRange', 'cuisine', 'poi', 'facility', 'city', 'restaurant_name', 'country',
                           'restaurant_type', 'served_dish'},
        'SearchCreativeWork': {'object_type', 'object_name'},
        'PlayMusic': {'playlist', 'genre', 'music_item', 'service', 'album', 'track', 'year', 'sort', 'artist'},
        'GetWeather': {'spatial_relation', 'state', 'timeRange', 'condition_description', 'condition_temperature',
                       'current_location', 'city', 'country', 'geographic_poi'}
    }
    if K == 5:
        with open(f'FewShotTagging/ACL2020data/xval_snips_shot_5/snips-{split}-{cv_i}-shot-5.json') as f:
            data = json.load(f)
            named_episodes = {}
            for name, episodes in data.items():
                named_episodes[name] = []
                for episode in episodes:
                    support = episode['support']
                    support = list(zip(support['seq_ins'], support['seq_outs']))
                    support_types = set([tag[2:] if tag != 'O' else 'O' for words, labels in support for tag in labels])
                    support = [(words, convert_bio_io(labels), {'bio_labels':labels}) for words, labels in support]
                    query = episode['batch']
                    query = list(zip(query['seq_ins'], query['seq_outs']))
                    query_types = set([tag[2:] if tag != 'O' else 'O' for words, labels in query for tag in labels])
                    query = [(words, convert_bio_io(labels), {'bio_labels':labels}) for words, labels in query]
                    assert len(query_types - support_types) == 0, (support_types, query_types)
                    named_episodes[name].append((split_2_types[name], support, query))
            return named_episodes

    elif K == 1:
        with open(f'FewShotTagging/ACL2020data/xval_snips/snips_{split}_{cv_i}.json') as f:
            data = json.load(f)
            named_episodes = {}
            for name, episodes in data.items():
                named_episodes[name] = []
                for episode in episodes:
                    support = episode['support']
                    support = list(zip(support['seq_ins'], support['seq_outs']))
                    support_types = set([tag[2:] if tag != 'O' else 'O' for words, labels in support for tag in labels])
                    support = [(words, convert_bio_io(labels), {'bio_labels':labels}) for words, labels in support]
                    query = episode['batch']
                    query = list(zip(query['seq_ins'], query['seq_outs']))
                    query_types = set([tag[2:] if tag != 'O' else 'O' for words, labels in query for tag in labels])
                    query = [(words, convert_bio_io(labels), {'bio_labels':labels}) for words, labels in query]
                    assert len(query_types - support_types) == 0, (support_types, query_types)
                    named_episodes[name].append((split_2_types[name], support, query))
            return named_episodes


def get_bio_spans(labels):
    spans = []
    span_count = Counter()

    i = 0
    while i < len(labels):
        assert type(labels[i]) == str, 'Unknown label type'
        assert labels[i] != '##NULL##', 'Null label'
        if labels[i] == 'O':
            tag, label = labels[i], labels[i]
        else:
            #B-creative-work does not work for labels[i].split('-')
            assert labels[i][0] in ['B','I'] and labels[i][1] == '-', labels
            tag, label = labels[i][0], labels[i][2:]
            assert label != ''

        if tag == 'B':
            start = i
            current_label = label
            i += 1
            while i < len(labels):
                if labels[i] == 'O':
                    tag, label = labels[i], labels[i]
                else:
                    assert labels[i][0] in ['B', 'I'] and labels[i][1] == '-'
                    tag, label = labels[i][0], labels[i][2:]
                    assert label != ''
                if tag == 'I'  and label == current_label:
                    i += 1
                else:
                    break
            spans.append((start, i - 1, current_label))
            span_count[current_label] += 1
        else:
            i += 1

    return spans, span_count


def SNIPS_sent_loader(subset_names):
    split_2_types = {
        'AddToPlaylist':{'playlist', 'music_item', 'entity_name', 'playlist_owner', 'artist'},
        'RateBook':{'best_rating', 'rating_value', 'object_type', 'object_select', 'object_part_of_series_type', 'rating_unit', 'object_name'},
        'SearchScreeningEvent':{'timeRange', 'movie_type', 'object_type', 'object_location_type', 'spatial_relation', 'movie_name', 'location_name'},
        'BookRestaurant':{'party_size_number', 'party_size_description', 'sort', 'spatial_relation', 'state', 'timeRange', 'cuisine', 'poi', 'facility', 'city', 'restaurant_name', 'country', 'restaurant_type', 'served_dish'},
        'SearchCreativeWork':{'object_type', 'object_name'},
        'PlayMusic':{'playlist', 'genre', 'music_item', 'service', 'album', 'track', 'year', 'sort', 'artist'},
        'GetWeather':{'spatial_relation', 'state', 'timeRange', 'condition_description', 'condition_temperature', 'current_location', 'city', 'country', 'geographic_poi'}
    }
    sents = []
    for split in ['train', 'valid', 'test']:
        with open(f'SlotGated-SLU/data/snips/{split}/seq.in') as f:
            words = [line.strip().split() for line in f]
        with open(f'SlotGated-SLU/data/snips/{split}/seq.out') as f:
            labels = [line.strip().split() for line in f]
        with open(f'SlotGated-SLU/data/snips/{split}/label') as f:
            intents = [line.strip() for line in f]
        sents.extend(zip(intents, words, labels))
    for intent, words, labels in sents:
        assert intent in split_2_types
        assert len(words) == len(labels)
        if intent in subset_names:
            yield words, convert_bio_io(labels), {'bio_labels':labels}




def load_ontonote(split):
    from fastNLP.io.loader.conll import OntoNotesNERLoader

    root_path = 'ontonote/'
    train_path = root_path + '/train.txt'
    dev_path = root_path + '/dev.txt'
    test_path = root_path + '/test.txt'

    for instance in OntoNotesNERLoader()._load(f'{root_path}/{split}.txt'):
        words, labels = instance['raw_words'], instance['target']
        labels = list(map(bio_2_io, labels))
        yield words, convert_bio_io(labels), {'bio_labels':labels}


def load_conll(split):
    from fastNLP.io.loader.conll import Conll2003NERLoader

    root_path = 'conll/'
    train_path = root_path + '/train.txt'
    dev_path = root_path + '/dev.txt'
    test_path = root_path + '/test.txt'

    for instance in Conll2003NERLoader()._load(f'{root_path}/{split}.txt'):
        words, labels = instance['raw_words'], instance['target']
        yield words, convert_bio_io(labels), {'bio_labels':labels}


class PerClassEpisodeSamplingDataset(EpisodeSamplingDataset):
    def __init__(self, N, K, Q, loader, backbone_name, max_length):
        super().__init__(N, K, Q, loader, backbone_name, max_length)
        self.relax_k = 2

    def gen_candidate_idx(self, target_classes):
        candidate_idx = set()
        per_class_cand = []
        for c in self.types:
            if c in target_classes:
                candidate_idx = candidate_idx | self.class_to_idx[c]
        for c in self.types:
            if c not in target_classes:
                candidate_idx = candidate_idx - self.class_to_idx[c]
        for c in target_classes:
            valid_cand_idx = list(self.class_to_idx[c] & candidate_idx)
            per_class_cand.append((valid_cand_idx, c, len(valid_cand_idx)))

        per_class_cand = sorted(per_class_cand, key=lambda x: x[2])
        return per_class_cand

    def select(self, set_counter, set_idx, n_way, k_shot, target_classes, candidates, select_class):
        if set_counter[select_class] >= k_shot:
            return True
        new_cand = []
        total_available = 0
        for index in random.sample(candidates, k=len(candidates)):
            if index not in set_idx and self.valid(self.samples[index], set_counter, target_classes, k_shot):
                new_cand.append(index)
                total_available += self.samples[index].span_count[select_class]
        if set_counter[select_class] + total_available < k_shot:
            return False
        for i, index in enumerate(new_cand):
            set_counter += self.samples[index].span_count
            set_idx.append(index)
            if set_counter[select_class] >= k_shot:
                return True
            else:
                if self.select(set_counter, set_idx, n_way, k_shot, target_classes, new_cand[i + 1:], select_class):
                    return True
                else:
                    set_counter -= self.samples[index].span_count
                    set_idx.pop()
        return False

    def gen_episode(self):
        support_counter = Counter()
        support_idx = []
        query_counter = Counter()
        query_idx = []
        target_classes = random.sample(self.types, self.N)
        per_class_candidates = self.gen_candidate_idx(target_classes)
        while not all([n >= (self.K + self.Q) for _, c, n in per_class_candidates]):
            target_classes = random.sample(self.types, self.N)
            per_class_candidates = self.gen_candidate_idx(target_classes)

        for in_class_cand, target_class, cand_num in per_class_candidates:
            if not self.select(support_counter, support_idx, self.N, self.K, target_classes, in_class_cand, target_class):
                print(target_classes, support_counter, support_idx)
                raise Exception('Impossible')

        check_counter = Counter()
        for index in support_idx:
            check_counter += self.samples[index].span_count
        assert len(check_counter) == self.N, check_counter
        for c in check_counter:
            assert c in target_classes
            assert self.K <= check_counter[c] <= self.relax_k * self.K

        query_idx.extend(support_idx)
        for in_class_cand, target_class, cand_num in per_class_candidates:
            if not self.select(query_counter, query_idx, self.N, self.Q, target_classes, in_class_cand, target_class):
                print(target_classes, query_counter, query_idx[len(support_idx):])
                raise Exception('Impossible')

        query_idx = query_idx[len(support_idx):]
        assert len(set(support_idx) & set(query_idx)) == 0
        check_counter = Counter()
        for index in query_idx:
            check_counter += self.samples[index].span_count
        assert len(check_counter) == self.N
        for c in check_counter:
            assert c in target_classes
            assert self.Q <= check_counter[c] <= self.relax_k * self.Q

        support_samples = [self.samples[idx] for idx in support_idx]
        query_samples = [self.samples[idx] for idx in query_idx]
        return target_classes, support_samples, query_samples
