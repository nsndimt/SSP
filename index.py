import json
import sys
import os
import time
os.chdir("/data/zhangyue/fewshotNER")
sys.path.append("/data/zhangyue/fewshotNER")
import logging
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from itertools import islice, chain, product
from pytorch_lightning.utilities.apply_func import move_data_to_device
from torch.utils.data import DataLoader
from tqdm import tqdm
from train import ProtoSpan
from reproduce import Proto
from utils import episode_collate_fn, sentence_collate_fn
from utils import EpisodeDataset, SentenceDataset, EpisodeSpanDataset, SentenceSpanDataset, get_io_spans, LabelEncoder

class entry:
    def __init__(self, words, labels):
        self.words = words
        self.labels = labels
        self.spans, self.span_count = get_io_spans(labels)
        self.subsent_len = []
        self.data_buf = []
        self.label_buf = []
        self.start_buf = []
        self.end_buf = []
        self.freeze = False
        
    def chunk_add(self, word_mask, rep, start, end):
        word_size = sum(word_mask)
        assert word_size <= len(self.words), (word_mask, self.words)
        offset = sum(self.subsent_len)
        self.subsent_len.append(word_size)
        
        word_idx = {}
        for i, wm in enumerate(word_mask):
            if wm > 0:
                word_idx[i] = len(word_idx)
#         print(word_idx)
        for v, i, j in zip(rep, start, end):
            word_i = offset + word_idx[i]
            word_j = offset + word_idx[j]
            assert 0<=word_i<=len(self.words), (word_i, offset, word_size, len(self.words))
            assert 0<=word_j<=len(self.words), (word_j, offset, word_size, len(self.words))
            assert word_i<=word_j<=word_i+10, (i, j, word_i, word_j)
            l = 'O'
            for x, y, z in self.spans:
                if x == word_i and y == word_j:
                    l = z
                    break
            self.add(word_i, word_j, v, l)
        
    def add(self, start, end, rep, label):
        assert not self.freeze
        self.data_buf.append(rep)
        self.label_buf.append(label)
        self.start_buf.append(start)
        self.end_buf.append(end)
        
    def concat(self):
        self.data_buf = torch.stack(self.data_buf)
        self.freeze = True


 
span_model_path = {
    (5, 1): "infochain_checkpoint/20220502-201906/epoch=0-step=3500.ckpt",
    (5, 5): "infochain_checkpoint/20220502-185916/epoch=0-step=6000.ckpt",
    (10, 1): "infochain_checkpoint/20220502-232207/epoch=0-step=6000.ckpt",
    (10, 5): "infochain_checkpoint/20220502-213946/epoch=0-step=6000.ckpt"
}

token_model_path = {
    (5, 1): "infochain_checkpoint/20220410-193213-874644/epoch=0-step=2999.ckpt",
    (5, 5): "infochain_checkpoint/20220410-182325-874640/epoch=0-step=2999.ckpt",
    (10, 1): "infochain_checkpoint/20220410-193532-874646/epoch=0-step=2999.ckpt",
    (10, 5): "infochain_checkpoint/20220410-191610-874642/epoch=0-step=4999.ckpt"
}

def sent_loader(task, split):
    buf = []
    with open(f'profile/{task}/{split}.txt') as f:
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
    with open(f'profile/episode-data/{task}/{split}_{n}_{k}.jsonl') as f:
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
        
def profile_model(N, K, model, backbone):
    pl.seed_everything(1)
    test_dataset = EpisodeDataset(truecase_episode_loader('inter', 'test', N, K), backbone, 96)
    test_dataset = EpisodeSpanDataset(test_dataset, -1, max_len=10, length_limited_full_span=True)
    test_data_loader = DataLoader(test_dataset, batch_size=1, num_workers=8, prefetch_factor=16,
                                  collate_fn=episode_collate_fn)
    trainer = pl.Trainer(
            logger=False,
            enable_checkpointing=False,
            gpus=1,
            callbacks=[]
        )
    start_time = time.time()
    trainer.test(model, test_data_loader)
    end_time = time.time()
    total_time = end_time - start_time
    return {'encode':model.encode_time, 'comp':model.comp_time, 'total': total_time}

def precompute(model):
    pl.seed_everything(1)
    test_dataset = SentenceDataset(sent_loader('inter', 'test'), 'bert-base-cased', 96)
    test_dataset = SentenceSpanDataset(test_dataset, neg_rate=1000, max_len=10, length_limited_full_span=True)
    test_data_loader = DataLoader(test_dataset, batch_size=32, num_workers=8, prefetch_factor=16,
                                  collate_fn=sentence_collate_fn)

    model.eval()
    device = torch.device('cuda')
    model = model.to(device)

    data = {}
    for batch in tqdm(test_data_loader):
        batch = move_data_to_device(batch, device)
        batch_entry = {}
        for i,sent_json in enumerate(batch['jsons']):
            words, labels, additional_info = json.loads(sent_json)
            assert additional_info == dict()
            sent_rep = ' '.join([f'{w}[{l}]'if l!='O' else w for w,l in zip(words,labels)])
            if sent_rep in data:
                try:
                    assert labels == data[sent_rep].labels
                    assert words == data[sent_rep].words
                    # print('dup sent rep')
                except AssertionError as e:
                    print((sent_rep, ' '.join(labels), ' '.join(data[sent_rep].labels)))
            else:
                data[sent_rep] = entry(words, labels)
                batch_entry[i]=data[sent_rep]
        
        with torch.no_grad():
            token_embedding = model.encoder(batch['token_id'], batch['atten_mask'])
            word_mask, full_span = batch['word_mask'], batch['full_span']

            full_span_rep, full_span_label = [], []
            for seq_span_emb, seq_word_mask, seq_full_span, seq_sent_id in zip(token_embedding, word_mask, full_span, batch['sent_id']):
                seq_full_mask = seq_full_span[:, :, 0] >= 0
                gather_start, gather_end = torch.nonzero(seq_full_mask, as_tuple=True)
                seq_full_span_rep = model.encoder.span_extractor(seq_span_emb, seq_word_mask, gather_start, gather_end)
                seq_full_span = seq_full_span[seq_full_mask]
                seq_full_span_label = seq_full_span[:, 0]

                if seq_sent_id in batch_entry:
                    seq_entry = batch_entry[seq_sent_id]
                    seq_entry.chunk_add(seq_word_mask.tolist(), seq_full_span_rep, gather_start.tolist(), gather_end.tolist())

        for i, sent_entry in batch_entry.items():
            sent_entry.concat()
    
    return data

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

def index_eval_step(episode, index):
    def index_lookup(words, labels):
        sent_rep = ' '.join([f'{w}[{l}]'if l!='O' else w for w,l in zip(words,labels)])
        return index[sent_rep]
    io_time, comp_time, eval_time = 0., 0., 0.
    
    io_start = time.time()
    device = torch.device('cuda')
    types, support, query = episode
    label_encoder = LabelEncoder(types)
    support_entry = [index_lookup(words,labels) for words, labels, additional_info in support]
    query_entry = [index_lookup(words,labels) for words, labels, additional_info in query]

    full_support_span_rep = torch.cat([entry.data_buf for entry in support_entry], dim=0)
    full_support_span_label = list(chain(*[entry.label_buf for entry in support_entry]))
    full_support_span_label = torch.Tensor(label_encoder.index(full_support_span_label)).to(device=device)

    full_query_span_rep = torch.cat([entry.data_buf for entry in query_entry], dim=0)
    rep_end_idx = np.cumsum([entry.data_buf.size(0) for entry in query_entry]).tolist()
    rep_start_idx = [0]+rep_end_idx[:-1]
    io_end = time.time()
    io_time += io_end - io_start

    comp_start = time.time()
    query_sim = []
    tag_list = [(0, 'O')] + [(lid + 1, l) for lid, l in enumerate(types)]
    for lid, l in tag_list:
        in_class_label_mask = (full_support_span_label == lid)
        in_class_emb = full_support_span_rep[in_class_label_mask]
        atten_weight = (full_query_span_rep.matmul(in_class_emb.T)).softmax(dim=-1)
        atten_proto = atten_weight.matmul(in_class_emb)
        in_class_sim = -(full_query_span_rep - atten_proto).pow(2).sum(1)

        query_sim.append(in_class_sim)

    query_sim = torch.stack(query_sim, dim=-1)
    query_logit = F.log_softmax(query_sim, dim=-1)
    pred_logit, pred_label = torch.max(query_logit, dim=-1)
    comp_end = time.time()
    comp_time += comp_end - comp_start


    io_start = time.time()
    pred_logit = pred_logit.tolist()
    pred_label = [tag_list[lid][1] for lid in pred_label.tolist()]
    io_end = time.time()
    io_time += io_end - io_start
    
    eval_start = time.time()
    pred_cnt = 0  # pred entity cnt
    label_cnt = 0  # true label entity cnt
    correct_cnt = 0  # correct predicted entity cnt

    within_cnt = 0  # span correct but of wrong fine-grained type
    outer_cnt = 0  # span correct but of wrong coarse-grained type
    total_span_cnt = 0  # span correct

    for ei,entry in enumerate(query_entry):
        target_spans = entry.spans
        entry_pred_label = pred_label[rep_start_idx[ei]:rep_end_idx[ei]]
        assert len(entry_pred_label) == entry.data_buf.size(0)
        entry_pred_logit = pred_logit[rep_start_idx[ei]:rep_end_idx[ei]]

        span_cand = [(i,j,l,s) for i,j,l,s in zip(entry.start_buf, entry.end_buf, entry_pred_label, entry_pred_logit) if l != 'O']
        pred_spans = [(i,j,l) for i,j,l,s in decode(span_cand)]

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
    eval_end = time.time()
    eval_time += eval_end - eval_start
    return (np.array([pred_cnt, label_cnt, correct_cnt, within_cnt, outer_cnt, total_span_cnt]), 
            np.array([io_time, comp_time, eval_time]))

def inference(N, K, index):
    res_cnt = np.zeros(6, dtype=np.int)
    time_acc = np.zeros(3, dtype=np.float)
    for episode in tqdm(truecase_episode_loader('inter','test', N, K)):
        episode_cnt, episode_time = index_eval_step(episode, index)
        res_cnt += episode_cnt
        time_acc += episode_time
    
    pred_cnt, label_cnt, correct_cnt, within_cnt, outer_cnt, total_span_cnt = res_cnt.tolist()
    io_time, comp_time, eval_time = time_acc.tolist()
    
    precision = correct_cnt / (pred_cnt + 1e-6)
    recall = correct_cnt / (label_cnt + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)

    within_error = within_cnt / (total_span_cnt + 1e-6)
    outer_error = outer_cnt / (total_span_cnt + 1e-6)
    # print(precision, recall ,f1)
    # print(within_error, outer_error)
    # print("io", io_time, "comp", comp_time, "eval", eval_time)
    return {"io":io_time, "comp":comp_time}


if __name__ == "__main__":
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s;%(levelname)s;%(message)s", "%Y-%m-%d %H:%M:%S")
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    root.addHandler(handler)

    transformers.utils.logging.set_verbosity_error()

    for N in [10]:
        for K in [1]:
            model = Proto.load_from_checkpoint(token_model_path[(N, K)])
            profile_res = profile_model(N, K, model, 'bert-base-uncased')
            print('token: encode, comp, total')
            print(profile_res['encode'], profile_res['comp'], profile_res['total'])
            model = ProtoSpan.load_from_checkpoint(span_model_path[(N, K)], pretrained_encoder='')
            profile_res = profile_model(N, K, model, 'bert-base-cased')
            index = precompute(model)
            index_res = inference(N, K, index)
            print('span: encode, comp, total, io, comp')
            print(profile_res['encode'], profile_res['comp'], profile_res['total'],
                  index_res['io'], index_res['comp'])
