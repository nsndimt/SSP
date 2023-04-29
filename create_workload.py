import json
import sys
import os
from random import choice
from argparse import ArgumentParser
from itertools import islice
from tqdm import tqdm
os.chdir("/data/zhangyue/fewshotNER")
sys.path.append("/data/zhangyue/fewshotNER")
from utils import PerClassEpisodeSamplingDataset, sent_loader
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--sn', type=int, default=100)
    parser.add_argument('--qn', type=int, default=100)
    parser.add_argument('--en', type=int, default=100)
    args = parser.parse_args()
    dir_path = os.path.dirname(__file__)
    target_query_num = args.qn
    target_support_num = args.sn
    target_episode_num = args.en
    with open(f'Few-NERD/data/episode-data/inter/test_10_1.jsonl') as fin,\
    open(f'{dir_path}/episode-data/inter/test_10_1.jsonl', 'w') as fout:
        for line in islice(fin, target_episode_num):
            line = json.loads(line.strip())
            query = line['query']
            query = [(words, labels) for words, labels in zip(query['word'], query['label'])]
            support = line['support']
            support = [(words, labels) for words, labels in zip(support['word'], support['label'])]
            types = line['types']
            newline = {
                'query': {'word': line['query']['word'], 'label':line['query']['label']},
                'support': {'word': line['support']['word'], 'label':line['support']['label']},
                'types': line['types']
            }
            assert len(support) <= target_support_num
            if target_support_num -len(support) > 0:
                for i in range(target_support_num - len(support)):
                    words, labels = choice(support)
                    newline['support']['word'].append(words)
                    newline['support']['label'].append(labels)
            assert len(query) <= target_query_num
            if target_query_num - len(query) > 0:
                for i in range(target_query_num - len(query)):
                    words, labels = choice(query)
                    newline['query']['word'].append(words)
                    newline['query']['label'].append(labels)
            fout.write(json.dumps(newline)+'\n')
    # loader = sent_loader('inter', 'test')
    # dataset = PerClassEpisodeSamplingDataset(N=10,
    #                                          K=target_support_num,
    #                                          Q=target_query_num,
    #                                          loader=loader,
    #                                          backbone_name='bert-base-cased',
    #                                          max_length=96
    #                                          )
    # with open(f'{dir_path}/episode-data/inter/test_10_1.jsonl', 'w') as fout:
    #     for line in tqdm(range(target_episode_num)):
    #         types, support, query = dataset.gen_episode()
    #         newline = {
    #             'query': {'word': [s.words for s in query], 'label':[s.labels for s in query]},
    #             'support': {'word': [s.words for s in support], 'label':[s.labels for s in support]},
    #             'types': types
    #         }
    #         fout.write(json.dumps(newline)+'\n')