import json
import datetime
import logging
import os
import sys
from argparse import ArgumentParser
from collections import defaultdict

import pytorch_lightning as pl
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AdamW, get_linear_schedule_with_warmup

from model import ClassificationBaseModel as BaseModel
from model import SpanEncoder, DotClassifier, L2SquareClassifier, TokenEncoder
from utils import sent_loader, LabelEncoder, SentenceDataset, SentenceSpanDataset, concat_dataset_until
from utils import sentence_collate_fn as collate_fn


class BERTSpan(BaseModel):
    def __init__(self, backbone_name, types, classifier, span_rep_dim, max_span_len, dropout, dropout_pos, span_output,
                 lr, bert_lr, wd, train_step, seed):
        super(BERTSpan, self).__init__()
        self.save_hyperparameters(ignore=['word_embed'])

        pl.seed_everything(seed)
        self.encoder = SpanEncoder(backbone_name, span_rep_dim, max_span_len, dropout, dropout_pos, span_output)
        self.label_encoder = LabelEncoder(types)
        num_class = len(types) + 1
        if classifier == 'Dot':
            self.span_classifier = DotClassifier(num_class, span_rep_dim)
        elif classifier == 'L2':
            self.span_classifier = L2SquareClassifier(num_class, span_rep_dim)
        else:
            raise Exception('Unknown Classifier')
        self.loss_fn = nn.NLLLoss(ignore_index=-1)

    def forward(self, batch):
        token_embedding = self.encoder(batch['token_id'], batch['atten_mask'])
        word_mask, full_span = batch['word_mask'], batch['full_span']

        sample_span_mask, full_span_rep, full_span_label = [], [], []
        full_span_i, full_span_j, full_span_k = [], [], []
        full_span_i_mat = torch.arange(0, full_span.size(0)).view(-1, 1, 1).expand_as(full_span[:, :, :, 0])
        for seq_span_emb, seq_word_mask, seq_full_span, seq_full_span_i_mat in zip(token_embedding, word_mask,
                                                                                   full_span, full_span_i_mat):
            seq_full_mask = seq_full_span[:, :, 0] >= 0
            gather_start, gather_end = torch.nonzero(seq_full_mask, as_tuple=True)
            seq_full_span_rep = self.encoder.span_extractor(seq_span_emb, seq_word_mask, gather_start, gather_end)
            seq_full_span = seq_full_span[seq_full_mask]
            seq_sample_span_mask = seq_full_span[:, -1].bool()
            seq_full_span_label = seq_full_span[:, 0]
            seq_full_span_i = seq_full_span_i_mat[seq_full_mask]
            seq_full_span_j = gather_start
            seq_full_span_k = gather_end

            sample_span_mask.append(seq_sample_span_mask)
            full_span_rep.append(seq_full_span_rep)
            full_span_label.append(seq_full_span_label)
            full_span_i.append(seq_full_span_i)
            full_span_j.append(seq_full_span_j)
            full_span_k.append(seq_full_span_k)

        sample_span_mask = torch.cat(sample_span_mask, 0)
        full_span_rep = torch.cat(full_span_rep, 0)
        full_span_label = torch.cat(full_span_label, 0)
        full_span_i = torch.cat(full_span_i, 0)
        full_span_j = torch.cat(full_span_j, 0)
        full_span_k = torch.cat(full_span_k, 0)
        sample_span_rep = full_span_rep[sample_span_mask]
        sample_span_label = full_span_label[sample_span_mask]

        return {'sample_span_label': sample_span_label, 'full_span_label': full_span_label,
                'sample_span_rep': sample_span_rep, 'full_span_rep': full_span_rep,
                'full_span_i': full_span_i, 'full_span_j': full_span_j, 'full_span_k': full_span_k}

    def training_step(self, batch, batch_idx):
        output = self(batch)
        logit = self.span_classifier(output['sample_span_rep'])
        loss = self.loss_fn(logit, output['sample_span_label'])
        self.log('train/loss', loss, batch_size=batch['batch_size'])
        return loss

    def eval_step(self, batch):
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
            for elem in sorted(cand, key=lambda x: -x[-1]):
                flag = False
                current = (elem[0], elem[1])
                for prior in filter_list:
                    flag = conflict_judge(current, (prior[0], prior[1]))
                    if flag:
                        break
                if not flag:
                    filter_list.append(elem)

            return filter_list

        output = self(batch)
        full_span_logit = self.span_classifier(output['full_span_rep'])
        pred_logit, pred_label = torch.max(full_span_logit, dim=-1)

        word_mask = batch['word_mask'].bool()
        word_label = batch['label'].masked_fill(~word_mask, -1)

        decoded_pred = torch.ones_like(word_label, device=torch.device('cpu')) * -1
        sent_spans = defaultdict(list)
        decoded_cand_buf = {}
        for logit, label, i, j, k in zip(pred_logit.tolist(), pred_label.tolist(),
                                         output['full_span_i'].tolist(), output['full_span_j'].tolist(),
                                         output['full_span_k'].tolist()):
            if label != 0:
                sent_spans[i].append((j, k, label, logit))

        for idx, subword_len in enumerate(batch['subword_len'].tolist()):
            decoded_pred[idx, 1:subword_len + 1] = 0
        for i, span_cand in sent_spans.items():
            decoded_cand = decode(span_cand)
            decoded_cand_buf[i] = decoded_cand
            for j, k, pred, logit in decoded_cand:
                decoded_pred[i, j:k + 1] = pred
        batch_preds, batch_targets = self.assemble_sentence(
            {'pred': decoded_pred, 'target': word_label, 'sent_id': batch['sent_id']})
        batch_jsons = [json.loads(json_str) for json_str in batch['jsons']]
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
        scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, self.hparams.train_step)

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


class BertToken(BaseModel):
    def __init__(self, backbone_name, types, classifier, token_rep_dim, dropout, dropout_pos, token_output,
                 lr, bert_lr, wd, train_step, seed):
        super(BertToken, self).__init__()
        self.save_hyperparameters()

        pl.seed_everything(seed)
        self.encoder = TokenEncoder(backbone_name, token_rep_dim, dropout, dropout_pos, token_output)
        self.label_encoder = LabelEncoder(types)
        num_class = len(types) + 1
        if classifier == 'Dot':
            self.token_classifier = DotClassifier(num_class, hidden)
        elif classifier == 'L2':
            self.token_classifier = L2SquareClassifier(num_class, hidden)
        else:
            raise Exception('Unknown Classifier')
        self.loss_fn = nn.NLLLoss(ignore_index=-1)

    def forward(self, batch):
        token_rep = self.encoder(batch['id'], batch['atten_mask'])
        word_mask = batch['word_mask'].bool()
        label = batch['label']

        word_label = label.masked_fill(~word_mask, -1)
        token_logit = self.token_classifier(token_rep)

        loss_logits = token_logit.view(-1, token_logit.size(-1))
        loss_targets = word_label.view(-1)
        loss = self.loss_fn(loss_logits, loss_targets)
        _, pred = token_logit.max(dim=-1)

        return {
            'loss': loss, 'pred': pred, 'target': word_label,
            'sent_id': batch['sent_id']
        }

    def training_step(self, batch, batch_idx):
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
                    'frequency': 1
                }
            }
        ]


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--mode', default='inter',
                        help='training mode, must be in [inter, intra]')
    parser.add_argument('--batch_size', default=32, type=int,
                        help='batch size')
    parser.add_argument('--grad_acc', default=1, type=int,
                        help='grad acc step')
    parser.add_argument('--train_step', default=12000, type=int,
                        help='num of step in training')
    parser.add_argument('--val_interval', default=1000, type=int,
                        help='num of interval between validation')
    parser.add_argument('--model', default='span',
                        help='model name, must be in [token, span]')
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
    parser.add_argument('--classifier', default='Dot',
                        help='either Dot or L2')
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

    logging.info(f'model:{args.model}-{args.backbone_name} hidden:{args.hidden} dropout:{args.dropout}'
                 f' classifier:{args.classifier} dropout position:{args.dropout_pos} span output:{args.span_output}')
    logging.info(f'fewshot:{args.mode}-{args.max_length}')
    logging.info(f'train:{args.train_step} val:{args.val_interval} bs:{args.batch_size} grad acc:{args.grad_acc}')
    if args.model == 'span':
        logging.info(f'span max len:{args.max_span_len} neg:{args.neg_rate}')
    logging.info(f'lr:{args.lr} bert lr:{args.bert_lr} wd:{args.wd}')
    logging.info(f'seed:{args.seed}')

    worker_num = min(os.cpu_count(), 4)
    pl.seed_everything(args.seed)
    train_val_dataset = SentenceDataset(sent_loader(args.mode, 'train'), args.backbone_name, args.max_length)
    types = train_val_dataset.types
    logging.info(f'train dataset:{args.mode} max_length:{args.max_length}')
    train_length = int(0.9 * len(train_val_dataset))
    val_length = len(train_val_dataset) - train_length
    train_dataset, dev_dataset = torch.utils.data.random_split(train_val_dataset, (train_length, val_length))
    train_dataset = concat_dataset_until(train_dataset, 10000000)
    if args.model == 'span':
        train_dataset = SentenceSpanDataset(train_dataset, neg_rate=args.neg_rate, max_len=args.max_span_len,
                                            length_limited_full_span=True)
        dev_dataset = SentenceSpanDataset(dev_dataset, neg_rate=args.neg_rate, max_len=args.max_span_len,
                                          length_limited_full_span=True)
    effective_batch_size = args.batch_size * args.grad_acc
    assert len(train_dataset) >= args.train_step * effective_batch_size
    train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=worker_num,
                                   collate_fn=collate_fn, shuffle=True)
    logging.info(f'training loader:{len(train_data_loader)} batch:{args.batch_size} worker:{worker_num}')

    dev_data_loader = DataLoader(dev_dataset, batch_size=args.batch_size, num_workers=worker_num,
                                 collate_fn=collate_fn)
    logging.info(f'dev loader:{len(dev_data_loader)} batch:{args.batch_size} worker:{worker_num}')

    if args.model == 'span':
        model = BERTSpan(args.backbone_name, types, args.classifier, args.hidden, args.max_span_len, args.dropout,
                         args.dropout_pos, args.span_output, args.lr, args.bert_lr, args.wd, args.train_step, args.seed)
    elif args.model == 'token':
        model = BERTToken(args.backbone_name, types, args.classifier, args.hidden, args.dropout, args.dropout_pos,
                          args.span_output, args.lr, args.bert_lr, args.wd, args.train_step, args.seed)
    else:
        raise Exception('Unknown model type')

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
            val_check_interval=args.val_interval * args.grad_acc,
            accumulate_grad_batches=args.grad_acc,
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
            num_sanity_val_steps=20,
            val_check_interval=args.val_interval * args.grad_acc,
            accumulate_grad_batches=args.grad_acc,
            callbacks=[]
        )

    trainer.fit(model, train_dataloaders=train_data_loader, val_dataloaders=dev_data_loader)
    if not args.debug_only:
        if args.model == 'span':
            model = BERTSpan.load_from_checkpoint(checkpoint_callback.best_model_path)
        elif args.model == 'token':
            model = BertToken.load_from_checkpoint(checkpoint_callback.best_model_path)
        else:
            raise Exception('Unknown model type')
        torch.save(model.encoder.state_dict(), os.path.join(checkpoint_dir, f'encoder.state'))
