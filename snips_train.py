import datetime
import logging
import os
import pytorch_lightning as pl
import sys
import torch
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from torch.utils.data.dataset import ConcatDataset
from utils import episode_collate_fn as collate_fn
from train import ProtoSpan
from reproduce import Proto, NNShot, StructShot
from utils import SNIPS_CV_loader, SNIPS_sent_loader, PerClassEpisodeSamplingDataset, EpisodeDataset, EpisodeSpanDataset
from itertools import chain

if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--cv_id', default=1, type=int,
                        help='1 to 7')
    parser.add_argument('--K', default=5, type=int,
                        help='K shot')
    parser.add_argument('--batch_size', default=1, type=int,
                        help='batch size')
    parser.add_argument('--grad_acc', default=1, type=int,
                        help='grad acc step')
    parser.add_argument('--train_epoch', default=40, type=int,
                        help='num of step in training')
    parser.add_argument('--model', default='span',
                        help='model name, must be in [span]')
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
    parser.add_argument('--metric', default='L2',
                        help='similarity metric, must be in [L2, dot, cosine]')
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
    parser.add_argument('--tau', default=0.32, type=float,
                        help='structshot tau')
    parser.add_argument('--subword_crf', action='store_false',
                        help='use subword level crf to decode')
    parser.add_argument('--sample_train', action='store_true',
                        help='use our own sampling for training')
    parser.add_argument('--output_dir_overwrite', action='store_true',
                        help='output to specified dir')
    parser.add_argument('--output_dir', default='',
                        help='output dir')
    parser.add_argument('--wandb_proj', default='SNIPS',
                        help='wandb project')
    args = parser.parse_args()

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s;%(levelname)s;%(message)s",
                                  "%Y-%m-%d %H:%M:%S")
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    root.addHandler(handler)

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

        handler = logging.FileHandler(os.path.join(checkpoint_dir, f'shell_output.log'))
        handler.setFormatter(formatter)
        root.addHandler(handler)

    logging.info('## Cmd line ##')
    logging.info(f"python {' '.join(sys.argv)}")

    logging.info(
        f'model:{args.model}-{args.backbone_name}-{args.metric}'
        f' pretrained encoder:{args.pretrained_encoder} reinit:{args.reinit}'
        f' hidden:{args.hidden} dropout:{args.dropout}  dropout position:{args.dropout_pos}'
        f' span output:{args.span_output} use atten:{args.use_atten}')
    logging.info(f'snips:{args.cv_id}-{args.K}-{args.max_length}')
    logging.info(f'train epoch:{args.train_epoch} bs:{args.batch_size} grad acc:{args.grad_acc}')
    if args.model == 'span':
        logging.info(f'span max len:{args.max_span_len} neg:{args.neg_rate}')
    logging.info(f'lr:{args.lr} bert lr:{args.bert_lr} wd:{args.wd}')
    logging.info(f'seed:{args.seed}')

    worker_num = min(os.cpu_count(), 4)
    pl.seed_everything(args.seed)
    types = set()
    train_names, train_dataset = [], []
    if not args.sample_train:
        for name, episodes in SNIPS_CV_loader('train', args.cv_id, args.K).items():
            train_names.append(name)
            dataset = EpisodeDataset(episodes, args.backbone_name, args.max_length)
            types.update(dataset.types)
            train_dataset.append(dataset)
    else:
        for name, episodes in SNIPS_CV_loader('train', args.cv_id, args.K).items():
            train_names.append(name)
            dataset = EpisodeDataset(episodes, args.backbone_name, args.max_length)
            dataset = PerClassEpisodeSamplingDataset(len(dataset.types), args.K, args.K, SNIPS_sent_loader(name),
                                                     args.backbone_name, args.max_length)
            assert len(dataset) >= args.train_epoch*500
            types.update(dataset.types)
            train_dataset.append(dataset)
    train_dataset = ConcatDataset(train_dataset)
    types = list(types)
    if args.model == 'span':
        train_dataset = EpisodeSpanDataset(train_dataset, args.neg_rate, max_len=args.max_span_len,
                                           length_limited_full_span=True)
    train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=worker_num,
                                   collate_fn=collate_fn, shuffle=True)
    logging.info(f"train dataset:{','.join(train_names)}-{args.K} sampled:{args.sample_train} "
                 f"max_length:{args.max_length}")

    logging.info(f'training loader:{len(train_data_loader)} batch:{args.batch_size} worker:{worker_num}')

    dev_loader_items = list(SNIPS_CV_loader('valid', args.cv_id, args.K).items())
    assert len(dev_loader_items) == 1
    dev_name, dev_episodes = dev_loader_items[0]
    dev_dataset = EpisodeDataset(dev_episodes, args.backbone_name, args.max_length)
    if args.model == 'span':
        dev_dataset = EpisodeSpanDataset(dev_dataset, -1, max_len=args.max_span_len, length_limited_full_span=True)
    logging.info(f'dev dataset:{dev_name}-{args.K} max_length:{args.max_length}')
    dev_data_loader = DataLoader(dev_dataset, batch_size=args.batch_size, num_workers=worker_num,
                                 collate_fn=collate_fn)
    logging.info(f'dev loader:{len(dev_data_loader)} batch:{args.batch_size} worker:{worker_num}')

    test_loader_items = list(SNIPS_CV_loader('test', args.cv_id, args.K).items())
    assert len(test_loader_items) == 1
    test_name, test_episodes = test_loader_items[0]
    test_dataset = EpisodeDataset(test_episodes, args.backbone_name, args.max_length)
    if args.model == 'span':
        test_dataset = EpisodeSpanDataset(test_dataset, -1, max_len=args.max_span_len, length_limited_full_span=True)
    logging.info(f'test dataset:{test_name}-{args.K} max_length:{args.max_length}')
    test_data_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=worker_num,
                                  collate_fn=collate_fn)
    logging.info(f'test loader:{len(test_data_loader)} batch:{args.batch_size} worker:{worker_num}')
    if not args.sample_train:
        train_steps = len(train_data_loader) * args.train_epoch
    else:
        train_steps = 500 * args.train_epoch
    if args.model == 'span':
        model = ProtoSpan(args.backbone_name, args.pretrained_encoder, args.reinit, types, args.hidden, args.dropout,
                          args.dropout_pos, args.span_output, args.max_span_len, args.use_atten,
                          args.lr, args.bert_lr, args.wd, train_steps, args.seed)
        model.set_snips_mode()
        if not args.sample_train:
            model.unset_dual_loss_mode()
        else:
            model.set_dual_loss_mode()
    elif args.model == 'proto':
        model = Proto(args.backbone_name, args.metric,
                      args.dropout, args.locked_dropout,
                      args.bert_layer, args.bert_layer_agg,
                      args.cosface, args.cosface_s, args.cosface_m,
                      args.subword_proto, args.subword_eval,
                      args.bert_lr, args.wd,
                      train_steps, args.seed)
        model.set_snips_mode()
    elif args.model == 'nnshot':
        model = NNShot(args.backbone_name, args.metric,
                       args.dropout, args.locked_dropout,
                       args.bert_layer, args.bert_layer_agg,
                       args.cosface, args.cosface_s, args.cosface_m,
                       args.subword_proto, args.subword_eval,
                       args.bert_lr, args.wd,
                       train_steps, args.seed)
        model.set_snips_mode()
    elif args.model == 'structshot':
        model = StructShot(args.backbone_name, args.metric,
                           args.dropout, args.locked_dropout,
                           args.bert_layer, args.bert_layer_agg,
                           args.cosface, args.cosface_s, args.cosface_m,
                           args.subword_proto, args.subword_eval,
                           args.tau, args.subword_crf,
                           args.bert_lr, args.wd,
                           train_steps, args.seed)
        episode_loader= chain(*SNIPS_CV_loader('train', args.cv_id, args.K).values())
        model.set_abstract_transitions(episode_loader)
        model.set_snips_mode()
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

        if not args.sample_train:
            trainer = pl.Trainer(
                logger=logger,
                gpus=1,
                precision=16,
                max_epochs=args.train_epoch,
                num_sanity_val_steps=0,
                accumulate_grad_batches=args.grad_acc,
                callbacks=[checkpoint_callback, lr_monitor]
            )
        else:
            trainer = pl.Trainer(
                logger=logger,
                gpus=1,
                precision=16,
                max_steps=args.train_epoch*500,
                max_epochs=100,
                num_sanity_val_steps=0,
                val_check_interval=500 * args.grad_acc,
                accumulate_grad_batches=args.grad_acc,
                callbacks=[checkpoint_callback, lr_monitor]
            )
    else:
        if not args.sample_train:
            trainer = pl.Trainer(
                logger=False,
                enable_checkpointing=False,
                gpus=1,
                precision=16,
                max_epochs=args.train_epoch,
                num_sanity_val_steps=20,
                accumulate_grad_batches=args.grad_acc,
                callbacks=[]
            )
        else:
            trainer = pl.Trainer(
                logger=False,
                enable_checkpointing=False,
                gpus=1,
                precision=16,
                max_steps=args.train_epoch*500,
                max_epochs=100,
                num_sanity_val_steps=20,
                val_check_interval=500 * args.grad_acc,
                accumulate_grad_batches=args.grad_acc,
                callbacks=[]
            )

    trainer.fit(model, train_dataloaders=train_data_loader, val_dataloaders=dev_data_loader)
    if not args.debug_only:
        trainer.test(model, ckpt_path='best', dataloaders=test_data_loader)
    else:
        trainer.test(model, dataloaders=test_data_loader)

