from typing import Dict, Union, Callable, Optional
import argparse
import os
import json
from multiprocessing import cpu_count
import warnings
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

import torchvision
from torchvision import transforms

from imgaug import augmenters as iaa

from simple_trainer.models import UpdateFunction
from simple_trainer.trainer import Trainer
from simple_trainer import functions

from captioning import models
from captioning.utils import misc as utils
from captioning.modules import losses
from captioning.utils.rewards import init_scorer, get_self_critical_reward


import dataloader
from dataloader import ResnetAttention
from vocab import Vocabulary


def have_cuda():
    return torch.cuda.is_available()


def get_model_params(vocab, args):
    gpus = args.gpus.split(",")
    if not gpus:
        device = torch.device("cpu")
    elif len(gpus) == 1:
        gpu = gpus[0]
        device = torch.device(f"cuda:{gpu}")
    else:
        raise NotImplementedError("Multiple gpus not implemented right now.")
    for x in ["embed", "hidden", "fc"]:
        if not getattr(args, f"{x}_dropout_ratio"):
            setattr(args, f"{x}_dropout_ratio", None)
    return {"together": {"enc_fc_dim": args.vis_dim,
                         "att_dim": args.vis_num,
                         "embed_dim": args.embed_dim,
                         "hidden_dim": args.hidden_dim,
                         "vocab": vocab,
                         "vocab_size": len(vocab),
                         "num_layers": args.num_layers,
                         "dropout_ratio": args.embed_dropout_ratio,
                         "device": device},
            "topk": {"vis_dim": args.vis_dim,
                     "vis_num": args.vis_num,
                     "embed_dim": args.embed_dim,
                     "hidden_dim": args.hidden_dim,
                     "vocab": vocab,
                     "vocab_size": len(vocab),
                     "num_layers": args.num_layers,
                     "dropout_ratio": args.embed_dropout_ratio,
                     "device": device},
            "new": {"vis_dim": args.vis_dim,
                    "vis_num": args.vis_num,
                    "embed_dim": args.embed_dim,
                    "hidden_dim": args.hidden_dim,
                    "vocab": vocab,
                    "vocab_size": len(vocab),
                    "num_layers": args.num_layers,
                    "dropout_ratio": args.embed_dropout_ratio,
                    "device": device},
            "EncoderDecoder": {"encoder": args.encoder,
                               "vis_dim": args.vis_dim,
                               "vis_num": args.vis_num,
                               "embed_dim": args.embed_dim,
                               "hidden_dim": args.hidden_dim,
                               "vocab": vocab,
                               "num_layers": args.num_layers,
                               "dropout_ratio": args.embed_dropout_ratio,
                               "device": device},
            "DecoderOld": {"vis_dim": args.vis_dim,
                           "vis_num": args.vis_num,
                           "embed_dim": args.embed_dim,
                           "hidden_dim": args.hidden_dim,
                           "vocab": vocab,
                           "num_layers": args.num_layers,
                           "dropout_ratio": args.embed_dropout_ratio,
                           "device": device},
            "Decoder": {"feat_dim": args.vis_dim,
                        "embed_dim": args.embed_dim,
                        "feat_projection_dim": 512,
                        "hidden_dim": args.hidden_dim,
                        "att_dim": args.vis_num,
                        "vocab": vocab,
                        "num_layers": args.num_layers,
                        "embed_dropout_ratio": args.embed_dropout_ratio,
                        "hidden_dropout_ratio": args.hidden_dropout_ratio,
                        "fc_dropout_ratio": args.fc_dropout_ratio,
                        "device": device},
            "Att2inModel": {"feat_dim": args.vis_dim,
                            "embed_dim": args.embed_dim,
                            "feat_projection_dim": 512,
                            "hidden_dim": args.hidden_dim,
                            "att_dim": args.vis_num,
                            "vocab": vocab,
                            "num_layers": args.num_layers,
                            "embed_dropout_ratio": args.embed_dropout_ratio,
                            "hidden_dropout_ratio": args.hidden_dropout_ratio,
                            "fc_dropout_ratio": args.fc_dropout_ratio,
                            "device": device}}


def check_vocab_and_generate_if_required(vocab, data_dir):
    vocab_path = os.path.join(data_dir, "vocab.json")
    if os.path.exists(vocab_path):
        print(f"Loading vocab from {vocab_path}")
        vocab.load_from_json(vocab_path)
    else:
        print(f"No vocab found. Building to {vocab_path}")
        with open(os.path.join(data_dir, "dataset.json")) as f:
            data = json.load(f)
        for d in data["images"]:
            if d['split'] == "train":
                sents = d["sentences"]
                for sent in sents:
                    for word in sent['tokens']:
                        vocab.add_word(word)
        vocab.truncate("frequency", 2)
        vocab.dump_to_json(vocab_path)


def get_caption_data_with_features(dataroot: str, dataset_name: str,
                                   batch_size: Union[int, Dict[str, int]],
                                   workers: Union[int, Dict[str, int]],
                                   generate_features: bool = False,
                                   train_sampler: Optional[Callable] = None,
                                   mode: str = "single",
                                   in_memory: Optional[bool] = False):
    num_cpu = cpu_count()
    if isinstance(batch_size, int):
        print("Only one batch_size given. Rest will be set to 128")
        batch_size = {"train": batch_size, "val": 128, "test": 128}
    if isinstance(workers, int):
        print("Only train workers given. Rest will be set to 0")
        if workers == num_cpu:
            print("Warning: Workers == cpu_count given")
        workers = {"train": max(0, min(workers, num_cpu)), "val": 0, "test": 0}
    else:
        workers = {"train": max(0, min(num_cpu, workers.get("train", 0))),
                   "val": max(0, min(num_cpu, workers.get("val", 0))),
                   "test": max(0, min(num_cpu, workers.get("test", 0)))}
    if dataset_name in ["flickr8k", "flickr30k"]:
        data_keys = {"train": True, "val": True, "test": True}
        train_keys = ["train"]
        vocab = Vocabulary(pad_is_end=True)
        data_dir = os.path.join(dataroot, "flickr30k")
        check_vocab_and_generate_if_required(vocab, data_dir)
        backbone_model = "resnet101"
        if generate_features:
            model = torchvision.models.resnet101()
            weights = torch.load("./data/imagenet_weights/resnet101.pth")
            model.load_state_dict(weights)
            backbone = ResnetAttention(backbone_model, 14, model)
            backbone_gpu = 2
        else:
            backbone = None
            backbone_gpu = None
        feature_path = f"{dataroot}/{dataset_name}/{dataset_name}_{backbone_model}_features.h5"
        _data = dataloader.CaptionData(dataset_name, "resnet101",
                                       backbone, backbone_gpu,
                                       os.path.join(dataroot, dataset_name, "images"),
                                       feature_path=feature_path,
                                       ann_path=os.path.join(dataroot, dataset_name, "dataset.json"),
                                       labels_file=os.path.join(dataroot, dataset_name, "labels.h5"),
                                       vocab=vocab,
                                       in_memory=bool(in_memory))
        data = {k: _data.get_data(k, None, mode) if v else None
                for k, v in data_keys.items()}
    else:
        raise NotImplementedError("Only flickr8k and flickr30k implemented for now")
    _batch_size = {}
    _workers = {}
    for k, v in data.items():
        if v is not None:
            if k in train_keys:
                _batch_size[k] = batch_size["train"]
                _workers[k] = workers["train"]
            elif k in batch_size:
                _batch_size[k] = batch_size[k]
                _workers[k] = workers[k]
            else:
                try:
                    if k in val_keys:
                        _batch_size[k] = batch_size["val"]
                        _workers[k] = workers["val"]
                except Exception:
                    if k in test_keys:
                        _batch_size[k] = batch_size["test"]
                        _workers[k] = workers["test"]
    data["name"] = dataset_name
    batch_size = _batch_size
    workers = _workers
    if train_sampler:
        dataloaders = {x: torch.utils.data.DataLoader(
            data[x], batch_size=batch_size[x],
            sampler=train_sampler(range(len(data[x])), False) if x in train_keys else None,
            shuffle=False,
            collate_fn=data[x]._collate_fn_normal
            if mode == "single" else data[x]._collate_fn_group,
            num_workers=workers[x]) if y else None for x, y in data_keys.items()}
    else:
        dataloaders = {x: torch.utils.data.DataLoader(
            data[x], batch_size=batch_size[x], shuffle=True if x == "train" else False,
            collate_fn=data[x]._collate_fn_normal
            if mode == "single" else data[x]._collate_fn_group,
            num_workers=workers[x]) if y else None for x, y in data_keys.items()}
    return data, dataloaders, vocab


def check_min(values):
    keys = [*values.keys()]
    keys.sort()
    vals = [x["total"] for x in values.values()]
    if len(values) < 2:
        return False
    elif values[keys[-1]] == min(vals):
        return True
    else:
        return False


def check_max(values):
    keys = [*values.keys()]
    keys.sort()
    vals = [x["total"] for x in values.values()]
    if len(values) < 2:
        return False
    elif values[keys[-1]] == max(vals):
        return True
    else:
        return False


class LossWrapper(torch.nn.Module):
    def __init__(self, model, label_smoothing, train_sample_method,
                 train_beam_size, sc_beam_size, train_sample_n,
                 structure_loss_opts):
        super().__init__()
        self.model = model
        self.label_smoothing = label_smoothing
        if label_smoothing > 0:
            self.crit = losses.LabelSmoothing(smoothing=label_smoothing)
        else:
            self.crit = losses.LanguageModelCriterion()
        self.train_sample_method = train_sample_method
        self.train_beam_size = train_beam_size
        self.sc_beam_size = sc_beam_size
        self.train_sample_n = train_sample_n
        self.structure_loss_opts = structure_loss_opts
        self.rl_crit = losses.RewardCriterion()
        self.struc_crit = losses.StructureLosses(self.structure_loss_opts)

    def forward(self, fc_feats, att_feats, labels, masks, att_masks, gts, gt_indices,
                sc_flag, struc_flag, drop_worst_flag):
        opt = self.opt
        out = {}
        reduction = 'none' if drop_worst_flag else 'mean'

        if struc_flag:
            struc_loss_w = self.structure_loss_opts.structure_loss_weight
            struc_use_logsoftmax = self.structure_loss_opts.struc_use_logsoftmax
            struc_loss_type = self.structure_loss_opts.structure_loss_type
            if struc_loss_w < 1:
                lm_loss = self.crit(self.model(fc_feats, att_feats, labels[..., :-1], att_masks),
                                    labels[..., 1:], masks[..., 1:],
                                    reduction=reduction)
            else:
                lm_loss = torch.tensor(0).type_as(fc_feats)
            if struc_loss_w > 0:
                output_logsoftmax = (struc_use_logsoftmax or
                                     struc_loss_type == 'softmax_margin' or
                                     'margin' not in struc_loss_type)
                options = {'sample_method': self.train_sample_method,
                           'beam_size': self.train_beam_size,
                           'output_logsoftmax': output_logsoftmax,
                           'sample_n': self.train_sample_n}
                gen_result, sample_logprobs = self.model(fc_feats, att_feats, att_masks,
                                                         opt=options,
                                                         mode='sample')
                gts = [gts[_] for _ in gt_indices.tolist()]
                struc_loss = self.struc_crit(sample_logprobs, gen_result, gts, reduction=reduction)
            else:
                struc_loss = {'loss': torch.tensor(0).type_as(fc_feats),
                              'reward': torch.tensor(0).type_as(fc_feats)}
            loss = (1-struc_loss_w) * lm_loss + struc_loss_w * struc_loss['loss']
            out['lm_loss'] = lm_loss
            out['struc_loss'] = struc_loss['loss']
            out['reward'] = struc_loss['reward']
        elif not sc_flag:
            loss = self.crit(self.model(fc_feats, att_feats, labels[..., :-1], att_masks),
                             labels[..., 1:],
                             masks[..., 1:],
                             reduction=reduction)
        else:
            self.model.eval()
            with torch.no_grad():
                greedy_res, _ = self.model(fc_feats, att_feats, att_masks, mode='sample',
                                           opt={'sample_method': opt.sc_sample_method,
                                                'beam_size': opt.sc_beam_size})
            self.model.train()
            options = {'sample_method': self.train_sample_method,
                       'beam_size': self.train_beam_size,
                       'sample_n': self.train_sample_n}
            gen_result, sample_logprobs = self.model(fc_feats, att_feats, att_masks,
                                                     opt=options,
                                                     mode='sample')
            gts = [gts[_] for _ in gt_indices.tolist()]
            reward = get_self_critical_reward(greedy_res, gts, gen_result, self.opt)
            reward = torch.from_numpy(reward).to(sample_logprobs)
            loss = self.rl_crit(sample_logprobs, gen_result.data, reward, reduction=reduction)
            out['reward'] = reward[:, 0].mean()
        out['loss'] = loss
        return out


class CaptionStepV2(UpdateFunction):
    def __init__(self):
        self._train = True
        self._returns = ["loss", "correct", "labels", "total"]

    @property
    def train(self):
        return self._train

    @train.setter
    def train(self, x):
        self._train = x

    def returns(self):
        raise AttributeError("Cannot set attribute returns")

    def __call__(self, batch, criterion, model, optimizer, **kwargs):
        """Assumes that the sentence begins with a start token. Returns the named
        variables as a dictionary"""
        features, labels, masks, gts = batch["att_feats"], batch["labels"], batch["masks"], batch["gts"]
        features, labels, masks, gts = map(model.to_, [features, labels, masks, gts])
        if self.train:
            optimizer.zero_grad()
        labels_ = labels[..., 1:]
        masks_ = masks[..., 1:]
        predictions = model(batch["fc_feats"], features, labels[..., :-1], None)
        predictions = model.to_(predictions)
        loss = criterion(predictions, labels_, masks_, reduction="mean")
        inds = torch.where(masks_.view(*predictions.shape[:2]))
        correct = (predictions.argmax(2)[inds] == labels_.view(*predictions.shape[:2])[inds]).sum()
        # import ipdb; ipdb.set_trace()
        if self.train:
            loss.backward()
            optimizer.step()
        return {"loss": loss.data.item(), "labels": labels.detach().cpu(),
                "correct": correct.detach().cpu(), "total": len(inds[0])}


def main():
    parser = argparse.ArgumentParser()
    # data params
    parser.add_argument('model')
    parser.add_argument('dataset_name')
    parser.add_argument('--load-and-eval', action='store_true')
    parser.add_argument('--batch-size', '-b', type=int, default=128)
    parser.add_argument('--num-workers', type=int, default=4)
    # trainer params
    parser.add_argument('--gpus', default="0")
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--savedir', default="saves")
    parser.add_argument('--logdir', default="logs")
    # model setting
    parser.add_argument('--encoder', type=str, default="resnet50")
    parser.add_argument('--vis-dim', type=int, default=512)
    parser.add_argument('--vis-num', type=int, default=196)
    parser.add_argument('--embed-dim', type=int, default=512)
    parser.add_argument('--hidden-dim', type=int, default=512)
    parser.add_argument('--num-layers', type=int, default=1)
    parser.add_argument('--embed-dropout-ratio', type=float, default=0.5)
    parser.add_argument('--hidden-dropout-ratio', type=float, default=0.5)
    parser.add_argument('--fc-dropout-ratio', type=float, default=0.5)
    # optimizer setting
    parser.add_argument('--optimizer', default="adam")
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--num-epochs', type=int, default=50)
    args = parser.parse_args()

    print("Loading data with features")
    data, dataloaders, vocab = get_caption_data_with_features(
        "captiondata",
        args.dataset_name,
        {k: args.batch_size for k in
         ["train", "val", "test", "restval"]},
        {k: args.num_workers for k in
         ["train", "val", "test", "restval"]},
        mode="group")
    model_params = get_model_params(vocab, args)
    model_opt = SimpleNamespace(**model_params[args.model])
    if args.model not in model_params:
        raise ValueError(f"Unknown model {args.model}")
    model_opt.vocab_size = len(vocab)
    model_opt.input_encoding_size = model_opt.embed_dim
    model_opt.rnn_size = model_opt.hidden_dim
    model_opt.drop_prob_lm = 0.5
    model_opt.fc_feat_size = 2048
    model_opt.att_feat_size = 2048
    model_opt.att_hid_size = 512
    model_opt.caption_model = "att2in2"
    vocab.items = lambda: vocab.idx2word.items()
    model = models.setup(model_opt)
    model.model_name = args.model
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-9)
    update_function = CaptionStepV2()
    criterion = losses.LanguageModelCriterion().cuda()
    model = model.cuda(int(args.gpus))
    # torch.autograd.set_detect_anomaly(True)
    # from common_pyutil.monitor import Timer
    # timer = Timer()
    # it = dataloaders["train"].__iter__()
    # for _ in range(10):
    #     with timer:
    #         features, targets, lengths = it.__next__()
    #         features, targets = features.cuda(0), targets.cuda(0)
    #     print(timer.time)
    #     with timer:
    #         out = model(features, targets, lengths)
    #     print(timer.time)
    #     import ipdb; ipdb.set_trace()
    # it = dataloaders["train"].__iter__()
    # batch = it.__next__()
    # model.to_ = lambda x: x.to(torch.device(f"cuda:{args.gpus}"))
    # output = update_function(batch, criterion, model, optimizer)
    # import ipdb; ipdb.set_trace()
    trainer_params = {"gpus": [*map(int, args.gpus.split(","))], "cuda": True,
                      "seed": 42, "resume": True,
                      "metrics": ["loss", "correct", "total"], "val_frequency": 1,
                      "test_frequency": 5, "log_frequency": 5, "max_epochs": args.num_epochs}
    trainer = Trainer("caption_trainer", trainer_params, optimizer, model, data,
                      dataloaders, update_function, criterion,
                      args.savedir, args.logdir, ddp_params={},
                      extra_opts={"model_params": model_params[args.model]})
    trainer.trainer_params.save_best_on = "val"
    trainer.trainer_params.save_best_by = "loss"
    desc = trainer.describe_hook("post_batch_hook")
    if not any("post_batch_progress" in x for x in desc):
        trainer.add_to_hook_at_end("post_batch_hook", functions.post_batch_progress)
    trainer._save_best_predicate = check_min
    trainer.start()


if __name__ == '__main__':
    main()
