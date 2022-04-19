from typing import List, Dict, Optional, Union
from pathlib import Path
import os
import json
import random

import h5py
import numpy as np
from PIL import Image

import torch
import torch.utils.data as data
from torchvision import models, transforms
import nltk

from common_pyutil.monitor import Timer

from vocab import Vocabulary


class ResnetAttention(torch.nn.Module):
    def __init__(self, arch, att_size, resnet=None):
        super(ResnetAttention, self).__init__()
        if resnet is not None:
            self.resnet = getattr(models, arch)(pretrained=True)
        else:
            self.resnet = resnet
        self.att_size = att_size
        self.conv1 = self.resnet.conv1
        self.bn1 = self.resnet.bn1
        self.relu = self.resnet.relu
        self.maxpool = self.resnet.maxpool
        self.layer1 = self.resnet.layer1
        self.layer2 = self.resnet.layer2
        self.layer3 = self.resnet.layer3
        self.layer4 = self.resnet.layer4
        for dest, source in zip(self.parameters(), self.resnet.parameters()):
            dest.data.copy_(source.data)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        fc = x.mean(3).mean(2).squeeze()
        # Which one?
        # att = F.adaptive_avg_pool2d(x,[att_size,att_size]).squeeze().permute(1, 2, 0)
        # Current one seems to be working
        att = torch.nn.functional.adaptive_avg_pool2d(x, [self.att_size, self.att_size]).permute(0, 2, 3, 1)
        return fc, att


class TempData(data.Dataset):
    def __init__(self, img_paths, transform, return_name=False):
        self.transform = transform
        self._img_paths = img_paths
        self.return_name = return_name
        assert self.transform is not None

    def __getitem__(self, indx):
        if self.return_name:
            return self._img_paths[indx], self.transform(Image.open(
                self._img_paths[indx]).convert("RGB"))
        else:
            return self.transform(Image.open(self._img_paths[indx]).convert("RGB"))

    def __len__(self):
        return len(self._img_paths)


def get_caption_all_tokens(annotations_path):
    """Return ALL the tokens for caption annotations
    """
    with open(annotations_path) as f:
        annotations = json.load(f)
    tokens = []
    for item in annotations["images"]:
        for sent in item["sentences"]:
            tokens.extend(sent["tokens"])
    return tokens


def get_caption_train_tokens(annotations_path):
    """Return the tokens in ONLY the training split for caption annotations
    """
    tokens = []
    with open(annotations_path) as f:
        annotations = json.load(f)
    for item in annotations['images']:
        if item['split'] == "train":
            for sent in item['sentences']:
                tokens.extend(sent["tokens"])
    return tokens


def get_non_unk_counts(annotations_path, vocab):
    """Get counts of words and tokens which are not `<unk>` from dataset for each split
    """
    with open(annotations_path) as f:
        annotations = json.load(f)
    words = set(vocab.word2idx.keys())
    tokens = {"train": [], "val": [], "test": [], "restval": []}
    for item in annotations['images']:
        for sent in item['sentences']:
            tokens[item['split']].extend(sent["tokens"])
    unk_count = {}
    for k, v in tokens.items():
        counts = [t in words for t in v]
        unk_count[k] = {"not_unk": np.sum(counts), "total": len(counts)}
    unk_count["overall_percentage"] = (sum([unk_count[k]["not_unk"]
                                            for k in ["train", "val", "test", "restval"]]) /
                                       sum([unk_count[k]["total"]
                                            for k in ["train", "val", "test", "restval"]]))
    return unk_count


class CaptionData:
    def __init__(self, data_name: str, model_name: str, backbone: torch.nn.Module,
                 backbone_gpu: Optional[int], image_root: str,
                 feature_path: Optional[str], ann_path: str, vocab: Vocabulary,
                 labels_file: str,
                 in_memory: bool):
        self._data_name = data_name
        self._model_name = model_name  # backbone name
        self._image_root = image_root
        self._ann_path = ann_path
        self._feature_path = feature_path
        self._vocab = vocab
        self._labels_file = labels_file
        self._backbone = backbone
        self._backbone_gpu = backbone_gpu
        if self._backbone is not None:
            self._backbone = self._backbone.eval()
            self._backbone.requires_grad_(False)
        with open(self._ann_path) as f:
            self._annotations = json.load(f)
        self._img_names = os.listdir(self._image_root)
        self._data = [a for a in self._annotations['images']]
        self._data.sort(key=lambda x: x["imgid"])
        if self._feature_path is not None:
            if not os.path.exists(self._feature_path):
                self.maybe_extract_features()
            self._features = self.load_features()
        else:
            self._features = None
        if "filepath" in self._data[0]:
            self._filepath = self._data[0]["filepath"]
        else:
            self._filepath = None

    def maybe_extract_features(self):
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        normalize = transforms.Normalize(mean=mean, std=std)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
            normalize])
        if self._backbone_gpu is not None:
            device = torch.device(f"cuda:{self._backbone_gpu}")
        else:
            device = torch.device("cpu")
        if self._backbone is not None:
            self._backbone = self._backbone.to(device)
            self._backbone = self._backbone.eval()
        else:
            raise AttributeError("Backbone not given and feature extraction asked")
        h5_mode = "w"
        features_shape = [2048, [14, 14, 2048]]
        img_paths = {x['imgid']: os.path.join(self._image_root, x['filename'])
                     for x in self._data}
        print(f"Extracting features to {self._feature_path}")
        timer = Timer()
        with h5py.File(self._feature_path, h5_mode) as h5_file:
            _data = TempData(img_paths, transform, return_name=False)
            temp_loader = data.DataLoader(_data,
                                          batch_size=256,
                                          num_workers=12,
                                          shuffle=False,
                                          pin_memory=True,
                                          drop_last=False)
            fc_dset = h5_file.create_dataset('fc', shape=(len(_data),
                                                          features_shape[0]),
                                             chunks=(1, features_shape[0]),
                                             dtype='float32')
            att_dset = h5_file.create_dataset('att', shape=(len(_data),
                                                            *features_shape[1]),
                                              chunks=(1, *features_shape[1]),
                                              dtype='float32')
            with torch.no_grad():
                for i, batch in enumerate(temp_loader):
                    with timer:
                        tensors = batch
                    # print(f"Got batch in {timer.time} seconds")
                    with timer:
                        features = self._backbone(tensors.to(device))
                    # print(f"Got features in {timer.time} seconds")
                    bs = features[0].shape[0]
                    with timer:
                        fc_dset[i*bs: (i+1)*bs] = features[0].detach().cpu()
                        att_dset[i*bs: (i+1)*bs] = features[1].detach().cpu()
                    # print(f"Written to h5 file in {timer.time} seconds")
                    if (i+1) % 10 == 0:
                        print(f"{i+1} out of {len(temp_loader)} iterations done")

    def load_features(self):
        if self._feature_path.endswith(".npz"):
            with open(".".join(self._feature_path.split(".")[:-1]) + "_names.json") as f:
                img_names = json.load(f)
            features = np.load(self._feature_path)['arr_0']
            assert len(img_names) == len(features)
            self._features = {"names": img_names, "features": features}
            self._h5_feature_path = None
        elif self._feature_path.endswith(".h5"):
            self._features = None
            self._h5_feature_path = self._feature_path
        else:
            raise ValueError(f"Unknown extension for features {self._feature_path.split('.')[-1]}")

    def get_data(self, split, transform, mode="single"):
        data = [a for a in self._annotations['images']
                if split == a['split']]
        if self._features or self._feature_path:
            assert transform is None, "transform cannot be given with features"
            backbone = None
            backbone_gpu = None
        else:
            backbone = self._backbone
            backbone_gpu = self._backbone_gpu
        image_names = [x["filename"] for x in data]
        return CaptionDataset(data, self._image_root, image_names, self._vocab,
                              self._features, self._h5_feature_path,
                              backbone, backbone_gpu, mode, self._labels_file, transform)


class CaptionDataset(data.Dataset):
    def __init__(self, data, image_root, img_names, vocab, features, h5_features_path,
                 backbone, backbone_gpu, mode, labels_file, transform):
        self._data = data
        self._image_root = image_root
        self._img_names = img_names
        self._vocab = vocab
        self._backbone = backbone
        self._backbone_gpu = backbone_gpu
        if self._backbone is not None or self._backbone_gpu is not None:
            print("backbone and backbone_gpu are ignored for now")
        self._mode = mode
        if self._mode not in {"single", "group"}:
            raise ValueError(f"Unknown mode {self._mode}")
        self.transform = transform
        self.collate_fn = self._collate_fn_normal
        self._data = []
        self._imgs = [os.path.join(self._image_root, x) for x in self._img_names]
        self._captions = []
        self._features = features
        self._h5_features_path = h5_features_path
        self.h5_file = h5py.File(self._h5_features_path, 'r')
        if self._h5_features_path is not None:
            self._features = True
        elif self._features is not None:
            self._features = self._features["features"]
        if self._mode == "single":
            for i, d in enumerate(data):
                captions = [x['tokens'] for x in d['sentences']]
                for c in captions:
                    self._data.append((i, c))
                    self._captions.append(c)
            self._labels_file = None
        elif self._mode == "group":
            self._labels_file = h5py.File(labels_file, mode="r")
            self.labels = self._labels_file['labels'][:]
            self._seq_length = self.labels[0].shape[0]
            print('max sequence length in data is', self._seq_length)
            # load the pointers in full to RAM (should be small enough)
            self.labels_start_ix = self._labels_file['label_start_ix'][:]
            self.labels_end_ix = self._labels_file['label_end_ix'][:]
            for i, d in enumerate(data):
                self._data.append({"imgid": d["imgid"], "filename": d["filename"]})
                self._captions.extend([x['tokens'] for x in d['sentences']])
        self._seq_per_img = 5
        self._use_fc = False
        self._repeat = False

    @property
    def repeat(self):
        return self._repeat

    def features(self, index):
        """Return features correspoding to `index` in the h5 file."""
        if self._use_fc:
            return {"att_Feats": self.h5_file["att"][index],
                    "fc_feats": self.h5_file["fc"][index]}
        else:
            return {"att_feats": self.h5_file["att"][index],
                    "fc_feats": []}

    def get_captions_from_label_file(self, ix):
        # fetch the sequence labels
        ix1 = self.labels_start_ix[ix] - 1  # label_start_ix starts from 1
        ix2 = self.labels_end_ix[ix] - 1
        ncap = ix2 - ix1 + 1 # number of captions available for this image
        assert ncap > 0, 'an image does not have any label. this can be handled but right now isn\'t'
        if ncap < self._seq_per_img:
            # we need to subsample (with replacement)
            seq = np.zeros([self._seq_per_img, self._seq_length], dtype='int32')
            for q in range(self._seq_per_img):
                ixl = random.randint(ix1, ix2)
                seq[q, :] = self.labels[ixl, :self._seq_length]
        else:
            ixl = random.randint(ix1, ix2 - self._seq_per_img + 1)
            seq = self.labels[ixl: ixl + self._seq_per_img, :self._seq_length]
        return seq

    def get_feature_caption_single(self, index):
        feature_or_image_index, caption = self._data[index]
        if self._features is not None:
            return self.features(feature_or_image_index), caption
        else:
            img = self.transform(self.get_image(feature_or_image_index))
            return img, caption

    def get_feature_captions_group(self, index):
        img_id, filename = self._data[index].values()
        feature = self.features(img_id)
        captions = self.get_captions_from_label_file(index)
        return feature, np.int32(captions)

    def get_image(self, index) -> Image:
        return Image.open(self._imgs[index])

    @property
    def imgs(self) -> List[str]:
        return self._imgs

    @property
    def captions(self):
        return self._captions

    def _get_caption_tensor(self, tokens):
        vocab = self._vocab
        target = [vocab('<start>'), *[vocab(word) for word in tokens],
                  vocab('<end>')]
        target = torch.Tensor(target).long()
        return target

    def _get_captions_tensor(self, captions):
        captions = [nltk.tokenize.word_tokenize(str(caption).lower())
                    for caption in captions]
        lengths = [len(c)+2 for c in captions]
        targets = torch.zeros(len(captions), max(lengths)).long()
        for i, (caption, length) in enumerate(zip(captions, lengths)):
            targets[i, 0] = self._vocab('<start>')
            targets[i, 1:length-1] = torch.tensor([self._vocab(word) for word in caption])
            targets[i, length-1] = self._vocab('<end>')
        return targets, lengths

    def __getitem__(self, index):
        if self._mode == "single":
            feature, caption = self.get_feature_caption_single(index)
            caption = self._get_caption_tensor(caption)
            # if self._topk else self._get_caption_tensor(caption)
            if not isinstance(feature, torch.Tensor):
                return {k: torch.tensor(v) for k, v in feature.items()}, caption
        else:
            feature, captions = self.get_feature_captions_group(index)
            labels = np.zeros((captions.shape[0], captions.shape[1]+2), dtype='int32')
            labels[:, 1:-1] = captions.copy()
            masks = np.zeros((captions.shape[0], captions.shape[1]+2), dtype='float32')
            masks_ix = (labels > 0).sum(1) + 2
            for i in range(masks.shape[0]):
                masks[i, :masks_ix[i]] = 1
            if self._repeat:
                feature = {k: torch.tensor(feature[k]).unsqueeze(0).repeat(len(captions), 1, 1, 1)
                           for k in feature}
            else:
                feature = {k: torch.tensor(feature[k]) for k in feature}
            return feature, torch.tensor(captions), torch.tensor(labels), torch.tensor(masks)

    def _collate_fn_group(self, data):
        features, captions, labels, masks = zip(*data)
        if self._repeat:
            return {"fc_feats": torch.tensor([[] for _ in features]),
                    "att_feats": torch.cat([x["att_feats"] for x in features], 0),
                    "att_masks": None,
                    "labels": torch.cat(labels, 0).long(),
                    "masks": torch.cat(masks, 0),
                    "gts": torch.cat(captions, 0)}
        else:
            return {"fc_feats": torch.tensor([[] for _ in features]),
                    "att_feats": torch.stack([x["att_feats"] for x in features], 0),
                    "att_masks": None,
                    "labels": torch.stack(labels, 0).long(),
                    "masks": torch.stack(masks, 0),
                    "gts": torch.stack(captions, 0)}


    def _collate_fn_normal(self, data):
        # data.sort(key=lambda x: len(x[1]), reverse=True)
        features, captions = zip(*data)
        features = torch.stack(features, 0)
        # batch_size-by-512-196
        lengths = [len(cap) for cap in captions]
        targets = torch.zeros(len(captions), max(lengths)).long()
        for i, cap in enumerate(captions):
            end = lengths[i]
            targets[i, :end] = cap[:end]
        return features, targets, lengths

    def __len__(self):
        return len(self._data)
