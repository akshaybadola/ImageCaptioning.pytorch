from typing import List, Dict, Optional, Union
import os
import json
from collections import defaultdict

import torch
import numpy as np


def decode_captions(captions, idx_to_word: Dict):
    N, D = captions.shape
    if isinstance(captions, torch.Tensor):
        captions = captions.cpu().numpy()
    decoded = []
    for idx in range(N):
        words = []
        for wid in range(D):
            word = idx_to_word[captions[idx, wid]]
            if word == "<start>":
                continue
            elif word == "<end>":  # in {'<end>', '<start>', '<unk>'}:
                words.append('.')
            else:
                words.append(word)
        decoded.append(words)
    return decoded


def convert_predictions_and_targets_to_words(predictions, targets, vocab):
    predictions = map(vocab.idx2word.__getitem__,
                      torch.max(predictions, 1)[1].detach().cpu().numpy())
    targets = map(vocab.idx2word.__getitem__, targets.detach().cpu().numpy())
    return list(zip(predictions, targets))


def unpad_targets_k(targets, lengths, k=1, has_start_token=True):
    retvals = []
    if k == 1:
        for t, l in zip(targets, lengths):
            if has_start_token:
                retvals.append(t[1: l - 1].tolist())
            else:
                retvals.append(t[:l - 1].tolist())
    else:
        for t, l in zip(targets, lengths):
            temp_sentences = []
            for t_, l_ in zip(t, l):
                if has_start_token:
                    temp_sentences.append(t_[1: l_ - 1].tolist())
                else:
                    temp_sentences.append(t_[:l_ - 1].tolist())
            retvals.append(temp_sentences)
    return retvals


def unpad_targets(targets, lengths, has_start_token=True):
    retvals = []
    for t, l in zip(targets, lengths):
        temp_sentences = []
        for t_, l_ in zip(t, l):
            if has_start_token:
                temp_sentences.append(t_[1: l_ - 1].tolist())
            else:
                temp_sentences.append(t_[:l_ - 1].tolist())
        retvals.append(temp_sentences)
    return retvals


def takewhile(seq, predicate):
    len_seq = len(seq)
    j = 0
    while j < len_seq and predicate(seq[j]):
        yield seq[j]
        j += 1


def unpad_hypotheses(hypotheses, pad_token=2):
    return [[*takewhile(h if isinstance(h, list) else h.tolist(),
                        lambda x: x != pad_token)]
            for h in hypotheses]



class Vocabulary(object):
    # TODO: Add <start>, <end> and <unk> consistently
    #       Should I add <pad> also?
    def __init__(self, words=None, special_tokens=True, pad_is_end=False):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0
        self._counts = defaultdict(int)
        self._all_counts = defaultdict(int)
        self._truncated = False
        self._pad_is_end = pad_is_end
        self._special_tokens = special_tokens
        if self._special_tokens:
            self.add_special_tokens()
        # intialize with words
        if words is not None:
            for w in words:
                self.add_word(w)
                self._all_counts[w] += 1
            self._counts = self._all_counts.copy()
        self._compute_sorted()

    def add_special_tokens(self):
        if not self.pad_is_end:
            self.add_word("<pad>")
            self.add_word("<start>")
            self.add_word("<end>")
        else:
            self.add_word("<end>")
            self.add_word("<start>")
            self.word2idx['<pad>'] = self.word2idx['<end>']
        self.add_word("<unk>")

    def _init(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0
        if self._special_tokens:
            self.add_special_tokens()
        for word in self._counts:
            self.add_word(word)
        self._compute_sorted()

    @property
    def special_tokens(self):
        return self._special_tokens

    @property
    def pad_is_end(self):
        return self._pad_is_end

    def reset(self):
        self._truncated = False
        self._counts = self._all_counts.copy()
        self._init()

    def items(self):
        return self.idx2word.items()

    def truncate(self, cutoff_by, threshold, force=False):
        """Removes least frequent words. Cuts off the word by specified method

        :param threshold: threshold for cutoff
        :param cutoff_by: cutoff method
        :param force:
        :returns:
        :rtype:

        """
        if self._truncated and not force:
            print("Vocab is already _truncated, use force to truncate again")
        elif self._truncated and force:
            print("Forcing truncation.")
        # tokens = nltk.tokenize.word_tokenize(str(corpus).lower())
        # either we keep the original counts somewhere
        if cutoff_by == "total":
            counts = dict(self._sorted_counts[:threshold])
        elif cutoff_by == "frequency":
            # remove if frequency is smaller than threshold
            if threshold <= 0 or not isinstance(threshold, int):
                raise ValueError("threshold must be and integer and greater than 0")
            counts = self._counts.copy()
            tokens = set(self._counts.keys())
            for x in tokens:
                if counts[x] < threshold:
                    counts.pop(x)
        elif cutoff_by == "sort":
            # remove if the frequency is in bottom threshold fraction
            assert threshold < 0.5 and threshold > 0, "threshold must be between 0 and 0.5"
            counts_list = [*self._counts.items()]
            counts_list.sort(key=lambda x: x[1])
            counts = self._counts.copy()
            # print(len([_ for _ in self._counts if self._counts[_] > 1]))
            for x in counts_list:
                if (counts_list.index(x) < int(len(counts_list) * threshold))\
                   or counts[x[0]] == 1:
                    counts.pop(x[0])
        elif cutoff_by == "relative":
            # remove if relative frequency is smaller than a specified fraction
            # e.g., if most frequent is "the" and it occurs 1000 times and
            # threshold is .01, then remove all words occuring fewer than 10 times
            # Actually it could be total also
            assert threshold < 1.0, "threshold must be smaller than 0"
            counts_list = [*self._counts.items()]
            counts_list.sort(key=lambda x: x[1])
            counts = self._counts.copy()
            for x in counts_list[len(counts_list) * threshold:]:
                counts.pop(x[0])
            # Actually not implementing it right now
            raise NotImplementedError
        else:
            raise ValueError(f"Unknown type of cutoff {cutoff_by}")
        self._counts = counts
        self._truncated = True
        self._init()

    def _compute_sorted(self):
        self._sorted_counts = [(k, v) for k, v in self._counts.items()]
        self._sorted_counts.sort(key=lambda x: x[1], reverse=True)

    def topk(self, k: Union[int, float]):
        if isinstance(k, int):
            return [*filter(lambda c: c[1] > k, self._sorted_counts)]
        elif isinstance(k, float):
            len_counts = np.array([*self._sorted_counts.values()])
            len_counts /= len_counts.sum()
            len_counts = len_counts.cumsum()
            return [*filter(lambda x, y: y > k, zip(self._sorted_counts, len_counts))]  # type: ignore
        else:
            raise TypeError("Can only be int or float")

    def load_from_json(self, json_file):
        with open(os.path.expanduser(json_file)) as f:
            saved_state = json.load(f)
        # NOTE: Although this works for now, but it's dangerous
        self.__dict__.update(saved_state)
        self.idx2word = dict((int(k), v) for k, v in self.idx2word.items())
        self.word2idx = dict((k, int(v)) for k, v in self.word2idx.items())
        self.idx = int(self.idx)
        self._counts = dict((k, int(v)) for k, v in self._counts.items())
        assert all([x in self.word2idx
                    for x in ["<pad>", "<start>", "<end>", "<unk>"]])

    def dump_to_json(self, json_file):
        with open(os.path.expanduser(json_file), "w") as f:
            json.dump(self.__dict__, f)

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1
        self._all_counts[word] += 1
        self._counts[word] += 1

    def __call__(self, word):
        if word not in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return self.idx
