from __future__ import print_function
import os
import json
import numpy as np
import utils
import h5py
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

try:
    import cPickle as pkl
except ModuleNotFoundError:
    import _pickle as pkl

class Dictionary(object):
    def __init__(self, word2idx=None, idx2word=None):
        if word2idx is None:
            word2idx = {}
        if idx2word is None:
            idx2word = []
        self.word2idx = word2idx
        self.idx2word = idx2word

    @property
    def ntoken(self):
        return len(self.word2idx)

    @property
    def padding_idx(self):
        return len(self.word2idx)

    def tokenize(self, sentence, add_word):
        sentence = sentence.lower()
        sentence = sentence.replace(',', '').replace('?', '').replace('\'s', ' \'s')
        words = sentence.split()
        tokens = []
        if add_word:
            for w in words:
                tokens.append(self.add_word(w))
        else:
            for w in words:
                tokens.append(self.word2idx[w])
        return tokens

    def dump_to_file(self, path):
        pkl.dump([self.word2idx, self.idx2word], open(path, 'wb'))
        print('dictionary dumped to %s' % path)

    @classmethod
    def load_from_file(cls, path):
        print('loading dictionary from %s' % path)
        word2idx, idx2word = pkl.load(open(path, 'rb'))
        d = cls(word2idx, idx2word)
        return d

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


def _create_entry(img, question, answer):
    answer.pop('image_id')
    answer.pop('question_id')
    entry = {
        'question_id' : question['question_id'],
        'image_id'    : question['image_id'],
        'image'       : img,
        'question'    : question['question'],
        'answer'      : answer}
    return entry


def _load_dataset(dataroot, name, img_id2val):
    """Load entries

    img_id2val: dict {img_id -> val} val can be used to retrieve image or features
    dataroot: root path of dataset
    name: 'train', 'val'
    """
    question_path = os.path.join(
        dataroot, 'v2_OpenEnded_mscoco_%s2014_questions.json' % name)
    questions = sorted(json.load(open(question_path))['questions'],
                       key=lambda x: x['question_id'])
    answer_path = os.path.join(dataroot, 'cache', '%s_target.pkl' % name)
    answers = pkl.load(open(answer_path, 'rb'))
    answers = sorted(answers, key=lambda x: x['question_id'])

    utils.assert_eq(len(questions), len(answers))
    entries = []
    for question, answer in zip(questions, answers):
        utils.assert_eq(question['question_id'], answer['question_id'])
        utils.assert_eq(question['image_id'], answer['image_id'])
        img_id = question['image_id']
        entries.append(_create_entry(img_id2val[img_id], question, answer))

    return entries


class VQAFeatureDataset(Dataset):
    def __init__(self, name, dictionary, args, dataroot='vqa_data'):
        super(VQAFeatureDataset, self).__init__()
        assert name in ['train', 'val']
        self.split_name = name
        self.model = args.model
        # self.image_size = args.image_size
        self.cpu_size = args.cpu_size

        ans2label_path = os.path.join(dataroot, 'cache', 'trainval_ans2label.pkl')
        label2ans_path = os.path.join(dataroot, 'cache', 'trainval_label2ans.pkl')
        self.ans2label = pkl.load(open(ans2label_path, 'rb'))
        self.label2ans = pkl.load(open(label2ans_path, 'rb'))
        self.num_ans_candidates = len(self.ans2label)

        self.dictionary = dictionary

        print('loading features from h5 file')
        features_path = os.path.join(dataroot, 'features', args.feature_dir)

        if self.model in ['soft']:
            with open(os.path.join(features_path, '{}_id.pkl'.format(name)), 'rb') as f:
                self.img_id2idx = pkl.load(f)

            h5_path = os.path.join(features_path, '{}_feat.h5'.format(name))

            if self.cpu_size == 128:
                with h5py.File(h5_path, 'r') as hf:
                    self.features = np.array(hf.get('image_features'))
            elif self.cpu_size == 64:
                if self.split_name == 'val':
                    with h5py.File(h5_path, 'r') as hf:
                        self.features = np.array(hf.get('image_features'))
                else:
                    hf = h5py.File(h5_path, 'r')
                    self.features = hf['image_features']
            else:
                hf = h5py.File(h5_path, 'r')
                self.features = hf['image_features']
        else:
            raise ValueError

        self.entries = _load_dataset(dataroot, name, self.img_id2idx)

        self.tokenize()
        self.tensorize()

        if self.model in ['soft']:
            size_dim = 2
        else:
            raise ValueError

        self.v_dim = self.features.shape[size_dim]

    def tokenize(self, max_length=14):
        """Tokenizes the questions.

        This will add q_token in each entry of the dataset.
        -1 represent nil, and should be treated as padding_idx in embedding
        """
        for entry in tqdm(self.entries):
            tokens = self.dictionary.tokenize(entry['question'], False)
            tokens = tokens[:max_length]
            if len(tokens) < max_length:
                # Note here we pad in front of the sentence
                padding = [self.dictionary.padding_idx] * (max_length - len(tokens))
                tokens = padding + tokens
            utils.assert_eq(len(tokens), max_length)
            entry['q_token'] = tokens

    def tensorize(self):

        if self.cpu_size == 128:
            self.features = torch.from_numpy(self.features)
        elif self.cpu_size == 64:
            if self.split_name == 'val':
                self.features = torch.from_numpy(self.features)
        else:
            pass

        for entry in tqdm(self.entries):
            question = torch.from_numpy(np.array(entry['q_token']))
            entry['q_token'] = question

            answer = entry['answer']
            labels = np.array(answer['labels'])
            scores = np.array(answer['scores'], dtype=np.float32)
            if len(labels):
                labels = torch.from_numpy(labels)
                scores = torch.from_numpy(scores)
                entry['answer']['labels'] = labels
                entry['answer']['scores'] = scores
            else:
                entry['answer']['labels'] = None
                entry['answer']['scores'] = None

    def __getitem__(self, index):
        entry = self.entries[index]

        if self.cpu_size == 128:
            features = self.features[entry['image']]
        elif self.cpu_size == 64:
            if self.split_name == 'val':
                features = self.features[entry['image']]
            else:
                features = torch.from_numpy(self.features[entry['image']])
        else:
            features = torch.from_numpy(self.features[entry['image']])

        question = entry['q_token']
        answer = entry['answer']
        labels = answer['labels']
        scores = answer['scores']
        target = torch.zeros(self.num_ans_candidates)
        if labels is not None:
            target.scatter_(0, labels, scores)

        return features, question, target

    def __len__(self):
        return len(self.entries)
