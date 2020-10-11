from sacred import Experiment

from ingredients.corpus import ing as corpus_ingredient, read_train_jsonl
from data import IndosumDataset
from torch.utils.data import DataLoader
import torch
import itertools
from collections import Counter
from torchtext.vocab import Vocab, Vectors
import gensim
import sys
from model import HSSAS
from torch import nn

ex = Experiment(name='run_hssas', ingredients=[corpus_ingredient])

@ex.config
def config():
    # pretrained embedding path
    embedding_path = './id-vectors.txt'
    # word embedding dimension
    embedding_dim = 300
    # lstm hidden size
    lstm_hidden_size = 200
    # attention size
    attention_size = 400

@ex.capture
def prepare_vocab(embedding_path, _log):
    _log.info(f'preparing vocabulary...')
    counter = Counter()
    dataset = IndosumDataset(read_train_jsonl())
    for sentences, _ in dataset:
        counter.update(list(itertools.chain(*sentences)))
    vectors = Vectors(embedding_path, cache='./')
    return Vocab(counter, vectors=vectors)

@ex.automain
def evaluate(embedding_dim, lstm_hidden_size, attention_size):
    vocab = prepare_vocab()
    hssas = HSSAS(vocab, embedding_dim, lstm_hidden_size, attention_size)
    for sentences, _ in IndosumDataset(read_train_jsonl()):
        y = hssas(sentences)
        print(y)
        return
    
