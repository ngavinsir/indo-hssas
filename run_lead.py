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

ex = Experiment(name='run-lead', ingredients=[corpus_ingredient])

@ex.capture
def prepare_vocab(_log):
    _log.info(f'preparing vocabulary...')
    counter = Counter()
    dataset = IndosumDataset(read_train_jsonl())
    for sentences, _ in dataset:
        counter.update(list(itertools.chain(*sentences)))
    vectors = Vectors('./id-vectors.txt', cache='./')
    return Vocab(counter, vectors=vectors)

@ex.automain
def evaluate():
    vocab = prepare_vocab()
    

    
