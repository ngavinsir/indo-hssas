from sacred import Experiment

from ingredients.corpus import (
    ing as corpus_ingredient,
    read_train_jsonl,
    read_dev_jsonl,
    read_test_jsonl,
)
from data import IndosumDataset, IndosumDataModule
import torch
import itertools
from model import HSSAS
from torch import nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from utils import eval_summaries

ex = Experiment(name="run_hssas", ingredients=[corpus_ingredient])


@ex.config
def config():
    # pretrained embedding path
    embedding_path = "./id-vectors.txt"
    # word embedding dimension
    embedding_dim = 300
    # lstm hidden size
    lstm_hidden_size = 200
    # attention size
    attention_size = 400
    # saved model path
    model_path = "./lightning_logs/version_138/checkpoints/epoch=6.ckpt"
    # delete temporary folder to save summaries
    delete_temps = False


@ex.command
def test():
    vocab = prepare_vocab()
    print(len(vocab.vectors))


@ex.command
def train(embedding_dim, lstm_hidden_size, attention_size, embedding_path):
    dm = IndosumDataModule(
        read_train_jsonl(), read_dev_jsonl(), read_test_jsonl(), embedding_path
    )
    hssas = HSSAS(dm.vocab, embedding_dim, lstm_hidden_size, attention_size)

    trainer = pl.Trainer(
        gpus=1,
        callbacks=[EarlyStopping(monitor="val_loss")],
    )
    trainer.fit(hssas, dm)


@ex.automain
def evaluate(
    model_path,
    delete_temps,
    embedding_dim,
    embedding_path,
    lstm_hidden_size,
    attention_size,
    _log,
    _run,
):
    hssas = HSSAS.load_from_checkpoint(model_path)

    docs = read_test_jsonl()
    dm = IndosumDataModule(
        read_train_jsonl(), read_dev_jsonl(), read_test_jsonl(), embedding_path
    )
    summaries = (summary for x, _ in dm.test_dataloader() for summary in hssas(x))

    abs_score, ext_score = eval_summaries(summaries, docs, logger=_log, delete_temps=delete_temps)
    for name, value in abs_score.items():
        _run.log_scalar(name, value)
    for name, value in ext_score.items():
        _run.log_scalar(name, value)
    return abs_score["ROUGE-1-F"], ext_score["ROUGE-1-F"]
