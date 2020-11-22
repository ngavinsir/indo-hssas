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
from utils import eval_summaries, setup_mongo_observer
from pytorch_lightning.callbacks import ModelCheckpoint
from gensim.models import Word2Vec

ex = Experiment(name="run_hssas", ingredients=[corpus_ingredient])
setup_mongo_observer(ex)


@ex.config
def config():
    # pretrained embedding path
    embedding_path = "./idwiki_word2vec_100.txt"
    # word embedding dimension
    embedding_dim = 100
    # lstm hidden size
    lstm_hidden_size = 200
    # attention size
    attention_size = 400
    # saved model path
    model_path = "./lightning_logs/version_257/checkpoints/epoch=0.ckpt"
    # delete temporary folder to save summaries
    delete_temps = False
    # batch size
    batch_size = 8
    # model's optimizer learning rate
    learning_rate = 1
    # max document's sentence length
    max_doc_len = 100
    # max sentence's word length
    max_sen_len = 50
    # resume trainer from path
    resume_path = None

    word2vec_model = "./idwiki_word2vec_100.model"


@ex.command
def model_to_word2vec(word2vec_model):
    model = Word2Vec.load(word2vec_model)
    model.wv.save_word2vec_format("word2vec.txt")


@ex.command
def test(
    model_path,
    embedding_path,
    batch_size,
    embedding_dim,
    lstm_hidden_size,
    attention_size,
):
    dm = IndosumDataModule(
        read_train_jsonl(), read_dev_jsonl(), read_test_jsonl(), embedding_path, 128
    )
    hssas = HSSAS(dm.vocab, embedding_dim, lstm_hidden_size, attention_size)
    total = 0
    x = 0
    for _, y in dm.train_dataloader():
        total += y[:, 0].sum().item()
        x += 128
    print(total, x)


@ex.command
def evaluate(
    model_path,
    delete_temps,
    embedding_path,
    batch_size,
    max_doc_len,
    max_sen_len,
    _log,
    _run,
    data_module=None,
):
    hssas = HSSAS.load_from_checkpoint(model_path)

    docs = read_test_jsonl()
    if data_module == None:
        data_module = IndosumDataModule(
            read_train_jsonl(),
            read_dev_jsonl(),
            read_test_jsonl(),
            embedding_path,
            max_doc_len,
            max_sen_len,
            batch_size,
        )
    summaries = (
        summary for x, _ in data_module.test_dataloader() for summary in hssas(x)
    )

    abs_score, ext_score = eval_summaries(
        summaries, docs, logger=_log, delete_temps=delete_temps
    )
    for name, value in abs_score.items():
        _run.log_scalar(name, value)
    for name, value in ext_score.items():
        _run.log_scalar(name, value)
    return abs_score["ROUGE-1-F"], ext_score["ROUGE-1-F"]


@ex.automain
def train(
    embedding_dim,
    lstm_hidden_size,
    attention_size,
    embedding_path,
    batch_size,
    max_doc_len,
    max_sen_len,
    learning_rate,
    resume_path,
):
    dm = IndosumDataModule(
        read_train_jsonl(),
        read_dev_jsonl(),
        read_test_jsonl(),
        embedding_path,
        max_doc_len,
        max_sen_len,
        batch_size,
    )
    hssas = HSSAS(
        dm.vocab,
        embedding_dim,
        lstm_hidden_size,
        attention_size,
        list(read_dev_jsonl()),
        learning_rate,
    )

    checkpoint_callback = ModelCheckpoint(monitor="val_loss")
    trainer = pl.Trainer(
        gpus=1,
        callbacks=[EarlyStopping(monitor="val_loss", mode="min", patience=5)],
        checkpoint_callback=checkpoint_callback,
        gradient_clip_val=5,
        resume_from_checkpoint=resume_path,
        max_epochs=5000,
    )
    trainer.fit(hssas, dm)
    evaluate(model_path=checkpoint_callback.best_model_path, data_module=dm)
