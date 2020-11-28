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
from utils import eval_summaries, setup_mongo_observer, extract_preds
from pytorch_lightning.callbacks import ModelCheckpoint
from gensim.models import Word2Vec

ex = Experiment(name="run_hssas", ingredients=[corpus_ingredient])
setup_mongo_observer(ex)


@ex.config
def config():
    seed = 472290281
    # pretrained embedding path
    embedding_path = "./idwiki_word2vec_100.txt"
    # word embedding dimension
    embedding_dim = 100
    # lstm hidden size
    lstm_hidden_size = 25
    # attention size
    attention_size = 50
    # saved model path
    model_path = "./lightning_logs/version_304/checkpoints/epoch=9.ckpt"
    # delete temporary folder to save summaries
    delete_temps = False
    # batch size
    batch_size = 4
    # model's optimizer learning rate
    learning_rate = 1
    # max document's sentence length
    max_doc_len = 15
    # max sentence's word length
    max_sen_len = 50
    # maximum gradient clip norm
    grad_clip_val = 5
    # resume trainer from path
    resume_path = None

    word2vec_model = "./idwiki_word2vec_100.model"


@ex.command
def model_to_word2vec(word2vec_model):
    model = Word2Vec.load(word2vec_model)
    model.wv.save_word2vec_format("word2vec.txt")


@ex.command
def test(
    seed,
    model_path,
    embedding_path,
    batch_size,
    embedding_dim,
    lstm_hidden_size,
    attention_size,
    delete_temps,
    max_doc_len,
    max_sen_len,
    grad_clip_val,
    learning_rate,
):
    dm = IndosumDataModule(
        read_train_jsonl(), read_dev_jsonl(), read_test_jsonl(), embedding_path, max_doc_len, max_sen_len, 4
    )
    hssas = HSSAS(
        dm.vocab,
        embedding_dim,
        lstm_hidden_size,
        attention_size,
        max_doc_len,
        list(read_dev_jsonl()),
        learning_rate,
    )
    summaries = (
        summary
        for x, _, doc_lens in dm.test_dataloader()
        for summary in hssas(x, doc_lens)
    )
    for summary in summaries:
        print(summary)
        return
    return


@ex.command
def evaluate(
    seed,
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
    pl.utilities.seed.seed_everything(seed)
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
        summary
        for x, _, doc_lens in data_module.test_dataloader()
        for summary in hssas(x, doc_lens)
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
    seed,
    embedding_dim,
    lstm_hidden_size,
    attention_size,
    embedding_path,
    batch_size,
    max_doc_len,
    max_sen_len,
    grad_clip_val,
    learning_rate,
    resume_path,
):
    pl.utilities.seed.seed_everything(seed)
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
        max_doc_len,
        list(read_dev_jsonl()),
        learning_rate,
    )

    checkpoint_callback = ModelCheckpoint(monitor="val_loss")
    trainer = pl.Trainer(
        gpus=1,
        # callbacks=[EarlyStopping(monitor="val_loss", mode="min", patience=3)],
        checkpoint_callback=checkpoint_callback,
        gradient_clip_val=grad_clip_val,
        resume_from_checkpoint=resume_path,
        max_epochs=5000,
        limit_train_batches=.01,
        limit_val_batches=.01
    )
    trainer.fit(hssas, dm)
    evaluate(model_path=checkpoint_callback.best_model_path, data_module=dm)
