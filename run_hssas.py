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
    lstm_hidden_size = 150
    # attention size
    attention_size = 300
    # saved model path
    model_path = "./lightning_logs/version_391/checkpoints/epoch=6.ckpt"
    # delete temporary folder to save summaries
    delete_temps = False
    # batch size
    batch_size = 12
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
    _log,
):
    pl.utilities.seed.seed_everything(seed)
    hssas = HSSAS.load_from_checkpoint(model_path)
    dm = IndosumDataModule(
        read_train_jsonl(), read_dev_jsonl(), read_test_jsonl(), embedding_path, max_doc_len, max_sen_len, 4
    )
 
    summaries = (
        summary
        for x, _, doc_lens, batch_sent_lens in dm.test_dataloader()
        for summary in hssas(x, doc_lens, batch_sent_lens)
    )

    for s in summaries:
        return
    
    eval_summaries(
        summaries, 
        list(read_test_jsonl())[1509:1511], 
        logger=_log, 
        delete_temps=delete_temps
    )

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
    # docs = (doc for doc in docs if any([True for sent in doc.sentences if sent.label and len(sent.words) >= 70]))
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
        for x, y, doc_lens, batch_sent_lens in data_module.test_dataloader()
        for summary in hssas(x, doc_lens, batch_sent_lens)
    )

    score = eval_summaries(
        summaries, 
        (d for d in docs if 1 not in [1 if sent.label else 0 for sent in d.sentences[:0]]), 
        logger=_log, 
        delete_temps=delete_temps
    )
    for name, value in score.items():
        _run.log_scalar(name, value)
    return score["ROUGE-1-F"]


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
        callbacks=[EarlyStopping(monitor="val_loss", mode="min", patience=3)],
        checkpoint_callback=checkpoint_callback,
        gradient_clip_val=grad_clip_val,
        resume_from_checkpoint=resume_path,
        max_epochs=5000,
    )
    torch.autograd.set_detect_anomaly(True)
    trainer.fit(hssas, dm)
    evaluate(model_path=checkpoint_callback.best_model_path, data_module=dm)
