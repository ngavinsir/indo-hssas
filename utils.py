import logging
import os
import shutil
import tempfile
from pythonrouge.pythonrouge import Pythonrouge
from dotenv import load_dotenv
from sacred.observers import MongoObserver
import torch
import numpy as np

load_dotenv()

SAVE_FILES = os.getenv("SACRED_SAVE_FILES", "false").lower() == "true"


def setup_mongo_observer(ex):
    mongo_url = os.getenv("SACRED_MONGO_URL")
    db_name = os.getenv("SACRED_DB_NAME")
    if mongo_url is not None and db_name is not None:
        ex.observers.append(MongoObserver.create(url=mongo_url, db_name=db_name))

def extract_preds(outputs):
    for pred, doc_lens in outputs:
        start = 0
        for doc_len in doc_lens:
            end = start + doc_len
            yield pred[start:end]
            start += doc_len

def eval_summaries(
    summaries, docs, logger=None, topk=3, encoding="utf-8", delete_temps=True, log=True
):
    if logger is None:
        logger = logging.getLogger(__name__)

    references = []
    hypotheses = []
    for i, (summary, doc) in enumerate(zip(summaries, docs)):
        # if doc.id_ != "1503203400-mainan-terlarang-bagi-pangeran-george-dan-putri-ch":
        #      continue
        if log:
            ext_index = [i for i, sent in enumerate(doc.sentences) if sent.label][:topk]
            hyp_index = [i.item() for i in torch.sort(summary[:len(doc.sentences)].topk(topk)[1])[0]]
            logger.info(f"Generating summary for doc {i} {ext_index} {hyp_index}")
            if np.in1d(hyp_index, ext_index).sum() >= 3:
                logger.info(f"Bagus: {i}")
        topk = min(topk, len(summary))
        refs = [[" ".join(sent) for sent in doc.summary]]
        hyp = [
            " ".join(doc.sentences[idx].words)
            for idx in torch.sort(summary[:len(doc.sentences)].topk(topk)[1])[0]
        ]
        references.append(refs)
        hypotheses.append(hyp)

    assert len(references) == len(
        hypotheses
    ), "Number of references and hypotheses mismatch"

    ref_dirname = tempfile.mkdtemp()
    hyp_dirname = tempfile.mkdtemp()

    if log:
        logger.info("References directory: %s", ref_dirname)
        logger.info("Hypotheses directory: %s", hyp_dirname)

    for doc_id, (refs, hyp) in enumerate(zip(references, hypotheses)):
        # Write references
        for rid, ref in enumerate(refs):
            ref_filename = os.path.join(ref_dirname, f"{doc_id}.{rid}.txt")
            write_to_file(ref_filename, encoding, ref)

        # Write hypothesis
        hyp_filename = os.path.join(hyp_dirname, f"{doc_id}.txt")
        write_to_file(hyp_filename, encoding, hyp)

    rouge = Pythonrouge(
        peer_path=hyp_dirname,
        model_path=ref_dirname,
        stemming=False,
        ROUGE_L=True,
        ROUGE_SU4=False,
    )
    score = rouge.calc_score()

    if log:
        logger.info("ROUGE scores: %s", score)

    if delete_temps:
        if log:
            logger.info("Deleting temporary files and directories")
        shutil.rmtree(ref_dirname)
        shutil.rmtree(hyp_dirname)

    return score


def write_to_file(filename, encoding, data):
    with open(filename, "w", encoding=encoding) as f:
        print("\n".join(data), file=f)
