import logging
import os
import shutil
import tempfile
from pythonrouge.pythonrouge import Pythonrouge
from dotenv import load_dotenv

load_dotenv()

SAVE_FILES = os.getenv("SACRED_SAVE_FILES", "false").lower() == "true"


def eval_summaries(
    summaries, docs, logger=None, topk=3, encoding="utf-8", delete_temps=True
):
    if logger is None:
        logger = logging.getLogger(__name__)

    abstract_references = []
    extract_references = []
    hypotheses = []
    for i, (summary, doc) in enumerate(zip(summaries, docs)):
        logger.info(f"Generating summary for doc {i+1}")
        topk = min(topk, len(summary))
        abs_refs = [[" ".join(sent) for sent in doc.summary][:topk]]
        ext_refs = [[" ".join(sent.words) for sent in doc.sentences if sent.label][:topk]]
        hyp = [" ".join(doc.sentences[idx].words) for idx in summary.topk(topk)[1]]
        abstract_references.append(abs_refs)
        extract_references.append(ext_refs)
        hypotheses.append(hyp)

    assert len(abstract_references) == len(
        hypotheses
    ), "Number of abstractive references and hypotheses mismatch"
    assert len(extract_references) == len(
        hypotheses
    ), "Number of extractive references and hypotheses mismatch"

    abs_ref_dirname = tempfile.mkdtemp()
    logger.info("Abstractive references directory: %s", abs_ref_dirname)
    ext_ref_dirname = tempfile.mkdtemp()
    logger.info("Extractive references directory: %s", ext_ref_dirname)
    hyp_dirname = tempfile.mkdtemp()
    logger.info("Hypotheses directory: %s", hyp_dirname)
    for doc_id, (abs_refs, ext_refs, hyp) in enumerate(
        zip(abstract_references, extract_references, hypotheses)
    ):
        # Write references
        for rid, ref in enumerate(abs_refs):
            ref_filename = os.path.join(abs_ref_dirname, f"{doc_id}.{rid}.txt")
            write_to_file(ref_filename, encoding, ref)
        for rid, ref in enumerate(ext_refs):
            ref_filename = os.path.join(ext_ref_dirname, f"{doc_id}.{rid}.txt")
            write_to_file(ref_filename, encoding, ref)

        # Write hypothesis
        hyp_filename = os.path.join(hyp_dirname, f"{doc_id}.txt")
        write_to_file(hyp_filename, encoding, hyp)

    abs_rouge = Pythonrouge(
        peer_path=hyp_dirname,
        model_path=abs_ref_dirname,
        stemming=False,
        ROUGE_L=True,
        ROUGE_SU4=False,
    )
    abs_score = abs_rouge.calc_score()
    logger.info("Abstractive ROUGE scores: %s", abs_score)

    ext_rouge = Pythonrouge(
        peer_path=hyp_dirname,
        model_path=ext_ref_dirname,
        stemming=False,
        ROUGE_L=True,
        ROUGE_SU4=False,
    )
    ext_score = ext_rouge.calc_score()
    logger.info("Extractive ROUGE scores: %s", ext_score)

    if delete_temps:
        logger.info("Deleting temporary files and directories")
        shutil.rmtree(abs_ref_dirname)
        shutil.rmtree(ext_ref_dirname)
        shutil.rmtree(hyp_dirname)

    return abs_score, ext_score


def write_to_file(filename, encoding, data):
    with open(filename, "w", encoding=encoding) as f:
        print("\n".join(data), file=f)
