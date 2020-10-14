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

    references = []
    hypotheses = []
    for i, (summary, doc) in enumerate(zip(summaries, docs)):
        logger.info(f"Generating summary for doc {i+1}")
        topk = min(topk, len(summary))
        refs = [[" ".join(sent) for sent in doc.summary]]
        hyp = [" ".join(doc.sentences[idx].words) for idx in summary.topk(topk)[1]]
        references.append(refs)
        hypotheses.append(hyp)

    assert len(references) == len(
        hypotheses
    ), "Number of references and hypotheses mismatch"

    ref_dirname = tempfile.mkdtemp()
    logger.info("References directory: %s", ref_dirname)
    hyp_dirname = tempfile.mkdtemp()
    logger.info("Hypotheses directory: %s", hyp_dirname)
    for doc_id, (refs, hyp) in enumerate(zip(references, hypotheses)):
        # Write references
        for rid, ref in enumerate(refs):
            ref_filename = os.path.join(ref_dirname, f"{doc_id}.{rid}.txt")
            with open(ref_filename, "w", encoding=encoding) as f:
                print("\n".join(ref), file=f)
        # Write hypothesis
        hyp_filename = os.path.join(hyp_dirname, f"{doc_id}.txt")
        with open(hyp_filename, "w", encoding=encoding) as f:
            print("\n".join(hyp), file=f)

    rouge = Pythonrouge(
        peer_path=hyp_dirname,
        model_path=ref_dirname,
        stemming=False,
        ROUGE_L=True,
        ROUGE_SU4=False,
    )
    score = rouge.calc_score()
    logger.info("ROUGE scores: %s", score)

    if delete_temps:
        logger.info("Deleting temporary files and directories")
        shutil.rmtree(ref_dirname)
        shutil.rmtree(hyp_dirname)

    return score
