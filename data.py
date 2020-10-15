##########################################################################
# Copyright 2018 Kata.ai
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##########################################################################

import torch
from typing import Any, Callable, Container, List, Mapping, Optional, Sequence, Union
import itertools
import re
import string
import pytorch_lightning as pl
import logging
from torchtext.vocab import Vocab, Vectors
from collections import Counter
from torch.utils.data import DataLoader


Word = str
Label = Union[bool, float]


class Sentence(Sequence[Word]):
    DELIMITER = "\t\t\t"

    def __init__(
        self,
        words: Sequence[Word],
        gold_label: Optional[Label] = None,
        pred_label: Optional[Label] = None,
    ) -> None:
        self.words = words
        self.gold_label = gold_label
        self.pred_label = pred_label

    @property
    def label(self) -> Optional[Label]:
        return self.gold_label

    @label.setter
    def label(self, label):
        self.gold_label = label

    def __getitem__(self, key):
        return self.words[key]

    def __len__(self) -> int:
        return len(self.words)

    def __str__(self) -> str:
        words_str = " ".join(self.words)
        if self.label is None:
            return words_str

        if isinstance(self.label, bool):
            label = 1 if self.label else 0
        else:
            label = self.label
        return "{}{}{}".format(words_str, self.DELIMITER, label)

    def filter_words(self, predicate: Callable[[Word], bool]) -> "Sentence":
        return Sentence(
            list(filter(predicate, self.words)),
            gold_label=self.gold_label,
            pred_label=self.pred_label,
        )

    def map_words(self, func: Callable[[Word], Word]) -> "Sentence":
        return Sentence(
            list(map(func, self.words)),
            gold_label=self.gold_label,
            pred_label=self.pred_label,
        )


class Paragraph(Sequence[Sentence]):
    def __init__(self, sentences: Sequence[Sentence]) -> None:
        self.sentences = sentences

    @property
    def words(self) -> List[Word]:
        return [word for sent in self.sentences for word in sent]

    @property
    def labels(self) -> List[Optional[Label]]:
        return [sent.label for sent in self.sentences]

    @property
    def pred_labels(self) -> List[Optional[Label]]:
        return [sent.pred_label for sent in self.sentences]

    def __getitem__(self, key):
        return self.sentences[key]

    def __len__(self) -> int:
        return len(self.sentences)

    def __str__(self) -> str:
        return "\n".join(str(sent) for sent in self.sentences)

    def filter_words(self, predicate: Callable[[Word], bool]) -> "Paragraph":
        filtered_sents = [sent.filter_words(predicate) for sent in self.sentences]
        return Paragraph([sent for sent in filtered_sents if sent])

    def map_words(self, func: Callable[[Word], Word]) -> "Paragraph":
        return Paragraph([sent.map_words(func) for sent in self.sentences])

    @classmethod
    def from_sequence(
        cls,
        sents: Sequence[Sequence[Word]],
        gold_labels: Optional[Sequence[Label]] = None,
        pred_labels: Optional[Sequence[Label]] = None,
    ) -> "Paragraph":
        gold_labels_ = itertools.repeat(None) if gold_labels is None else gold_labels
        pred_labels_ = itertools.repeat(None) if pred_labels is None else pred_labels
        sentences = []
        for sent, gold, pred in zip(sents, gold_labels_, pred_labels_):  # type: ignore
            sentences.append(Sentence(sent, gold_label=gold, pred_label=pred))
        return cls(sentences)


class Document(Sequence[Paragraph]):
    def __init__(
        self,
        paragraphs: Sequence[Paragraph],
        summary: Optional[Paragraph] = None,
        category: Optional[str] = None,
        source: Optional[str] = None,
        source_url: Optional[str] = None,
        id_: Optional[str] = None,
        lower: bool = False,
        remove_puncts: bool = False,
        replace_digits: bool = False,
        stopwords: Optional[Container[Word]] = None,
    ) -> None:
        self.paragraphs = paragraphs
        self.preprocessed_paragraphs = paragraphs
        self.summary = summary
        self.category = category
        self.source = source
        self.source_url = source_url
        self.id_ = id_
        self.lower = lower
        self.remove_puncts = remove_puncts
        self.replace_digits = replace_digits
        self.stopwords = stopwords

        self.preprocess()

    def preprocess(self) -> None:
        if self.lower:
            self.preprocessed_paragraphs = [
                para.map_words(lambda w: w.lower()) for para in self.preprocessed_paragraphs
            ]
        if self.remove_puncts:
            self.preprocessed_paragraphs = [
                para.filter_words(lambda w: w not in string.punctuation)
                for para in self.preprocessed_paragraphs
            ]
        if self.replace_digits:
            self.preprocessed_paragraphs = [
                para.map_words(lambda w: re.sub(r"\d", "0", w))
                for para in self.preprocessed_paragraphs
            ]
        if self.stopwords is not None:
            self.preprocessed_paragraphs = [
                para.filter_words(lambda w: w not in self.stopwords)  # type: ignore
                for para in self.preprocessed_paragraphs
            ]

    @property
    def words(self) -> List[Word]:
        return list(itertools.chain(*[para.words for para in self.paragraphs]))

    @property
    def sentences(self) -> List[Sentence]:
        return list(itertools.chain(*self.paragraphs))

    @property
    def preprocessed_sentences(self) -> List[Sentence]:
        return list(itertools.chain(*self.preprocessed_paragraphs))

    def __getitem__(self, key):
        return self.paragraphs[key]

    def __len__(self) -> int:
        return len(self.paragraphs)

    def __str__(self) -> str:
        return "\n\n".join(str(para) for para in self.paragraphs)

    @classmethod
    def from_mapping(cls, obj: Mapping[str, Any], **kwargs) -> "Document":
        paragraphs = cls._get_paragraphs_from_mapping(obj)
        summary = obj.get("summary")
        if summary is not None:
            summary = Paragraph.from_sequence(summary)
        return cls(
            paragraphs,
            summary=summary,
            category=obj.get("category"),
            source=obj.get("source"),
            source_url=obj.get("source_url"),
            id_=obj.get("id"),
            **kwargs,
        )

    def to_dict(self) -> dict:
        paragraphs = [[sent.words for sent in para] for para in self.paragraphs]
        res: dict = {"paragraphs": paragraphs}

        if self.summary is not None:
            res["summary"] = [sent.words for sent in self.summary]

        for attr in "category source source_url".split():
            if getattr(self, attr) is not None:
                res[attr] = getattr(self, attr)

        if self.id_ is not None:
            res["id"] = self.id_

        gold_labels = [para.labels for para in self.paragraphs]
        if all(lab is not None for lab in itertools.chain.from_iterable(gold_labels)):
            res["gold_labels"] = gold_labels

        pred_labels = [para.pred_labels for para in self.paragraphs]
        if all(lab is not None for lab in itertools.chain.from_iterable(pred_labels)):
            res["pred_labels"] = pred_labels

        return res

    @staticmethod
    def _get_paragraphs_from_mapping(obj: Mapping[str, Any]) -> List[Paragraph]:
        gold_labels = obj.get("gold_labels", itertools.repeat(None))
        pred_labels = obj.get("pred_labels", itertools.repeat(None))
        paragraphs = [
            Paragraph.from_sequence(p, gold_labels=gl, pred_labels=pl)
            for p, gl, pl in zip(obj["paragraphs"], gold_labels, pred_labels)
        ]
        return paragraphs


class IndosumDataset(torch.utils.data.Dataset):
    def __init__(self, data_iter):
        self.data = list(map(self.doc_mapper, data_iter))

    def doc_mapper(self, doc):
        sentences = list(map(lambda sent: sent.words, doc.preprocessed_sentences))
        labels = torch.FloatTensor(
            list(map(lambda sent: 1 if sent.label else 0, doc.preprocessed_sentences))
        )
        
        return sentences, labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class IndosumDataModule(pl.LightningDataModule):
    def __init__(self, train_iter, dev_iter, test_iter, embedding_path, batch_size=8):
        super().__init__()
        self.train_data = IndosumDataset(train_iter)
        self.dev_data = IndosumDataset(dev_iter)
        self.test_data = IndosumDataset(test_iter)
        self.dl_kwargs = {
            "batch_size": batch_size,
            "collate_fn": self.collate,
            "num_workers": 8,
        }
        self._log = logging.getLogger(__name__)
        self.vocab = self.prepare_vocab(embedding_path)

    def prepare_vocab(self, embedding_path):
        self._log.info(f"preparing vocabulary...")
        counter = Counter()
        for sentences, _ in self.train_data:
            counter.update(list(itertools.chain(*sentences)))
        vectors = Vectors(embedding_path, cache="./")
        return Vocab(counter, vectors=vectors, min_freq=2)

    def collate(self, batch):
        x = []
        y = []

        max_sentence = 0
        max_word = 0
        for sentences, labels in batch:
            if max_sentence < len(sentences):
                max_sentence = len(sentences)
            for sentence in sentences:
                if max_word < len(sentence):
                    max_word = len(sentence)

        for sentences, labels in batch:
            results = []
            for sentence in sentences:
                sentence = list(map(lambda word: self.vocab.stoi[word], sentence))
                sentence = sentence + [
                    self.vocab.stoi["<pad>"] for _ in range(max_word - len(sentence))
                ]
                results.append(sentence)
            for _ in range(max_sentence - len(sentences)):
                results.append([self.vocab.stoi["<pad>"] for _ in range(max_word)])
            labels = torch.cat(
                (
                    labels,
                    torch.LongTensor([0 for _ in range(max_sentence - len(sentences))]),
                )
            )
            x.append(results)
            y.append(labels)

        return torch.LongTensor(x), torch.stack(y)

    def train_dataloader(self):
        return DataLoader(self.train_data, **self.dl_kwargs)

    def val_dataloader(self):
        return DataLoader(self.dev_data, **self.dl_kwargs)

    def test_dataloader(self):
        return DataLoader(self.test_data, **self.dl_kwargs)
