import pytorch_lightning as pl
from torch import nn
import torch
import itertools
import math


class SentenceEncoder(nn.Module):
    def __init__(self, hidden_size):
        super(SentenceEncoder, self).__init__()
        self.lstm = nn.LSTM(
            2 * hidden_size, hidden_size, batch_first=True, bidirectional=True
        )

    def forward(self, sentence_vectors):
        y, _ = self.lstm(sentence_vectors)
        return y


class Attention(nn.Module):
    def __init__(self, attention_size, lstm_hidden_size):
        super(Attention, self).__init__()

        self.w1 = nn.Parameter(torch.randn(2 * lstm_hidden_size, attention_size))
        self.w2 = nn.Parameter(torch.randn(attention_size))

    def forward(self, hidden_states):
        a = torch.matmul(hidden_states, self.w1)
        a = torch.tanh(a)
        a = torch.matmul(a, self.w2)
        a = nn.functional.softmax(a, dim=0)
        return a


class WordEncoder(nn.Module):
    def __init__(self, hidden_size, embedding_dim):
        super(WordEncoder, self).__init__()
        self.lstm = nn.LSTM(
            embedding_dim, hidden_size, batch_first=True, bidirectional=True
        )

    def forward(self, word_embeddings):
        y, _ = self.lstm(word_embeddings)
        return y


class HSSAS(pl.LightningModule):
    def __init__(self, vocab, embedding_dim, lstm_hidden_size, attention_size):
        super(HSSAS, self).__init__()

        self.save_hyperparameters()

        self.embedding = nn.Embedding(
            len(vocab), embedding_dim, padding_idx=1
        ).from_pretrained(vocab.vectors)
        self.word_encoder = WordEncoder(lstm_hidden_size, embedding_dim)
        self.word_attention = Attention(attention_size, lstm_hidden_size)
        self.sentence_encoder = SentenceEncoder(lstm_hidden_size)
        self.sentence_attention = Attention(attention_size, lstm_hidden_size)
        self.content = nn.Linear(2 * lstm_hidden_size, 1, bias=False)
        self.salience = nn.Bilinear(
            2 * lstm_hidden_size, 2 * lstm_hidden_size, 1, bias=False
        )
        self.novelty = nn.Bilinear(
            2 * lstm_hidden_size, 2 * lstm_hidden_size, 1, bias=False
        )
        self.position = nn.Linear(2 * lstm_hidden_size, 1, bias=False)
        self.bias = nn.Parameter(torch.FloatTensor(1).uniform_(-0.1, 0.1))

    def forward(self, sentences, log=False):
        sentence_embeddings = self.embedding(sentences)

        doc_len, sent_len, word_len, embedding_dim = sentence_embeddings.shape
        word_encoder_hidden_states = self.word_encoder(
            sentence_embeddings.view(doc_len * sent_len, word_len, embedding_dim)
        )
        word_attention_weights = self.word_attention(word_encoder_hidden_states)

        sentence_vectors = torch.squeeze(
            torch.matmul(
                torch.unsqueeze(word_attention_weights, 1), word_encoder_hidden_states
            )
        )
        sentence_vectors = sentence_vectors.view(doc_len, sent_len, -1)

        sentence_encoder_hidden_states = self.sentence_encoder(sentence_vectors)
        sentence_attention_weights = self.sentence_attention(
            sentence_encoder_hidden_states
        )

        document_vector = torch.squeeze(
            torch.matmul(
                torch.unsqueeze(sentence_attention_weights, 1),
                sentence_encoder_hidden_states,
            )
        )

        o = torch.zeros(doc_len, 2 * self.hparams.lstm_hidden_size, device=self.device)
        probs = [[] for _ in range(doc_len)]
        for pos in range(sent_len):
            sentence_vector = sentence_vectors[:, pos, :]

            C = self.content(sentence_vector)
            M = self.salience(sentence_vector, document_vector)
            N = self.novelty(sentence_vector, torch.tanh(o))

            positional_embedding = torch.tensor(
                list(
                    itertools.chain(
                        *[
                            [
                                math.sin(
                                    pos
                                    / (
                                        10000
                                        ** (2 * i / (2 * self.hparams.lstm_hidden_size))
                                    )
                                ),
                                math.cos(
                                    pos
                                    / (
                                        10000
                                        ** (2 * i / (2 * self.hparams.lstm_hidden_size))
                                    )
                                ),
                            ]
                            for i in range(self.hparams.lstm_hidden_size)
                        ]
                    )
                ),
                device=self.device,
            ).repeat(doc_len, 1)

            P = self.position(positional_embedding)

            batch_prob = C + M - N + P + self.bias
            for i, prob in enumerate(batch_prob):
                probs[i].append(prob)

            o = o + (batch_prob * sentence_vector)

        labels = []
        for prob in probs:
            labels.append(torch.cat(prob).view(1, -1))

        return torch.cat(labels)

    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x, log=batch_idx == 0)

        loss = nn.functional.binary_cross_entropy_with_logits(pred, y)

        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        loss = nn.functional.binary_cross_entropy_with_logits(pred, y)

        self.log("val_loss", loss.item(), prog_bar=True)
        return loss

    def test_step(self, batch, _):
        x, y = batch
        pred = self(x)
        loss = nn.functional.binary_cross_entropy_with_logits(pred, y)

        self.log("test_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
