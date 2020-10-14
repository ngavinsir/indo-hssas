import pytorch_lightning as pl
from torch import nn
import torch
import itertools
import math


class SentenceEncoder(nn.Module):
    def __init__(self, hidden_size):
        super(SentenceEncoder, self).__init__()
        self.lstm = nn.LSTM(2 * hidden_size, hidden_size, bidirectional=True)

    def forward(self, sentence_vectors):
        sentence_vectors = sentence_vectors.unsqueeze(0)
        y, _ = self.lstm(sentence_vectors)
        return y


class Attention(nn.Module):
    def __init__(self, attention_size, lstm_hidden_size):
        super(Attention, self).__init__()

        self.w1 = nn.Parameter(torch.randn(attention_size, 2 * lstm_hidden_size))
        self.w2 = nn.Parameter(torch.randn(1, attention_size))

    def forward(self, hidden_states):
        hidden_states = hidden_states.squeeze(0)
        a = torch.mm(self.w1, torch.t(hidden_states))
        a = torch.tanh(a)
        a = torch.mm(self.w2, a)
        a = nn.functional.softmax(a, dim=0)
        return a


class WordEncoder(nn.Module):
    def __init__(self, hidden_size, embedding_dim):
        super(WordEncoder, self).__init__()
        self.lstm = nn.LSTM(
            embedding_dim, hidden_size, batch_first=True, bidirectional=True
        )

    def forward(self, word_embeddings):
        # word_embeddings = word_embeddings.unsqueeze(0)
        y, _ = self.lstm(word_embeddings)
        return y


class HSSAS(pl.LightningModule):
    def __init__(self, vocab, embedding_dim, lstm_hidden_size, attention_size):
        super(HSSAS, self).__init__()

        self.vocab = vocab
        self.embedding_dim = embedding_dim
        self.lstm_hidden_size = lstm_hidden_size
        self.attention_size = attention_size

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
        print(sentence_embeddings.shape)
        print(self.word_encoder(sentence_embeddings).shape)
        sentence_vectors = []
        for i, word_embeddings in enumerate(sentence_embeddings):
            word_encoder_hidden_states = self.word_encoder(word_embeddings)

            word_attention_weights = self.word_attention(word_encoder_hidden_states)

            sentence_vector = torch.mm(
                word_attention_weights, word_encoder_hidden_states[0]
            )
            sentence_vectors.append(sentence_vector)
        sentence_vectors = torch.cat(sentence_vectors, 0)

        sentence_encoder_hidden_states = self.sentence_encoder(sentence_vectors)
        sentence_attention_weights = self.sentence_attention(
            sentence_encoder_hidden_states
        )
        document_vector = torch.mm(
            sentence_attention_weights, sentence_encoder_hidden_states[0]
        )
        if log:
            print(sentence_attention_weights)

        o = torch.zeros(1, 2 * self.lstm_hidden_size, device=self.device)
        probs = []
        for pos, sentence_vector in enumerate(sentence_vectors):
            sentence_vector = sentence_vector.view(1, -1)

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
                                    / (10000 ** (2 * i / (2 * self.lstm_hidden_size)))
                                ),
                                math.cos(
                                    pos
                                    / (10000 ** (2 * i / (2 * self.lstm_hidden_size)))
                                ),
                            ]
                            for i in range(self.lstm_hidden_size)
                        ]
                    )
                ),
                device=self.device,
            ).view(1, -1)
            P = self.position(positional_embedding)
            # if log:
            #     print(C.item(), M.item(), N.item(), P.item())

            prob = torch.sigmoid(C + M - N + P + self.bias)
            probs.append(prob)

            o = o + torch.mm(prob, sentence_vector)

        return torch.cat(probs).squeeze()

    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x, log=batch_idx == 0)
        loss = nn.functional.binary_cross_entropy(pred, y)
        if batch_idx == 0:
            print(y)
            print(pred)
            print(loss)

        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        return 0
        x, y = batch
        pred = self(x)
        loss = nn.functional.binary_cross_entropy(pred, y)

        self.log("val_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch, _):
        x, y = batch
        pred = self(x)
        loss = nn.functional.binary_cross_entropy(pred, y)

        self.log("test_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
