import pytorch_lightning as pl
from torch import nn
import torch

class WordEncoder(nn.Module):
    def __init__(self, hidden_size, embedding_dim):
        super(WordEncoder, self).__init__()

        self.lstm = nn.LSTM(embedding_dim, hidden_size, bidirectional=True)

    def forward(self, word_embeddings):
        word_embeddings.unsqueeze_(0)
        print(word_embeddings.shape)
        y, _ = self.lstm(word_embeddings)
        return y


class HSSAS(pl.LightningModule):
    def __init__(self, vocab, embedding_dim, lstm_hidden_size):
        super(HSSAS, self).__init__()

        self.vocab = vocab
        self.embedding = nn.Embedding(len(vocab), embedding_dim).from_pretrained(vocab.vectors)
        self.word_encoder = WordEncoder(lstm_hidden_size, embedding_dim)

    def forward(self, sentences):
        sentence_embeddings = [
            self.embedding(torch.LongTensor([self.vocab.stoi[word] for word in sentence]))
            for sentence in sentences
        ]
        for word_embeddings in sentence_embeddings:
            self.word_encoder(word_embeddings)
            return 0