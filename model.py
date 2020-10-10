import pytorch_lightning as pl
from torch import nn
import torch

class SentenceEncoder(nn.Module):
    def __init__(self, hidden_size):
        super(SentenceEncoder, self).__init__()
        self.lstm = nn.LSTM(2 * hidden_size, hidden_size, bidirectional=True)

    def forward(self, sentence_vectors):
        sentence_vectors.unsqueeze_(0)
        y, _ = self.lstm(sentence_vectors)
        return y

class Attention(nn.Module):
    def __init__(self, attention_size, lstm_hidden_size):
        super(Attention, self).__init__()

        self.w1 = nn.Parameter(torch.randn(attention_size, 2 * lstm_hidden_size))
        self.w2 = nn.Parameter(torch.randn(1, attention_size))

    def forward(self, hidden_states):
        a = torch.mm(self.w1, torch.t(hidden_states))
        a = torch.tanh(a)
        a = torch.mm(self.w2, a)
        a = nn.functional.softmax(a, dim=0)
        return a

class WordEncoder(nn.Module):
    def __init__(self, hidden_size, embedding_dim):
        super(WordEncoder, self).__init__()
        self.lstm = nn.LSTM(embedding_dim, hidden_size, bidirectional=True)

    def forward(self, word_embeddings):
        word_embeddings.unsqueeze_(0)
        y, _ = self.lstm(word_embeddings)
        return y


class HSSAS(pl.LightningModule):
    def __init__(self, vocab, embedding_dim, lstm_hidden_size, attention_size):
        super(HSSAS, self).__init__()

        self.vocab = vocab
        self.embedding = nn.Embedding(len(vocab), embedding_dim).from_pretrained(vocab.vectors)
        self.word_encoder = WordEncoder(lstm_hidden_size, embedding_dim)
        self.word_attention = Attention(attention_size, lstm_hidden_size)
        self.sentence_encoder = SentenceEncoder(lstm_hidden_size)
        self.sentence_attention = Attention(attention_size, lstm_hidden_size)

    def forward(self, sentences):
        sentence_embeddings = [
            self.embedding(torch.LongTensor([self.vocab.stoi[word] for word in sentence]))
            for sentence in sentences
        ]
        sentence_vectors = []
        for i, word_embeddings in enumerate(sentence_embeddings):
            word_encoder_hidden_states = self.word_encoder(word_embeddings)
            word_encoder_hidden_states.squeeze_(0)

            word_attention_weights = self.word_attention(word_encoder_hidden_states)

            sentence_vector = torch.mm(word_attention_weights, word_encoder_hidden_states)
            sentence_vectors.append(sentence_vector)
        sentence_vectors = torch.cat(sentence_vectors, 0)
        
        sentence_encoder_hidden_states = self.sentence_encoder(sentence_vectors)
        sentence_encoder_hidden_states.squeeze_(0)

        sentence_attention_weights = self.sentence_attention(sentence_encoder_hidden_states)
        
        document_vector = torch.mm(sentence_attention_weights, sentence_encoder_hidden_states)
        print(document_vector.shape) 
        return 0