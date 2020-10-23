# python3.5 buildtagger.py <train_file_absolute_path> <model_file_absolute_path>

import os
import math
import sys
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class CharCNN(nn.Module):
    def __init__(self, hidden_size, l=3, k=3):
        super(CharCNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(128, self.hidden_size)
        self.conv1 = nn.Conv1d(hidden_size, l, kernel_size=k, padding=k)
        self.pool = nn.AdaptiveMaxPool1d(1)

    def forward(self, input_chars):
        # TODO: Consider adding dropout layer here
        output = self.embedding(input_chars)
        # output = torch.transpose(output_batches, 2, 1)  # for list of words
        output = torch.transpose(output, 0, 1).unsqueeze(0)
        output = self.conv1(output)
        output = self.pool(output)
        return torch.flatten(output)


class BiLSTM(nn.Module):
    def __init__(self, word_embed_size, vocab_size, ix_to_word_chars):
        super(BiLSTM, self).__init__()
        self.word_embed_size = word_embed_size
        self.ix_to_word_chars = ix_to_word_chars

        self.embedding = nn.Embedding(vocab_size, self.word_embed_size)
        self.char_cnn = CharCNN(word_embed_size)

    def forward(self, input_words):
        output = self.embedding(torch.tensor(input_words, dtype=torch.long))
        char_embeddings = torch.empty((0, self.word_embed_size))
        # TODO: This might be expensive operaiton
        for word_ix in input_words:
            chars = self.ix_to_word_chars[word_ix]
            char_embedding = self.char_cnn(chars)  # Merge this together with the word embeddingss
            char_embeddings = torch.cat((char_embeddings, char_embedding.unsqueeze(0)))
        output = torch.cat((output, char_embeddings), 1)

        return output


def train_model(train_file, model_file):
    # write your code here. You can add functions as well.
    # use torch library to save model parameters, hyperparameters, etc. to model_file
    # TODO: follow the seq2seq tutorial from pytorch
    # TODO: run k-fold cross validation?

    term_count = defaultdict(int)
    pos_count = defaultdict(int)
    word_to_ix = {}
    ix_to_word = {}
    pos_to_ix = {}
    ix_to_pos = {}
    ix_to_word_chars = {}
    # Tokenizer
    with open(train_file) as f:
        lines = f.readlines()
        for line in lines:
            for term_pos_pairs in line.split():
                term_pos_pairs = term_pos_pairs.split('/')
                pos = term_pos_pairs[-1]
                term_pos_pairs.pop()
                term = '/'.join(term_pos_pairs)

                term_count[term] += 1
                pos_count[pos] += 1

    # TODO: Include the unknown word as well
    for i, term in enumerate(term_count.keys()):
        word_to_ix[term] = i
        ix_to_word[i] = term
        ix_to_word_chars[i] = torch.tensor(
            [ord(character) for character in term], dtype=torch.long
        )
    for i, pos in enumerate(pos_count.keys()):
        pos_to_ix[pos] = i
        ix_to_pos[i] = pos

    # TODO: Training, remember to split the training set first
    # char_cnn = CharCNN(hidden_size=2)
    bilstm = BiLSTM(3, len(word_to_ix), ix_to_word_chars)

    with open(train_file) as f:
        lines = f.readlines()
        for line in lines:
            words = []
            tags = []
            for term_pos_pairs in line.split():
                term_pos_pairs = term_pos_pairs.split('/')
                pos = term_pos_pairs[-1]
                term_pos_pairs.pop()
                term = '/'.join(term_pos_pairs)
                words.append(term)
                tags.append(pos)
            # char_indexes = torch.tensor([
            #     [ord(character)
            #     for character in input_word] for input_word in words
            # ], dtype=torch.long)
            # FIXME: How do we handle cases when words are smaller than window size
            # char_cnn(torch.tensor(ix_to_word_chars[word_to_ix[words[0]]]))
            bilstm([word_to_ix[word] for word in words])
            break

    print('Finished...')


if __name__ == "__main__":
    # make no changes here
    train_file = sys.argv[1]
    model_file = sys.argv[2]
    train_model(train_file, model_file)
