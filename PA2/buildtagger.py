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
        self.conv1 = nn.Conv1d(hidden_size, l, kernel_size=k)
        self.pool = nn.MaxPool1d(kernel_size=k)

    def forward(self, input):
        # TODO: Verify that the transformation is correct
        # TODO: Consider adding dropout layer here
        output_batches = self.embedding(input)
        output = torch.empty(0)
        for batch in output_batches:
            output = torch.cat((output, torch.transpose(batch, 0, 1).unsqueeze(0)))
        output = self.conv1(output)
        output = self.pool(output)
        return output


class BiLSTM(nn.Module):
    def __init__(self, hidden_size, vocab_size):
        super(BiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, self.hidden_size)

    def forward(self, input, input_char_embeddings):
        # TODO: Will input be a sentence here, so we can disssect and find the embeddings for
        # char as well?
        pass


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
    ix_to_char = {}
    char_to_ix = {}
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

    # TODO: Obtain character embeddings as well
    # TODO: Add word embeddings from pytorch
    # TODO: Create a character embeddings too
    # TODO: Randomize the weights for the embeddings

    # TODO: Include the unknown word as well
    for i, term in enumerate(term_count.keys()):
        word_to_ix[term] = i
        ix_to_word[i] = term
    for i, pos in enumerate(pos_count.keys()):
        pos_to_ix[pos] = i
        ix_to_pos[i] = pos
    for i in range(0, 128):
        ix_to_char[i] = chr(i)
        char_to_ix[chr(i)] = i

    char_cnn = CharCNN(hidden_size=2)
    sentence = ['hello', 'world']
    char_cnn(torch.tensor([
        [char_to_ix[character]
        for character in input_word] for input_word in sentence
    ], dtype=torch.long))

    print('Finished...')


if __name__ == "__main__":
    # make no changes here
    train_file = sys.argv[1]
    model_file = sys.argv[2]
    train_model(train_file, model_file)
