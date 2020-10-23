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
        self.pool = nn.MaxPool1d(kernel_size=k)

    def forward(self, input):
        # TODO: Consider adding dropout layer here
        output = self.embedding(input)
        # output = torch.transpose(output_batches, 2, 1)  # for list of words
        output = torch.transpose(output, 0, 1).unsqueeze(0)
        output = self.conv1(output)
        output = self.pool(output)
        return output


class BiLSTM(nn.Module):
    def __init__(self, hidden_size, vocab_size, ix_to_word_chars):
        super(BiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.ix_to_word_chars = ix_to_word_chars

        self.embedding = nn.Embedding(vocab_size, self.hidden_size)
        self.char_cnn = CharCNN(10)

    def forward(self, input, chars_input):
        # TODO: Concatenate the character embeddings with the word embeddings
        # TODO: Will input be a sentence here, so we can disssect and find the embeddings for
        # char as well?
        # We might have to put a for loop in here and loop thru word by word to generate the
        # embedding matrix for it and tag it to the cnn
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
        ix_to_word_chars[i] = [ord(character) for character in term]
    for i, pos in enumerate(pos_count.keys()):
        pos_to_ix[pos] = i
        ix_to_pos[i] = pos

    # TODO: Training, remember to split the training set first
    char_cnn = CharCNN(hidden_size=2)
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
            char_cnn(torch.tensor(ix_to_word_chars[word_to_ix[words[0]]]))
            break

    print('Finished...')


if __name__ == "__main__":
    # make no changes here
    train_file = sys.argv[1]
    model_file = sys.argv[2]
    train_model(train_file, model_file)
