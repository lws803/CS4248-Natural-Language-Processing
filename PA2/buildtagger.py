# python3.5 buildtagger.py <train_file_absolute_path> <model_file_absolute_path>

import os
import math
import sys
from collections import defaultdict
import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
EPOCHS = 1  # See if we really need to run 2 epochs
WORD_EMBEDDINGS_SIZE = 50
CHAR_EMBEDDINGS_SIZE = 50
CHAR_CONV_FILTERS = 50
CHAR_CONV_WINDOW_SIZE = 5
LSTM_HIDDEN_SIZE = 256
WORD_CHAR_PADDING = 30
UNK_WORDS_THRESHOLD = 1


class CharCNN(nn.Module):
    def __init__(self):
        super(CharCNN, self).__init__()
        self.hidden_size = CHAR_EMBEDDINGS_SIZE
        self.embedding = nn.Embedding(128, self.hidden_size)
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv1d(
            self.hidden_size, CHAR_CONV_FILTERS, kernel_size=CHAR_CONV_WINDOW_SIZE,
            padding=CHAR_CONV_WINDOW_SIZE
        )
        self.pool = nn.AdaptiveMaxPool1d(1)

    def forward(self, input_chars_list):
        output = self.embedding(input_chars_list)
        output = torch.transpose(output, 2, 1)  # for list of words
        # output = torch.transpose(output, 0, 1)
        output = self.relu(self.conv1(output))
        output = self.pool(output)
        return output.squeeze(2)


class BiLSTM(nn.Module):
    def __init__(
        self, word_embed_size, lstm_hidden_size,
        vocab_size, tag_size, ix_to_word_chars
    ):
        super(BiLSTM, self).__init__()
        self.word_embed_size = word_embed_size
        self.ix_to_word_chars = ix_to_word_chars

        self.embedding = nn.Embedding(vocab_size, self.word_embed_size)
        self.char_cnn = CharCNN()
        self.bilstm = nn.LSTM(
            word_embed_size + CHAR_CONV_FILTERS, lstm_hidden_size, bidirectional=True
        )
        self.linear = nn.Linear(lstm_hidden_size * 2, tag_size)

    def forward(self, input_words):
        output = self.embedding(torch.tensor(input_words, dtype=torch.long, device=DEVICE))
        list_of_word_chars = torch.tensor(
            [self.ix_to_word_chars[idx] for idx in input_words], dtype=torch.long,
            device=DEVICE
        )
        char_embeddings = self.char_cnn(list_of_word_chars)
        output = torch.cat((output, char_embeddings), 1)
        hidden, _ = self.bilstm(output.unsqueeze(1))
        hidden_reformatted = hidden.view(len(input_words), -1)
        tag_space = self.linear(hidden_reformatted)
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores


def train_model(train_file, model_file):
    # use torch library to save model parameters, hyperparameters, etc. to model_file
    term_count = defaultdict(int)
    pos_count = defaultdict(int)
    word_to_ix = {}
    ix_to_word = {}
    pos_to_ix = {}
    ix_to_pos = {}
    ix_to_word_chars = {}
    sentences = []
    sentence_tags = []

    with torch.no_grad():
        # Tokenizer
        with open(train_file) as f:
            lines = f.readlines()
            for line in lines:
                tags = []
                words = []
                for term_pos_pairs in line.split():
                    term_pos_pairs = term_pos_pairs.split('/')
                    pos = term_pos_pairs[-1]
                    term_pos_pairs.pop()
                    term = '/'.join(term_pos_pairs)

                    term_count[term] += 1
                    pos_count[pos] += 1

        for i, term in enumerate(term_count.keys()):
            word_to_ix[term] = i
            ix_to_word[i] = term
            word_chars = [ord(character) for character in term]
            # Padded words
            ix_to_word_chars[i] = word_chars[0:WORD_CHAR_PADDING] + [
                0 for i in range(WORD_CHAR_PADDING - len(word_chars))
            ]
        for i, pos in enumerate(pos_count.keys()):
            pos_to_ix[pos] = i
            ix_to_pos[i] = pos

        # Add unknown words
        ix_to_word_chars[len(ix_to_word_chars)] = [0 for i in range(30)]
        word_to_ix['<UNK>'] = len(word_to_ix)

        with open(train_file) as f:
            lines = f.readlines()
            for line in lines:
                tags = []
                words = []
                for term_pos_pairs in line.split():
                    term_pos_pairs = term_pos_pairs.split('/')
                    pos = term_pos_pairs[-1]
                    term_pos_pairs.pop()
                    term = '/'.join(term_pos_pairs)
                    if term_count[term] <= UNK_WORDS_THRESHOLD:
                        term = '<UNK>'
                    words.append(term)
                    tags.append(pos)
                sentences.append(words)
                sentence_tags.append(tags)

    bilstm = BiLSTM(
        WORD_EMBEDDINGS_SIZE, LSTM_HIDDEN_SIZE,
        len(word_to_ix), len(pos_to_ix), ix_to_word_chars
    )
    bilstm.to(DEVICE)

    loss_function = nn.NLLLoss()
    optimizer = optim.Adam(bilstm.parameters())
    for i in range(EPOCHS):
        epoch_loss = None
        for index, (words, tags) in enumerate(zip(sentences, sentence_tags)):
            bilstm.zero_grad()

            tag_scores = bilstm([word_to_ix[word] for word in words])
            loss = loss_function(tag_scores, torch.tensor(
                [pos_to_ix[tag] for tag in tags], device=DEVICE
            ))
            loss.backward()
            optimizer.step()
            epoch_loss = loss
        print(epoch_loss)
    torch.save(
        {
            'state_dict': bilstm.state_dict(),
            'word_to_ix': word_to_ix,
            'ix_to_word_chars': ix_to_word_chars,
            'ix_to_pos': ix_to_pos
        }, model_file
    )
    print('Finished...')


if __name__ == "__main__":
    # make no changes here
    train_file = sys.argv[1]
    model_file = sys.argv[2]
    start_time = datetime.datetime.now()
    train_model(train_file, model_file)
    end_time = datetime.datetime.now()
    print('Time:', end_time - start_time)
