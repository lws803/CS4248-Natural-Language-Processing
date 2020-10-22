# python3.5 buildtagger.py <train_file_absolute_path> <model_file_absolute_path>

import os
import math
import sys
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def train_model(train_file, model_file):
    # write your code here. You can add functions as well.
    # use torch library to save model parameters, hyperparameters, etc. to model_file
    # TODO: follow the seq2seq tutorial from pytorch

    term_count = defaultdict(int)
    pos_count = defaultdict(int)
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

    # TODO: Add word embeddings from pytorch
    # TODO: Create a character embeddings too
    # TODO: Randomize the weights for the embeddings
    print('Finished...')
    
if __name__ == "__main__":
    # make no changes here
    train_file = sys.argv[1]
    model_file = sys.argv[2]
    train_model(train_file, model_file)
