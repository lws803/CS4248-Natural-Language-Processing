# python3.5 buildtagger.py <train_file_absolute_path> <model_file_absolute_path>

import os
import math
import sys
import datetime
from collections import defaultdict


def check_capitalization(term):
    # Capitizalied non-initial
    if term.isupper():
        return 2
    # Capitalized initial
    if term[0].isupper():
        return 1
    return 0


def train_model(train_file, model_file):
    pos_bigrams = defaultdict(int)
    word_pos_pair = defaultdict(int)
    pos_count = defaultdict(int)
    capitalized_pos_count = defaultdict(int)
    morphological_pos_count = defaultdict(int)

    with open(train_file) as f:
        for line in f.readlines():
            # Should we ignore the punctuations or account for them as well?
            prev_pos = '<s>'
            for term_pos_pairs in line.split():
                term_pos_pairs = term_pos_pairs.split('/')
                pos = term_pos_pairs[-1]
                term_pos_pairs.pop()
                term = '/'.join(term_pos_pairs)

                pos_bigrams[(prev_pos, pos)] += 1
                pos_count[pos] += 1
                word_pos_pair[(pos, term)] += 1

                capitalized_pos_count[(check_capitalization(term), pos)] += 1

                prev_pos = pos
        pos_bigrams[(prev_pos, '</s>')] += 1
    print('Finished...')


if __name__ == "__main__":
    # make no changes here
    train_file = sys.argv[1]
    model_file = sys.argv[2]
    start_time = datetime.datetime.now()
    train_model(train_file, model_file)
    end_time = datetime.datetime.now()
    print('Time:', end_time - start_time)
