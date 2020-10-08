# python3.5 buildtagger.py <train_file_absolute_path> <model_file_absolute_path>

import sys
import datetime
from collections import defaultdict
import pickle
import numpy as np


UNK = '<UNK>'
unk_words_threshold = 1

# TODO: Do an add one smoothing. We need non zero transition paths
def no_smoothing(
    pos_count, pos_bigrams, word_pos_pair, word_count, pos_index, word_index
):
    word_emission_probabilities = np.zeros((len(pos_index), len(word_index)))
    transition_probabilities = np.zeros((len(pos_index), len(pos_index)))

    for k, v in pos_bigrams.items():
        prev_pos = k[0]
        prev_pos_index = pos_index[k[0]]
        curr_pos_index = pos_index[k[1]]
        transition_probabilities[prev_pos_index][curr_pos_index] = v / pos_count[prev_pos]
    for k, v in word_pos_pair.items():
        pos = k[0]
        curr_pos_index = pos_index[k[0]]
        curr_word_index = word_index[k[1]]
        word_emission_probabilities[curr_pos_index][curr_word_index] = v / pos_count[pos]

    return transition_probabilities, word_emission_probabilities


def train_model(train_file, model_file):
    pos_bigrams = defaultdict(int)
    word_pos_pair = defaultdict(int)
    pos_count = defaultdict(int)
    word_count = defaultdict(int)
    output_word_count = defaultdict(int)
    pos_bigram_types = defaultdict(set)
    word_pos_pair_types = defaultdict(set)

    with open(train_file) as f:
        lines = f.readlines()
        # Count word freq
        for line in lines:
            for term_pos_pairs in line.split():
                term_pos_pairs = term_pos_pairs.split('/')
                pos = term_pos_pairs[-1]
                term_pos_pairs.pop()
                term = '/'.join(term_pos_pairs)
                word_count[term] += 1

        for line in lines:
            # Should we ignore the punctuations or account for them as well?
            prev_pos = '<s>'
            pos_count['<s>'] += 1
            for term_pos_pairs in line.split():
                term_pos_pairs = term_pos_pairs.split('/')
                pos = term_pos_pairs[-1]
                term_pos_pairs.pop()
                term = '/'.join(term_pos_pairs)

                pos_bigrams[(prev_pos, pos)] += 1
                pos_count[pos] += 1

                # Remove unk words for now
                if (word_count[term] <= unk_words_threshold):
                    term = UNK
                word_pos_pair[(pos, term)] += 1
                output_word_count[term] += 1
                word_pos_pair_types[pos].add(term)

                pos_bigram_types[prev_pos].add(pos)

                prev_pos = pos
        pos_bigrams[(prev_pos, '</s>')] += 1
        pos_bigram_types[prev_pos].add('</s>')
        pos_count['</s>'] += 1
        pos_index = {}
        word_index = {}
        pos_list = [None] * len(pos_count.keys())

        for index, val in enumerate(pos_count.keys()):
            pos_index[val] = index
            pos_list[index] = val
        for index, val in enumerate(output_word_count.keys()):
            word_index[val] = index

        transition_probabilities, word_emission_probabilities = no_smoothing(
            pos_count, pos_bigrams, word_pos_pair, output_word_count, pos_index, word_index
        )

    with open(model_file, 'wb') as f:
        pickle.dump({
            'pos_bigrams': dict(pos_bigrams),
            'word_pos_pair': dict(word_pos_pair),
            'pos_count': dict(pos_count),
            'word_count': dict(output_word_count),
            'pos_bigram_types': {kv[0]: len(kv[1]) for kv in pos_bigram_types.items()},
            'word_pos_pair_types': {kv[0]: len(kv[1]) for kv in word_pos_pair_types.items()},
            'word_index': word_index,
            'pos_index': pos_index,
            'pos_list': pos_list,
            'transition_probabilities': transition_probabilities,
            'word_emission_probabilities': word_emission_probabilities,
        }, f)

    print('Finished...')


if __name__ == "__main__":
    # make no changes here
    train_file = sys.argv[1]
    model_file = sys.argv[2]
    start_time = datetime.datetime.now()
    train_model(train_file, model_file)
    end_time = datetime.datetime.now()
    print('Time:', end_time - start_time)
