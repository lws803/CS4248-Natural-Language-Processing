# python3.5 buildtagger.py <train_file_absolute_path> <model_file_absolute_path>

import sys
import datetime
from collections import defaultdict
import pickle
import numpy as np
import re


UNK = '<UNK>'
unk_words_threshold = 1


def compute_emission_probabilities(
    word_pos_pair, pos_index, word_index, pos_count, word_emission_probabilities, suffix_pos,
    suffix_emission_probabilities, capital_pos, capitalization_emission_probabilities
):
    for k, v in word_pos_pair.items():
        pos = k[0]
        curr_pos_index = pos_index[k[0]]
        curr_word_index = word_index[k[1]]
        word_emission_probabilities[curr_pos_index][curr_word_index] = v / pos_count[pos]

    for k, v in suffix_pos.items():
        curr_pos = pos_index[k[1]]
        suffix_code = k[0]
        suffix_emission_probabilities[curr_pos][suffix_code] = v / pos_count[k[1]]
    for k, v in capital_pos.items():
        curr_pos = pos_index[k[1]]
        capital_code = k[0]
        capitalization_emission_probabilities[curr_pos][capital_code] = v / pos_count[k[1]]

    return (
        word_emission_probabilities,
        suffix_emission_probabilities,
        capitalization_emission_probabilities
    )


def no_smoothing(
    pos_count, pos_bigrams, word_pos_pair, word_count, pos_index,
    word_index, capital_pos, suffix_pos, pos_list, word_list, pos_bigram_types
):
    word_emission_probabilities = np.zeros((len(pos_index), len(word_index)))
    suffix_emission_probabilities = np.zeros((len(pos_index), 8))
    capitalization_emission_probabilities = np.zeros((len(pos_index), 3))
    transition_probabilities = np.zeros((len(pos_index), len(pos_index)))

    for k, v in pos_bigrams.items():
        prev_pos = k[0]
        prev_pos_index = pos_index[k[0]]
        curr_pos_index = pos_index[k[1]]
        transition_probabilities[prev_pos_index][curr_pos_index] = v / pos_count[prev_pos]

    (
        word_emission_probabilities,
        suffix_emission_probabilities,
        capitalization_emission_probabilities
    ) = compute_emission_probabilities(
        word_pos_pair, pos_index, word_index, pos_count, word_emission_probabilities, suffix_pos,
        suffix_emission_probabilities, capital_pos, capitalization_emission_probabilities
    )

    return (
        transition_probabilities, word_emission_probabilities, suffix_emission_probabilities,
        capitalization_emission_probabilities
    )


def ao_smoothing(
    pos_count, pos_bigrams, word_pos_pair, word_count, pos_index,
    word_index, capital_pos, suffix_pos, pos_list, word_list, pos_bigram_types
):
    word_emission_probabilities = np.zeros((len(pos_index), len(word_index)))
    suffix_emission_probabilities = np.zeros((len(pos_index), 8))
    capitalization_emission_probabilities = np.zeros((len(pos_index), 3))
    transition_probabilities = np.zeros((len(pos_index), len(pos_index)))

    for i in range(0, len(pos_index)):
        for j in range(0, len(pos_index)):
            prev_pos = pos_list[i]
            curr_pos = pos_list[j]
            v = pos_bigrams[(prev_pos, curr_pos)] if (prev_pos, curr_pos) in pos_bigrams else 0
            transition_probabilities[i][j] = (
                (v + 1) / (pos_count[prev_pos] + len(pos_index))
            )

    # We don't want smoothing for word emission
    (
        word_emission_probabilities,
        suffix_emission_probabilities,
        capitalization_emission_probabilities
    ) = compute_emission_probabilities(
        word_pos_pair, pos_index, word_index, pos_count, word_emission_probabilities, suffix_pos,
        suffix_emission_probabilities, capital_pos, capitalization_emission_probabilities
    )

    return (
        transition_probabilities, word_emission_probabilities, suffix_emission_probabilities,
        capitalization_emission_probabilities
    )



def witten_bell_smoothing(
    pos_count, pos_bigrams, word_pos_pair, word_count, pos_index,
    word_index, capital_pos, suffix_pos, pos_list, word_list, pos_bigram_types
):
    word_emission_probabilities = np.zeros((len(pos_index), len(word_index)))
    suffix_emission_probabilities = np.zeros((len(pos_index), 8))
    capitalization_emission_probabilities = np.zeros((len(pos_index), 3))
    transition_probabilities = np.zeros((len(pos_index), len(pos_index)))

    for i in range(0, len(pos_index)):
        for j in range(0, len(pos_index)):
            prev_pos = pos_list[i]
            curr_pos = pos_list[j]
            v = pos_bigrams[(prev_pos, curr_pos)] if (prev_pos, curr_pos) in pos_bigrams else 0
            Tw0 = len(pos_bigram_types[prev_pos])
            if v > 0:
                transition_probabilities[i][j] = (
                    v / (pos_count[prev_pos] + Tw0)
                )
            else:
                Zw0 = len(pos_index) - Tw0
                transition_probabilities[i][j] = (
                    Tw0 / (Zw0 * (pos_count[prev_pos] + Tw0))
                )

    # We don't want smoothing for word emission
    (
        word_emission_probabilities,
        suffix_emission_probabilities,
        capitalization_emission_probabilities
    ) = compute_emission_probabilities(
        word_pos_pair, pos_index, word_index, pos_count, word_emission_probabilities, suffix_pos,
        suffix_emission_probabilities, capital_pos, capitalization_emission_probabilities
    )

    return (
        transition_probabilities, word_emission_probabilities, suffix_emission_probabilities,
        capitalization_emission_probabilities
    )


def simple_rule_based_tagger(term):
    rules = [
        (r'.*ing$', 1),
        (r'.*ed$', 2),
        (r'.*ion$', 3),
        (r'.*s$', 4),
        (r'.*al$', 5),
        (r'.*ive$', 6),
        (r'[-]', 7),
        (r'.*', 0)
    ]
    for rule in rules:
        if re.match(rule[0], term):
            return rule[1]


def capitalization_detector(term):
    if (term.isupper()):
        return 1
    if (term[0].isupper()):
        return 2
    return 0


def train_model(train_file, model_file):
    pos_bigrams = defaultdict(int)
    word_pos_pair = defaultdict(int)
    pos_count = defaultdict(int)
    word_count = defaultdict(int)
    output_word_count = defaultdict(int)
    pos_bigram_types = defaultdict(set)
    word_pos_pair_types = defaultdict(set)
    capital_pos = defaultdict(int)
    suffix_pos = defaultdict(int)

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

                suffix_pos[(simple_rule_based_tagger(term), pos)] += 1
                capital_pos[(capitalization_detector(term), pos)] += 1

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
        word_list = [None] * len(word_count.keys())

        for index, val in enumerate(pos_count.keys()):
            pos_index[val] = index
            pos_list[index] = val
        for index, val in enumerate(output_word_count.keys()):
            word_index[val] = index
            word_list[index] = val

        (transition_probabilities, word_emission_probabilities, suffix_emission_probabilities,
        capitalization_emission_probabilities) = witten_bell_smoothing(
            pos_count, pos_bigrams, word_pos_pair, output_word_count, pos_index, word_index,
            capital_pos, suffix_pos, pos_list, word_list, pos_bigram_types
        )

    with open(model_file, 'wb') as f:
        pickle.dump({
            'word_index': word_index,
            'pos_index': pos_index,
            'pos_list': pos_list,
            'transition_probabilities': transition_probabilities,
            'word_emission_probabilities': word_emission_probabilities,
            'suffix_emission_probabilities': suffix_emission_probabilities,
            'capitalization_emission_probabilities': capitalization_emission_probabilities,
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
