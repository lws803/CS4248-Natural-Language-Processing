# python3.5 runtagger.py <test_file_absolute_path> <model_file_absolute_path> <output_file_absolute_path>

import math
import sys
import datetime
import pickle
import numpy as np


class WittenBellSmoothing:
    @staticmethod
    def witten_bell_smoothing(
        pos_count, pos_bigrams, word_pos_pair, pos_bigram_types, word_pos_pair_types
    ):
        curr_tag_given_previous_tag = {}
        curr_word_given_tag = {}
        for k, v in pos_bigrams.items():
            curr_tag_given_previous_tag[(k[1], k[0])] = (
                v / (pos_count[k[0]] + pos_bigram_types[k[0]])
            )

        for k, v in word_pos_pair.items():
            pos = k[0]
            curr_word_given_tag[(k[1], k[0])] = (
                v / (pos_count[pos] + word_pos_pair_types[pos])
            )
        return curr_tag_given_previous_tag, curr_word_given_tag

    @staticmethod
    def compute_score(
        prev_state_score, prev_tag, curr_tag, curr_term,
        curr_tag_given_previous_tag, curr_word_given_tag, pos_count,
        pos_bigram_types, word_pos_pair_types, word_count,
        last_state=False
    ):
        if (curr_term not in word_count):
            curr_term = '<UNK>'

        score = 0
        T_prev_tag = pos_bigram_types[prev_tag]
        pr_curr_tag_prev_tag = (
            curr_tag_given_previous_tag[(curr_tag, prev_tag)]
            if (curr_tag, prev_tag) in curr_tag_given_previous_tag
            else T_prev_tag / (
                (len(pos_count) - T_prev_tag) * (pos_count[prev_tag] + T_prev_tag))
        )
        if not last_state:
            T_seen_word_types_given_tag = word_pos_pair_types[curr_tag]
            pr_curr_term_curr_tag = (
                curr_word_given_tag[(curr_term, curr_tag)]
                if (curr_term, curr_tag) in curr_word_given_tag
                else T_seen_word_types_given_tag / (
                    (len(word_count) - T_seen_word_types_given_tag) * (
                        pos_count[curr_tag] + T_seen_word_types_given_tag
                    )
                )
            )
            score = (
                prev_state_score + math.log(pr_curr_tag_prev_tag) + math.log(pr_curr_term_curr_tag)
            )
        else:
            score = prev_state_score + math.log(pr_curr_tag_prev_tag)
        return score


class HiddenMarkovModel:
    def __init__(self, model):
        self.pos_bigrams = model['pos_bigrams']
        self.word_pos_pair = model['word_pos_pair']
        self.pos_count = model['pos_count']
        self.word_count = model['word_count']
        self.pos_bigram_types = model['pos_bigram_types']
        self.word_pos_pair_types = model['word_pos_pair_types']
        self.word_index = model['word_index']
        self.pos_index = model['pos_index']
        self.transition_probabilities = model['transition_probabilities']
        self.word_emission_probabilities = model['word_emission_probabilities']
        self.pos_list = model['pos_list']
        self.curr_tag_given_previous_tag = None
        self.curr_word_given_tag = None

    def compute_viterbi(self, sentence):
        start_tag = '<s>'
        end_tag = '</s>'
        terms = sentence.split()
        # TODO: Refactor and change implementation to use log probabilities
        viterbi_table = np.zeros((len(self.pos_index), len(terms)), dtype=float)
        backtrack = np.zeros((len(self.pos_index), len(terms)), dtype='int') - 1
        # Init
        viterbi_table[:, 0] = np.multiply(
            self.word_emission_probabilities[
                :,
                self.word_index[terms[0] if terms[0] in self.word_index else '<UNK>']
            ], self.transition_probabilities[self.pos_index[start_tag], :])

        for i in range(1, len(terms)):
            for j in range(0, len(self.pos_index)):
                states_give_prev_pos = np.multiply(
                    viterbi_table[:, i - 1], self.transition_probabilities[:, j]
                )
                curr_term = terms[i]
                if terms[i] not in self.word_index:
                    curr_term = '<UNK>'
                word_index = self.word_index[curr_term]

                max_prev_pos = np.argmax(states_give_prev_pos)
                backtrack[j, i] = max_prev_pos
                viterbi_table[j, i] = (
                    states_give_prev_pos[max_prev_pos] *
                    self.word_emission_probabilities[j, word_index]
                )

        output = "\n"
        tagIndex = np.argmax(np.multiply(
            viterbi_table[:, -1], self.transition_probabilities[:, self.pos_index[end_tag]])
        )
        output = terms[-1] + '/' + self.pos_list[tagIndex] + output

        for i in range(0, len(terms) - 1):
            tagIndex = backtrack[tagIndex, len(terms) - 2 - i + 1]
            output = terms[len(terms) - 2 - i] + '/' + self.pos_list[tagIndex] + ' ' + output

        return output

def tag_sentence(test_file, model_file, out_file):
    model = None
    with open(model_file, 'rb') as f:
        model = pickle.load(f)
    hmm = HiddenMarkovModel(model)

    # sentence = input('input sentence: ')
    # tags = hmm.compute_viterbi(sentence)
    # output_str = ''
    # for i in range(0, len(tags)):
    #     output_str += '{}/{} '.format(sentence.split()[i], tags[i])
    # print(output_str)

    with open(test_file) as f:
        with open(out_file, 'w+') as f_output:
            lines = f.readlines()
            counter = 0
            for line in lines:
                print((counter / len(lines)) * 100)
                output_str = hmm.compute_viterbi(line)
                f_output.write(output_str)
                counter += 1

    print('Finished...')


if __name__ == "__main__":
    # make no changes here
    test_file = sys.argv[1]
    model_file = sys.argv[2]
    out_file = sys.argv[3]
    start_time = datetime.datetime.now()
    tag_sentence(test_file, model_file, out_file)
    end_time = datetime.datetime.now()
    print('Time:', end_time - start_time)
