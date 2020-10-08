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

    def compute_score(
        self, prev_state_score, prev_tag_index, curr_tag_index, curr_term,
        last_state=False
    ):
        # TODO: Find a way to skip if there is no viterbi path here
        if last_state:
            score = prev_state_score + math.log(
                self.transition_probabilities[prev_tag_index][curr_tag_index]
                if self.transition_probabilities[prev_tag_index][curr_tag_index] > 0 else
                1e-300
            )
            return score

        if (curr_term not in self.word_index):
            curr_term = '<UNK>'
            # TODO: If word is unk we want to estimate based on the captalization or endings/ hyph
        curr_word_index = self.word_index[curr_term]
        score = prev_state_score + math.log(
            self.transition_probabilities[prev_tag_index][curr_tag_index]
            if self.transition_probabilities[prev_tag_index][curr_tag_index] > 0 else
            1e-300
        ) + math.log(
            self.word_emission_probabilities[curr_tag_index][curr_word_index]
            if self.word_emission_probabilities[curr_tag_index][curr_word_index] > 0 else
            1e-300
        )
        return score

    def compute_viterbi(self, sentence):
        backpointer_table = {}
        start_tag = '<s>'
        end_tag = '</s>'
        # tags = list(self.pos_count.keys())
        # tags.remove(start_tag)
        # tags.remove(end_tag)
        # FIXME: Do we need to remove these two tags
        terms = sentence.split()
        viterbi_table = np.full(
            (len(self.pos_index), len(terms)), -math.inf, dtype=float
        )
        # Init
        for i in range(0, len(self.pos_index)):
            viterbi_table[i][0] = (
                self.compute_score(0, self.pos_index[start_tag], i, terms[0])
            )
        # TODO: Find out how to optimize this with the numpy code.
        for i in range(1, len(terms)):
            for j in range(0, len(self.pos_index)):
                backpointer_table[(self.pos_list[j], terms[i])] = None
                for k in range(0, len(self.pos_index)):
                    prev_state_score = viterbi_table[k][i - 1]
                    score = self.compute_score(prev_state_score, k, j, terms[i])
                    if score > viterbi_table[j][i]:
                        viterbi_table[j][i] = score
                        backpointer_table[(self.pos_list[j], terms[i])] = self.pos_list[k]

        # End of sentence
        backpointer_table[(end_tag, terms[-1])] = None
        end_tag_index = self.pos_index[end_tag]
        for i in range(0, len(self.pos_index)):
            prev_state_score = viterbi_table[i][len(terms) - 1]
            score = self.compute_score(
                prev_state_score, i, end_tag_index, terms[-1], True
            )
            if score > viterbi_table[end_tag_index][len(terms) - 1]:
                viterbi_table[end_tag_index][len(terms) - 1] = score
                backpointer_table[(end_tag, terms[-1])] = self.pos_list[i]

        # Backtrack
        reversed_terms_list = terms[::-1]
        reversed_terms_list.pop()
        output_tags = [backpointer_table[(end_tag, reversed_terms_list[0])]]
        prev_tag = backpointer_table[(end_tag, reversed_terms_list[0])]
        for term in reversed_terms_list:
            output_tags.append(backpointer_table[(prev_tag, term)])
            prev_tag = backpointer_table[(prev_tag, term)]
        output_tags.reverse()

        return output_tags

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
                tags = hmm.compute_viterbi(line)
                output_str = ''
                for i in range(0, len(tags)):
                    output_str += '{}/{}'.format(line.split()[i], tags[i])
                    if i < len(tags) - 1:
                        output_str += ' '
                    else:
                        output_str += '\n'
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
