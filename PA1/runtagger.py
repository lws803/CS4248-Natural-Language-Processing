# python3.5 runtagger.py <test_file_absolute_path> <model_file_absolute_path> <output_file_absolute_path>

import os
import math
import sys
import datetime
import pickle


class HiddenMarkovModel:
    def __init__(self, model):
        self.pos_bigrams = model['pos_bigrams']
        self.word_pos_pair = model['word_pos_pair']
        self.pos_count = model['pos_count']
        self.word_count = model['word_count']
        self.curr_tag_given_previous_tag = None
        self.curr_word_given_tag = None

        self.compute_transition_probabilities()

    @staticmethod
    def ao_smoothing(pos_count, pos_bigrams, word_pos_pair):
        curr_tag_given_previous_tag = {}
        curr_word_given_tag = {}
        for k, v in pos_bigrams.items():
            prev_pos = k[0]
            curr_tag_given_previous_tag[(k[1], k[0])] = (
                (v + 1) / (len(pos_count) + pos_count[prev_pos])
            )
        for k, v in word_pos_pair.items():
            pos = k[0]
            curr_word_given_tag[(k[1], k[0])] = (
                (v + 1) / (len(pos_count) + pos_count[pos])
            )
        return curr_tag_given_previous_tag, curr_word_given_tag

    @staticmethod
    def witten_bell_smoothing():
        raise NotImplementedError

    @staticmethod
    def kneser_ney_smoothing():
        raise NotImplementedError

    def smoothing(self, smoothing_func, *args):
        curr_tag_given_previous_tag, curr_word_given_tag = smoothing_func(*args)
        return curr_tag_given_previous_tag, curr_word_given_tag

    def compute_transition_probabilities(self):
        self.curr_tag_given_previous_tag, self.curr_word_given_tag = (
            self.smoothing(
                HiddenMarkovModel.ao_smoothing, self.pos_count,
                self.pos_bigrams, self.word_pos_pair
            )
        )

    def compute_viterbi(self, sentence):
        # TODO: Depending on which smoothing algo was used, we adjust accordingly
        viterbi_table = {}
        backpointer_table = {}
        start_tag = '<s>'
        end_tag = '</s>'
        tags = list(self.pos_count.keys())
        tags.remove(start_tag)
        tags.remove(end_tag)
        terms = sentence.split()

        # Init
        for tag in tags:
            # AO smoothing
            viterbi_table[(tag, terms[0])] = (
                math.log(
                    self.curr_tag_given_previous_tag[(tag, start_tag)]
                    if (tag, start_tag) in self.curr_tag_given_previous_tag
                    else 1 / (len(self.pos_count) + self.pos_count[start_tag])
                ) + math.log(
                    self.curr_word_given_tag[(terms[0], tag)]
                    if (terms[0], tag) in self.curr_word_given_tag
                    else 1 / (len(self.pos_count) + self.pos_count[tag])
                )
            )

        for i in range(1, len(terms)):
            for curr_tag in tags:
                max_connecting_tag = None
                max_score = -math.inf
                for connecting_tag in tags:
                    score = viterbi_table[(connecting_tag, terms[i - 1])] + math.log(
                        self.curr_tag_given_previous_tag[(curr_tag, connecting_tag)]
                        if (curr_tag, connecting_tag) in self.curr_tag_given_previous_tag
                        else 1 / (len(self.pos_count) + self.pos_count[connecting_tag])
                    ) + math.log(
                        self.curr_word_given_tag[(terms[i], curr_tag)]
                        if (terms[i], curr_tag) in self.curr_word_given_tag
                        else 1 / (len(self.pos_count) + self.pos_count[curr_tag])
                    )
                    if score > max_score:
                        max_score = score
                        max_connecting_tag = connecting_tag
                backpointer_table[(curr_tag, terms[i])] = max_connecting_tag
                viterbi_table[(curr_tag, terms[i])] = max_score

        max_score = -math.inf
        max_connecting_tag = None
        for connecting_tag in tags:
            score = viterbi_table[(connecting_tag, terms[-1])] + math.log(
                self.curr_tag_given_previous_tag[(end_tag, connecting_tag)]
                if (end_tag, connecting_tag) in self.curr_tag_given_previous_tag
                else 1 / (len(self.pos_count) + self.pos_count[connecting_tag])
            ) + math.log(
                self.curr_word_given_tag[(terms[i], end_tag)]
                if (terms[i], end_tag) in self.curr_word_given_tag
                else 1 / (len(self.pos_count) + self.pos_count[end_tag])
            )
            if score > max_score:
                max_score = score
                max_connecting_tag = connecting_tag
        viterbi_table[(end_tag, terms[-1])] = max_score
        backpointer_table[(end_tag, terms[-1])] = max_connecting_tag

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

    sentence = (
        'And it was stupid .'
    )
    tags = hmm.compute_viterbi(sentence)
    output_str = ''
    for i in range(0, len(tags)):
        output_str += '{}/{} '.format(sentence.split()[i], tags[i])
    print(output_str)
    # TODO: Use add one smoothing or witten bell smoothing and kneser ney smoothing
    # and evaluate between them

    with open(test_file) as f:
        for line in f.readlines():
            # TODO: Generate the hidden markov model here
            pass
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
