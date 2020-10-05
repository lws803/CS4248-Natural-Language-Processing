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

        self.compute_emission_probabilities()

    def witten_bell_smoothing(self):
        raise NotImplementedError

    def kneser_ney_smoothing(self):
        raise NotImplementedError

    def ao_smoothing(self):
        curr_tag_given_previous_tag = {}
        curr_word_given_tag = {}
        for k, v in self.pos_bigrams.items():
            prev_pos = k[0]
            curr_tag_given_previous_tag[(k[1], k[0])] = (
                (v + 1) / (len(self.pos_count) + self.pos_count[prev_pos])
            )
        for k, v in self.word_pos_pair.items():
            pos = k[0]
            curr_word_given_tag[(k[1], k[0])] = (
                (v + 1) / (len(self.pos_count) + self.pos_count[pos])
            )
        return curr_tag_given_previous_tag, curr_word_given_tag

    def smoothing(self, smoothing_type="ao"):
        curr_tag_given_previous_tag = None
        curr_word_given_tag = None
        if (smoothing_type == "ao"):
            curr_tag_given_previous_tag, curr_word_given_tag = self.ao_smoothing()
        return curr_tag_given_previous_tag, curr_word_given_tag

    def compute_emission_probabilities(self):
        self.curr_tag_given_previous_tag, self.curr_word_given_tag = (
            self.smoothing(smoothing_type="ao")
        )


def tag_sentence(test_file, model_file, out_file):
    model = None
    with open(model_file, 'rb') as f:
        model = pickle.load(f)
    hmm = HiddenMarkovModel(model)
    # write your code here. You can add functions as well.
    # TODO: Use add one smoothing or witten bell smoothing and kneser ney smoothing
    # and evaluate between them
    # TODO: Use log probabilities to avoid underflow errors with floats

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
