# python3.5 runtagger.py <test_file_absolute_path> <model_file_absolute_path> <output_file_absolute_path>

import sys
import datetime
import pickle
import numpy as np
import re


class HiddenMarkovModel:
    def __init__(self, model):
        self.word_index = model['word_index']
        self.pos_index = model['pos_index']
        self.transition_probabilities = model['transition_probabilities']
        self.word_emission_probabilities = model['word_emission_probabilities']
        self.suffix_emission_probabilities = model['suffix_emission_probabilities']
        self.capitalization_emission_probabilities = model['capitalization_emission_probabilities']
        self.pos_list = model['pos_list']

    def simple_rule_based_tagger(self, term):
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

    def capitalization_detector(self, term):
        if (term.isupper()):
            return 1
        if (term[0].isupper()):
            return 2
        return 0

    def compute_viterbi(self, sentence):
        start_tag = '<s>'
        end_tag = '</s>'
        terms = sentence.split()
        viterbi_table = np.zeros((len(self.pos_index), len(terms)), dtype=float)
        backtrack = np.zeros((len(self.pos_index), len(terms)), dtype='int') - 1

        # Init
        pr_word_emission_mtx = None
        if terms[0] in self.word_index:
            pr_word_emission_mtx = self.word_emission_probabilities[:, self.word_index[terms[0]]]
        else:
            pr_word_emission_mtx = self.word_emission_probabilities[:, self.word_index['<UNK>']]
            suffix = self.simple_rule_based_tagger(terms[0])
            capital = self.capitalization_detector(terms[0])
            pr_word_emission_mtx = np.multiply(
                pr_word_emission_mtx, self.suffix_emission_probabilities[:, suffix]
            )
            pr_word_emission_mtx = np.multiply(
                pr_word_emission_mtx, self.capitalization_emission_probabilities[:, capital]
            )

        viterbi_table[:, 0] = np.multiply(
            pr_word_emission_mtx, self.transition_probabilities[self.pos_index[start_tag], :])

        for i in range(1, len(terms)):
            for j in range(0, len(self.pos_index)):
                states_give_prev_pos = np.multiply(
                    viterbi_table[:, i - 1], self.transition_probabilities[:, j]
                )
                pr_word_emission = 0
                if terms[i] in self.word_index:
                    word_index = self.word_index[terms[i]]
                    pr_word_emission = self.word_emission_probabilities[j, word_index]
                else:
                    pr_word_emission = self.word_emission_probabilities[j, self.word_index['<UNK>']]
                    suffix = self.simple_rule_based_tagger(terms[i])
                    capital = self.capitalization_detector(terms[i])
                    pr_word_emission *= self.suffix_emission_probabilities[j, suffix]
                    pr_word_emission *= self.capitalization_emission_probabilities[j, capital]

                max_prev_pos = np.argmax(states_give_prev_pos)
                backtrack[j, i] = max_prev_pos
                viterbi_table[j, i] = (
                    states_give_prev_pos[max_prev_pos] * pr_word_emission
                )

        output = "\n"
        last_tag_index = np.argmax(np.multiply(
            viterbi_table[:, -1], self.transition_probabilities[:, self.pos_index[end_tag]])
        )
        output = terms[-1] + '/' + self.pos_list[last_tag_index] + output

        # TODO: Refactor this a bit, make it a little bit different
        for i in range(0, len(terms) - 1):
            last_tag_index = backtrack[last_tag_index, len(terms) - 2 - i + 1]
            output = terms[len(terms) - 2 - i] + '/' + self.pos_list[last_tag_index] + ' ' + output

        return output

def tag_sentence(test_file, model_file, out_file):
    model = None
    with open(model_file, 'rb') as f:
        model = pickle.load(f)
    hmm = HiddenMarkovModel(model)

    with open(test_file) as f:
        with open(out_file, 'w+') as f_output:
            lines = f.readlines()
            counter = 0
            for line in lines:
                # print((counter / len(lines)) * 100)
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
