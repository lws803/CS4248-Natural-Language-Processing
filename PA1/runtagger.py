# python3.5 runtagger.py <test_file_absolute_path> <model_file_absolute_path> <output_file_absolute_path>

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
        # TODO: Double check the math again for each of these
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

        def calculate_score(prev_tag, curr_tag, prev_term, curr_term):
            score = viterbi_table[(prev_tag, prev_term)] + math.log(
                self.curr_tag_given_previous_tag[(curr_tag, prev_tag)]
                if (curr_tag, prev_tag) in self.curr_tag_given_previous_tag
                else 1 / (len(self.pos_count) + self.pos_count[prev_tag])
            ) + math.log(
                self.curr_word_given_tag[(curr_term, curr_tag)]
                if (curr_term, curr_tag) in self.curr_word_given_tag
                else 1 / (len(self.pos_count) + self.pos_count[curr_tag])
            )
            return score

        for i in range(1, len(terms)):
            for curr_tag in tags:
                backpointer_table[(curr_tag, terms[i])] = None
                viterbi_table[(curr_tag, terms[i])] = -math.inf
                for connecting_tag in tags:
                    score = calculate_score(connecting_tag, curr_tag, terms[i - 1], terms[i])
                    if score > viterbi_table[(curr_tag, terms[i])]:
                        viterbi_table[(curr_tag, terms[i])] = score
                        backpointer_table[(curr_tag, terms[i])] = connecting_tag

        # End of sentence
        viterbi_table[(end_tag, terms[-1])] = -math.inf
        backpointer_table[(end_tag, terms[-1])] = None
        for connecting_tag in tags:
            score = viterbi_table[(connecting_tag, terms[-1])] + math.log(
                self.curr_tag_given_previous_tag[(end_tag, connecting_tag)]
                if (end_tag, connecting_tag) in self.curr_tag_given_previous_tag
                else 1 / (len(self.pos_count) + self.pos_count[connecting_tag])
            )
            if score > viterbi_table[(end_tag, terms[-1])]:
                viterbi_table[(end_tag, terms[-1])] = score
                backpointer_table[(end_tag, terms[-1])] = connecting_tag

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

    # TODO: Use add one smoothing or witten bell smoothing and kneser ney smoothing
    # and evaluate between them

    with open(test_file) as f:
        with open(out_file, 'w+') as f_output:
            lines = f.readlines()
            counter = 0
            for line in lines:
                # print((counter / len(lines)) * 100)
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
