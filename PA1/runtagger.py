# python3.5 runtagger.py <test_file_absolute_path> <model_file_absolute_path> <output_file_absolute_path>

import math
import sys
import datetime
import pickle


class NoSmoothing:
    def no_smoothing(pos_count, pos_bigrams, word_pos_pair):
        curr_tag_given_previous_tag = {}
        curr_word_given_tag = {}
        for k, v in pos_bigrams.items():
            prev_pos = k[0]
            curr_tag_given_previous_tag[(k[1], k[0])] = (
                v / pos_count[prev_pos]
            )
        for k, v in word_pos_pair.items():
            pos = k[0]
            curr_word_given_tag[(k[1], k[0])] = (
                v / pos_count[pos]
            )
        return curr_tag_given_previous_tag, curr_word_given_tag

    @staticmethod
    def compute_score(
        prev_state_score, prev_tag, curr_tag, curr_term, curr_tag_given_previous_tag,
        curr_word_given_tag, pos_count, word_count, last_state=False
    ):
        if last_state:
            score = prev_state_score + math.log(
                curr_tag_given_previous_tag[(curr_tag, prev_tag)]
                if (curr_tag, prev_tag) in curr_tag_given_previous_tag else 1e-300
            )
            return score

        if (curr_term not in word_count):
            curr_term = '<UNK>'
            # TODO: If word is unk we want to estimate based on the captalization or endings/ hyph

        score = prev_state_score + math.log(
            curr_tag_given_previous_tag[(curr_tag, prev_tag)]
            if (curr_tag, prev_tag) in curr_tag_given_previous_tag else 1e-300
        ) + math.log(
            curr_word_given_tag[(curr_term, curr_tag)]
            if (curr_term, curr_tag) in curr_word_given_tag else 1e-300
        )
        return score


class AOSmoothing:
    # FIXME: Figure out what's wrong with add one, it should improve this technically
    ao_discount = 1

    @staticmethod
    def ao_smoothing(pos_count, pos_bigrams, word_pos_pair):
        # TODO: Might be good to calc these in the build tagger
        curr_tag_given_previous_tag = {}
        curr_word_given_tag = {}
        for k, v in pos_bigrams.items():
            prev_pos = k[0]
            curr_tag_given_previous_tag[(k[1], k[0])] = (
                (v + AOSmoothing.ao_discount) / (
                    len(pos_count) * AOSmoothing.ao_discount + pos_count[prev_pos]
                )
            )
        for k, v in word_pos_pair.items():
            pos = k[0]
            curr_word_given_tag[(k[1], k[0])] = (
                (v + AOSmoothing.ao_discount) / (
                    len(pos_count) * AOSmoothing.ao_discount + pos_count[pos]
                )
            )
        return curr_tag_given_previous_tag, curr_word_given_tag

    @staticmethod
    def compute_score(
        prev_state_score, prev_tag, curr_tag, curr_term, curr_tag_given_previous_tag,
        curr_word_given_tag, pos_count, word_count, last_state=False
    ):
        # TODO: This as well so we don't have to recalculate it each time
        if last_state:
            score = prev_state_score + math.log(
                curr_tag_given_previous_tag[(curr_tag, prev_tag)]
                if (curr_tag, prev_tag) in curr_tag_given_previous_tag
                else AOSmoothing.ao_discount / (
                    len(pos_count) * AOSmoothing.ao_discount + pos_count[prev_tag]
                )
            )
            return score

        if (curr_term not in word_count):
            curr_term = '<UNK>'

        pr_term_given_tag = (
            curr_word_given_tag[(curr_term, curr_tag)]
            if (curr_term, curr_tag) in curr_word_given_tag
            else AOSmoothing.ao_discount / (
                len(pos_count) * AOSmoothing.ao_discount + pos_count[curr_tag]
            )
        )

        score = prev_state_score + math.log(
            curr_tag_given_previous_tag[(curr_tag, prev_tag)]
            if (curr_tag, prev_tag) in curr_tag_given_previous_tag
            else AOSmoothing.ao_discount / (
                len(pos_count) * AOSmoothing.ao_discount + pos_count[prev_tag]
            )
        ) + math.log(pr_term_given_tag)
        return score

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
        self.curr_tag_given_previous_tag = None
        self.curr_word_given_tag = None

        self.compute_transition_probabilities()

    @staticmethod
    def kneser_ney_smoothing():
        raise NotImplementedError

    def smoothing(self, smoothing_func, *args):
        curr_tag_given_previous_tag, curr_word_given_tag = smoothing_func(*args)
        return curr_tag_given_previous_tag, curr_word_given_tag

    def compute_transition_probabilities(self):
        self.curr_tag_given_previous_tag, self.curr_word_given_tag = (
            # self.smoothing(
            #     WittenBellSmoothing.witten_bell_smoothing, self.pos_count,
            #     self.pos_bigrams, self.word_pos_pair,
            #     self.pos_bigram_types, self.word_pos_pair_types
            # )
            self.smoothing(
                AOSmoothing.ao_smoothing, self.pos_count, self.pos_bigrams, self.word_pos_pair
            )
            # self.smoothing(
            #     NoSmoothing.no_smoothing, self.pos_count, self.pos_bigrams, self.word_pos_pair
            # )
        )

    def compute_viterbi(self, sentence):
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
            viterbi_table[(tag, terms[0])] = (
                # WittenBellSmoothing.compute_score(
                #     0, start_tag, tag, terms[0], self.curr_tag_given_previous_tag,
                #     self.curr_word_given_tag, self.pos_count, self.pos_bigram_types,
                #     self.word_pos_pair_types, self.word_count
                # )
                AOSmoothing.compute_score(
                    0, start_tag, tag, terms[0], self.curr_tag_given_previous_tag,
                    self.curr_word_given_tag, self.pos_count, self.word_count
                )
                # NoSmoothing.compute_score(
                #     0, start_tag, tag, terms[0], self.curr_tag_given_previous_tag,
                #     self.curr_word_given_tag, self.pos_count, self.word_count
                # )
            )

        for i in range(1, len(terms)):
            for curr_tag in tags:
                backpointer_table[(curr_tag, terms[i])] = None
                viterbi_table[(curr_tag, terms[i])] = -math.inf
                for connecting_tag in tags:
                    curr_term = terms[i]
                    prev_tag = connecting_tag
                    prev_state_score = viterbi_table[(prev_tag, terms[i - 1])]
                    # score = WittenBellSmoothing.compute_score(
                    #     prev_state_score=prev_state_score, prev_tag=prev_tag,
                    #     curr_tag=curr_tag, curr_term=curr_term,
                    #     curr_tag_given_previous_tag=self.curr_tag_given_previous_tag,
                    #     curr_word_given_tag=self.curr_word_given_tag, pos_count=self.pos_count,
                    #     pos_bigram_types=self.pos_bigram_types,
                    #     word_pos_pair_types=self.word_pos_pair_types, word_count=self.word_count
                    # )
                    score = AOSmoothing.compute_score(
                        prev_state_score, prev_tag, curr_tag, curr_term,
                        self.curr_tag_given_previous_tag, self.curr_word_given_tag, self.pos_count,
                        self.word_count
                    )
                    # score = NoSmoothing.compute_score(
                    #     prev_state_score, prev_tag, curr_tag, curr_term,
                    #     self.curr_tag_given_previous_tag, self.curr_word_given_tag, self.pos_count,
                    #     self.word_count
                    # )
                    if score > viterbi_table[(curr_tag, terms[i])]:
                        viterbi_table[(curr_tag, terms[i])] = score
                        backpointer_table[(curr_tag, terms[i])] = connecting_tag

        # End of sentence
        viterbi_table[(end_tag, terms[-1])] = -math.inf
        backpointer_table[(end_tag, terms[-1])] = None
        for connecting_tag in tags:
            # score = WittenBellSmoothing.compute_score(
            #     viterbi_table[(connecting_tag, terms[-1])], connecting_tag, end_tag, terms[-1],
            #     self.curr_tag_given_previous_tag, self.curr_word_given_tag,
            #     self.pos_count, self.pos_bigram_types, self.word_pos_pair_types,
            #     self.word_count, True
            # )
            score = AOSmoothing.compute_score(
                viterbi_table[(connecting_tag, terms[-1])], connecting_tag, end_tag, terms[-1],
                self.curr_tag_given_previous_tag, self.curr_word_given_tag, self.pos_count,
                self.word_count, True
            )
            # score = NoSmoothing.compute_score(
            #     viterbi_table[(connecting_tag, terms[-1])], connecting_tag, end_tag, terms[-1],
            #     self.curr_tag_given_previous_tag, self.curr_word_given_tag, self.pos_count,
            #     self.word_count, True
            # )
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
