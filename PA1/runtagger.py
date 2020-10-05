# python3.5 runtagger.py <test_file_absolute_path> <model_file_absolute_path> <output_file_absolute_path>

import os
import math
import sys
import datetime
import pickle


def tag_sentence(test_file, model_file, out_file):
    data = None
    with open(model_file, 'rb') as f:
        data = pickle.load(f)
    import pdb; pdb.set_trace()
    # write your code here. You can add functions as well.
    # TODO: Use add one smoothing or witten bell smoothing and kneser ney smoothing
    # and evaluate between them
    # TODO: Use log probabilities to avoid underflow errors with floats
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
