# python3.5 runtagger.py <test_file_absolute_path> <model_file_absolute_path> <output_file_absolute_path>

import os
import math
import sys
import torch


DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
EPOCHS = 2  # See if we really need to run 2 epochs
WORD_EMBEDDINGS_SIZE = 10
CHAR_EMBEDDINGS_SIZE = 5
LSTM_HIDDEN_SIZE = 256


def tag_sentence(test_file, model_file, out_file):
    # write your code here. You can add functions as well.
	# use torch library to load model_file
    print('Finished...')

if __name__ == "__main__":
    # make no changes here
    test_file = sys.argv[1]
    model_file = sys.argv[2]
    out_file = sys.argv[3]
    tag_sentence(test_file, model_file, out_file)
