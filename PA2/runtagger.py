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


class CharCNN(nn.Module):
    def __init__(self, hidden_size, l=3, k=3):
        super(CharCNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(128, self.hidden_size)
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv1d(hidden_size, l, kernel_size=k, padding=k)
        self.pool = nn.AdaptiveMaxPool1d(1)

    def forward(self, input_chars_list):
        # TODO: Consider adding dropout layer here
        output = self.embedding(input_chars_list)
        output = torch.transpose(output, 2, 1)  # for list of words
        # output = torch.transpose(output, 0, 1)
        output = self.relu(self.conv1(output))
        output = self.pool(output)
        return output.squeeze(2)


class BiLSTM(nn.Module):
    def __init__(
        self, word_embed_size, char_embed_size, lstm_hidden_size,
        vocab_size, tag_size, ix_to_word_chars
    ):
        super(BiLSTM, self).__init__()
        self.word_embed_size = word_embed_size
        self.char_embed_size = char_embed_size
        self.ix_to_word_chars = ix_to_word_chars

        self.embedding = nn.Embedding(vocab_size, self.word_embed_size)
        self.char_cnn = CharCNN(char_embed_size, l=char_embed_size, k=char_embed_size)
        self.bilstm = nn.LSTM(
            word_embed_size + char_embed_size, lstm_hidden_size, bidirectional=True
        )
        self.linear = nn.Linear(lstm_hidden_size * 2, tag_size)

    def forward(self, input_words):
        output = self.embedding(torch.tensor(input_words, dtype=torch.long, device=DEVICE))
        list_of_word_chars = torch.tensor(
            [self.ix_to_word_chars[idx] for idx in input_words], dtype=torch.long,
            device=DEVICE
        )
        char_embeddings = self.char_cnn(list_of_word_chars)
        output = torch.cat((output, char_embeddings), 1)
        hidden, _ = self.bilstm(output.unsqueeze(1))
        hidden_reformatted = hidden.view(len(input_words), -1)
        tag_space = self.linear(hidden_reformatted)
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores


def tag_sentence(test_file, model_file, out_file):
    # write your code here. You can add functions as well.
	# use torch library to load model_file
    model_data = torch.load(model_file)
    state_dict = model_data['state_dict']
    word_to_ix = model_data['word_to_ix']
    ix_to_word_chars = model_data['ix_to_word_chars']
    ix_to_pos = model_data['ix_to_pos']

    bilstm = BiLSTM(
        WORD_EMBEDDINGS_SIZE, CHAR_EMBEDDINGS_SIZE, LSTM_HIDDEN_SIZE,
        len(word_to_ix), len(ix_to_pos), ix_to_word_chars
    )
    bilstm.to(DEVICE)
    bilstm.load_state_dict(state_dict)

    with open(out_file, 'w') as f_output:
        with open(test_file) as f:
            lines = f.readlines()
            for line in lines:
                words = line.split()
                prediction = bilstm([word_to_ix[word] for word in words])
                tags = [ix_to_pos[tag.item()] for tag in torch.argmax(prediction, dim=1)]
                output_str = ''
                for index, (word, pos) in enumerate(zip(words, tags)):
                    if (index == len(words) - 1):
                        output_str += f'{word}/{pos}\n'
                    else:
                        output_str += f'{word}/{pos} '
                f_output.write(output_str)

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
