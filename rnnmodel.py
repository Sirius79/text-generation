import nltk
nltk.download('punkt')
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
from torchtext import data, dataset

#helper functions

#returns position of the word in vocab
def word2index(tokens, word):
    return tokens.index(word)

#returns one hot tensor of the word
def wordToTensor(tokens, word):
    tensor = torch.zeros(1, len(tokens))
    tensor[0][word2index(tokens, word)] = 1
    return tensor

#returns tensor representation of the sentence
def sentenceToTensor(tokens, sentence):
    tensor = torch.zeros(len(sentence), 1, len(tokens))
    for li, word in enumerate(sentence):
        tensor[li][0][word2index(tokens, word)] = 1
    return tensor

#returns training pairs in each sentence in format (x_t, y_t)
def getTrainingPairs(sentence):
  pairs = {}
  for index in range(len(sentence)-1):
    word = sentence[index]
    pairs[word] = sentence[index+1]
  return pairs

#rnn model
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.o2o = nn.Linear(hidden_size + output_size, output_size)
        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        input_combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(input_combined)
        output = self.i2o(input_combined)
        output_combined = torch.cat((hidden, output), 1)
        output = self.o2o(output_combined)
        output = self.dropout(output)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return Variable(torch.zeros(1, self.hidden_size))

criterion = nn.NLLLoss()

learning_rate = 0.0005

#returns output and loss after training model on each training pair of a sentence
def train(sentence):
    hidden = rnn.initHidden()

    rnn.zero_grad()

    loss = 0
    pairs = getTrainingPairs(sentence)
    for key in pairs:
        _input = Variable(wordToTensor(tokens, key))
        _target = Variable(torch.LongTensor([tokens.index(pairs[key])]))
        output, hidden = rnn(_input, hidden)
        loss += criterion(output, _target)

    loss.backward()

    for p in rnn.parameters():
        p.data.add_(-learning_rate, p.grad.data)

    return output, loss.data[0] / len(sentence)