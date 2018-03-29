import preprocessing
import rnnmodel
import numpy as np

def sample(_input=None, _hidden=None):
  if _input is None:
    input = Variable(wordToTensor(tokens, 'SOS'))
  else:
    input = Variable(wordToTensor(tokens, _input))
  if _hidden is None:
    hidden = rnn.initHidden()
  else:
    hidden = _hidden
  output, hidden = rnn(input, hidden)
  prob = np.exp(output.data[0].numpy())
  sample = np.random.choice(np.arange(0,len(tokens)), 1, p=prob)
  return tokens[sample[0]], hidden

def generate(length):
  initword, hidden = sample()
  for i in range(length):
    next_word, next_hidden = sample(initword, hidden) 
    if next_word is 'EOS':
      print(".")
      initword = 'EOS'
    else:
      print(next_word, end=' ')
      initword = next_word
    hidden = next_hidden
    

p = preprocess('1661.txt.utf-8')
sentences = p.sentences
tokens = p.tokens

rnn = RNN(len(tokens), 128, len(tokens))

n_iters = 5 # increase this to around 10-15
plot_every = 2
all_losses = []
total_loss = 0 # Reset every plot_every iters

i = 0
for iter in range(1, n_iters + 1):
    for idx, sentence in enumerate(sentences):
        output, loss = train(sentence)
        total_loss += loss

        if idx % 100 == 0:
          #print(idx, loss)

        if iter % plot_every == 0:
            all_losses.append(total_loss / plot_every)
            total_loss = 0

generate(300)
    
