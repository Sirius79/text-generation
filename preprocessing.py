import nltk
nltk.download('punkt')
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

class preprocess():
  
  def __init__(self, filename):
    self.filename = filename
    self.sequences = []
    self.sentences = []
    self.tokens = []
    self.splitSentences()
    self.getTokens()
    
  @staticmethod
  def load(filename):
    file = open(filename, 'r')
    text = file.read()
    file.close()
    return text
  
  def getSequences(self):
    text = self.load(self.filename)
    _text = text.split("\n\n")
    for line in _text:
      self.sequences.append(line)
      continue
  
  def clean(self):
    self.getSequences()
    for i in range(len(self.sequences)):
      if 'Mr.' or 'MR.' in p.sequences[i]:
        self.sequences[i] = self.sequences[i].replace('Mr.', 'mister')
        self.sequences[i] = self.sequences[i].replace('MR.', 'mister')
      if 'Ms.' or 'MS.' in p.sequences[i]:
        self.sequences[i] = self.sequences[i].replace('Ms.', 'miss')
        self.sequences[i] = self.sequences[i].replace('MS.', 'miss')
      if 'Mrs.' or 'MRS.' in p.sequences[i]:
        self.sequences[i] = self.sequences[i].replace('Mrs.', 'mrs')
        self.sequences[i] = self.sequences[i].replace('MRS.', 'mrs')
      if 'Dr.' or 'DR.' in p.sequences[i]:
        self.sequences[i] = self.sequences[i].replace('Dr.', 'doctor')
        self.sequences[i] = self.sequences[i].replace('DR.', 'doctor')
      if 'St.' or 'ST.' in p.sequences[i]:
        self.sequences[i] = self.sequences[i].replace('St.', 'st')
        self.sequences[i] = self.sequences[i].replace('ST.', 'st')
      self.sequences[i] = self.sequences[i].lower()
      self.sequences[i] = self.sequences[i].replace("'", "")
      self.sequences[i] = self.sequences[i].replace('-', ' ')
      self.sequences[i] = self.sequences[i].replace('"', '')
      if '\n' in self.sequences[i]:
        self.sequences[i] = self.sequences[i].replace('\n', ' ')
      
   
  def splitSentences(self):
    self.clean()
    for i in range(len(self.sequences)):
      sentence = self.sequences[i].split('.')
      for sent in sentence:
        temp = []
        if sent is not '':
          temp = nltk.word_tokenize(sent) 
          temp.insert(0, 'SOS')
          temp.append('EOS')
          self.sentences.append(temp)
          
  def getTokens(self):
    for sent in self.sentences:
      for word in sent:
        if word not in self.tokens:
          self.tokens.append(word)
    self.tokens.sort()
  