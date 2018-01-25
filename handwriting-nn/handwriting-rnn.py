"""
Originated from minimal character-level Vanilla RNN model. Written by Andrej Karpathy (@karpathy)
BSD License

Revised by Yizhou Zhou (@yacht220) for study purpose.
"""
import numpy as np
import logging
import argparse
import random
import io
from operator import itemgetter
import pdb
import fileinput

parser = argparse.ArgumentParser()
#parser.add_argument("-u", "--utf8", action="store_true", help="Input is UTF8 encoded")
parser.add_argument("--input", type=str, help="Text file as corpus")
parser.add_argument("--save", type=str, help="Model (parameters) to save")
parser.add_argument("--train", action="store_true", help="Trainig")
parser.add_argument("--load", type=str, help="Load trained model (parameters)")
parser.add_argument("--gensize", type=int, default="200", help="Generation text size")
parser.add_argument("--generate", action="store_true", help="Generate text with current model")
parser.add_argument("--lossthreshold", type=int, help="Exit training when reaching loss threshold. No exit if not set")
parser.add_argument("--logfile", type=str, help="Log file")
parser.add_argument("--validate", action="store_true", help="Validate model")
parser.add_argument("--validatefile", type=str, help="Target file for model validation")
parser.add_argument("--byline", action="store_true", help="Train as line by line")
args = parser.parse_args()

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', datefmt='%Y-%m-%d %H:%M:%S', filename=args.logfile, filemode='w', level=logging.INFO)

# data I/O
#if args.utf8:
#data = open(args.input, 'r').read().decode("UTF-8") # should be simple plain text file
#else:
#data = open(args.input, 'r').read() # should be simple plain text file

#data = data.split()
#chars = list(set(data))
#chars = sorted(list(set(data)))

#logging.info('zhouy22 data - %s' % (data))
#logging.info('zhouy22 char - %s' % (chars))
#print 'zhouy22 data - %s' % (data)
#print 'zhouy22 char - %s' % (chars)
#data_size, vocab_size = len(data), len(chars)
#logging.info('data has %d characters, %d unique.' % (data_size, vocab_size))
#print 'data has %d characters, %d unique.' % (data_size, vocab_size)
#char_to_ix = { ch:i for i,ch in enumerate(chars) }
#ix_to_char = { i:ch for i,ch in enumerate(chars) }

# hyperparameters
input_size = 2
output_size = 10
hidden_size = 100 # size of hidden layer of neurons
# number of steps to unroll the RNN for
if args.byline:
  seq_length = 16
else:
  seq_length = 25 
learning_rate = 1e-1

# model parameters
if args.load is None:
  Wxh = np.random.randn(hidden_size, input_size)*0.01 # input to hidden
  Whh = np.random.randn(hidden_size, hidden_size)*0.01 # hidden to hidden
  Why = np.random.randn(output_size, hidden_size)*0.01 # hidden to output
  bh = np.zeros((hidden_size, 1)) # hidden bias
  by = np.zeros((output_size, 1)) # output bias
else:
  W = np.load(args.load)
  Wxh = W['wxh']
  Whh = W['whh']
  Why = W['why']
  bh = W['bh']
  by = W['by']

mWxh, mWhh, mWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
mbh, mby = np.zeros_like(bh), np.zeros_like(by) # memory variables for Adagrad
#smooth_loss = -np.log(1.0/vocab_size)*seq_length # loss at iteration 0

def train(text):
  if args.byline:
    show_count = 5
  else:
    show_count = 100
  
  # prepare inputs (we're sweeping from left to right in steps seq_length long)
    # forward seq_length characters through the net and fetch gradient
    #oldhprev = hprev
  loss, dWxh, dWhh, dWhy, dbh, dby = lossFun(text)
    #print "oldhprev", oldhprev, "hprev", hprev
    #gradCheck(inputs, targets, oldhprev)

    # sample from the model now and then
  logging.info('loss: %f' % (loss)) #print 'iter %d, loss: %f' % (n, smooth_loss) # print progress
  np.savez(args.save, wxh=Wxh, whh=Whh, why=Why, bh=bh, by=by)
    
    # perform parameter update with Adagrad
  for param, dparam, mem in zip([Wxh, Whh, Why, bh, by], 
                                [dWxh, dWhh, dWhy, dbh, dby], 
                                [mWxh, mWhh, mWhy, mbh, mby]):
    mem += dparam * dparam
    param += -learning_rate * dparam / np.sqrt(mem + 1e-8) # adagrad update


  return loss

def lossFun(inputs):
  """
  inputs,targets are both list of integers.
  hprev is Hx1 array of initial hidden state
  returns the loss, gradients on model parameters, and last hidden state
  """
  xs, hs = {}, {}
  hs[-1] = np.zeros((hidden_size,1)) 
  loss = 0
  length = (len(inputs) - 1) / 2
  # forward pass
  for t in xrange(length):
    xs[t] = np.zeros((input_size, 1))
    xs[t][0] = inputs[t * 2]
    xs[t][1] = inputs[t * 2 + 1]
    hs[t] = np.tanh(np.dot(Wxh, xs[t]) + np.dot(Whh, hs[t-1]) + bh) # hidden state
    
    #loss += -np.log(ps[t][targets[t],0]) # softmax (cross-entropy loss)
  # backward pass: compute gradients going backwards
  ys = np.dot(Why, hs[-1]) + by # unnormalized log probabilities for next chars
  ps = np.exp(ys) / np.sum(np.exp(ys)) # probabilities for next chars
  loss = -np.log(ps[inputs[-1], 0])
  dWxh, dWhh, dWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
  dbh, dby = np.zeros_like(bh), np.zeros_like(by)
  dhnext = np.zeros_like(hs[0])

  dy = np.copy(ps)
  for t in reversed(xrange(length)):
    dy[inputs[-1]] -= 1 # backprop into y. see http://cs231n.github.io/neural-networks-case-study/#grad if confused here
    dWhy += np.dot(dy, hs[t].T)
    dby += dy
    dh = np.dot(Why.T, dy) + dhnext # backprop into h
    dhraw = (1 - hs[t] * hs[t]) * dh # backprop through tanh nonlinearity
    dbh += dhraw
    dWxh += np.dot(dhraw, xs[t].T)
    dWhh += np.dot(dhraw, hs[t-1].T)
    dhnext = np.dot(Whh.T, dhraw)
  for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
    np.clip(dparam, -5, 5, out=dparam) # clip to mitigate exploding gradients
  return loss, dWxh, dWhh, dWhy, dbh, dby


def validate(inputs):
  match = 0
  hs = np.zeros((hidden_size, 1))
  length = (len(inputs) - 1) / 2
  for t in xrange(length):
    xs = np.zeros((input_size, 1))
    xs[0] = inputs[t * 2]
    xs[1] = inputs[t * 2 + 1]
    hs = np.tanh(np.dot(Wxh, xs) + np.dot(Whh, hs) + bh) # hidden state
  ys = np.dot(Why, hs) + by # unnormalized log probabilities for next chars
  ps = np.exp(ys) / np.sum(np.exp(ys)) # probabilities for next chars
 
  if ps.argmax() == inputs[-1]:
    return 1
  else:
    return 0
    

  

if args.train:
  if args.byline:

    epoch = 1
    while True:
      logging.info("epoch %d" %epoch)
      for line in fileinput.input(args.input):
        line = line.split(',')
        line = map(int, line)
        loss = train(line)
      epoch += 1
      match = 0
      total = 0

      for line in fileinput.input(args.validatefile):
        line = line.split(',')
        line = map(int, line)  
        total += 1
        match += validate(line)
      logging.info("match: %d, total %d", match, total)
  
 
elif args.validate:
  pass 

