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
data = open(args.input, 'r').read().decode("UTF-8") # should be simple plain text file
#else:
#data = open(args.input, 'r').read() # should be simple plain text file

#data = data.split()
chars = list(set(data))
#chars = sorted(list(set(data)))

#logging.info('zhouy22 data - %s' % (data))
#logging.info('zhouy22 char - %s' % (chars))
#print 'zhouy22 data - %s' % (data)
#print 'zhouy22 char - %s' % (chars)
data_size, vocab_size = len(data), len(chars)
logging.info('data has %d characters, %d unique.' % (data_size, vocab_size))
#print 'data has %d characters, %d unique.' % (data_size, vocab_size)
char_to_ix = { ch:i for i,ch in enumerate(chars) }
ix_to_char = { i:ch for i,ch in enumerate(chars) }

# hyperparameters
hidden_size = 100 # size of hidden layer of neurons
# number of steps to unroll the RNN for
if args.byline:
  seq_length = 5
else:
  seq_length = 25 
learning_rate = 1e-1

# model parameters
if args.load is None:
  Wxh = np.random.randn(hidden_size, vocab_size)*0.01 # input to hidden
  Whh = np.random.randn(hidden_size, hidden_size)*0.01 # hidden to hidden
  Why = np.random.randn(vocab_size, hidden_size)*0.01 # hidden to output
  bh = np.zeros((hidden_size, 1)) # hidden bias
  by = np.zeros((vocab_size, 1)) # output bias
else:
  W = np.load(args.load)
  Wxh = W['wxh']
  Whh = W['whh']
  Why = W['why']
  bh = W['bh']
  by = W['by']

mWxh, mWhh, mWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
mbh, mby = np.zeros_like(bh), np.zeros_like(by) # memory variables for Adagrad
smooth_loss = -np.log(1.0/vocab_size)*seq_length # loss at iteration 0

def train(text):
  global smooth_loss
  loss = 0
  n = 0
  p = 0 # go from start of text
  hprev = np.zeros((hidden_size,1)) # reset RNN memory
  if args.byline:
    show_count = 5
  else:
    show_count = 100
  
  # prepare inputs (we're sweeping from left to right in steps seq_length long)
  while p+seq_length+1 < len(text):
    inputs = [char_to_ix[ch] for ch in text[p:p+seq_length]]
    targets = [char_to_ix[ch] for ch in text[p+1:p+seq_length+1]]

    # forward seq_length characters through the net and fetch gradient
    #oldhprev = hprev
    loss, dWxh, dWhh, dWhy, dbh, dby, hprev = lossFun(inputs, targets, hprev)
    #print "oldhprev", oldhprev, "hprev", hprev
    #gradCheck(inputs, targets, oldhprev)
    smooth_loss = smooth_loss * 0.999 + loss * 0.001

    # sample from the model now and then
    if n % show_count == 0: 
      sample_ix = sample(hprev, inputs[0], 200)
      sample_char = [ix_to_char[ix] for ix in sample_ix]
      txt = ''.join(sample_char)
      logging.info('----\n %s \n----' % (txt))
      logging.info('iter %d, loss: %f, smooth loss: %f' % (n, loss, smooth_loss)) #print 'iter %d, loss: %f' % (n, smooth_loss) # print progress
      np.savez(args.save, wxh=Wxh, whh=Whh, why=Why, bh=bh, by=by)
    
    # perform parameter update with Adagrad
    for param, dparam, mem in zip([Wxh, Whh, Why, bh, by], 
                                  [dWxh, dWhh, dWhy, dbh, dby], 
                                  [mWxh, mWhh, mWhy, mbh, mby]):
      mem += dparam * dparam
      param += -learning_rate * dparam / np.sqrt(mem + 1e-8) # adagrad update

    p += seq_length # move text pointer
    n += 1 # iteration counter

  return loss, smooth_loss

def lossFun(inputs, targets, hprev):
  """
  inputs,targets are both list of integers.
  hprev is Hx1 array of initial hidden state
  returns the loss, gradients on model parameters, and last hidden state
  """
  xs, hs, ys, ps = {}, {}, {}, {}
  hs[-1] = np.copy(hprev)
  loss = 0
  # forward pass
  for t in xrange(len(inputs)):
    xs[t] = np.zeros((vocab_size,1)) # encode in 1-of-k representation
    xs[t][inputs[t]] = 1
    hs[t] = np.tanh(np.dot(Wxh, xs[t]) + np.dot(Whh, hs[t-1]) + bh) # hidden state
    ys[t] = np.dot(Why, hs[t]) + by # unnormalized log probabilities for next chars
    ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t])) # probabilities for next chars
    loss += -np.log(ps[t][targets[t],0]) # softmax (cross-entropy loss)
  # backward pass: compute gradients going backwards
  dWxh, dWhh, dWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
  dbh, dby = np.zeros_like(bh), np.zeros_like(by)
  dhnext = np.zeros_like(hs[0])
  for t in reversed(xrange(len(inputs))):
    dy = np.copy(ps[t])
    dy[targets[t]] -= 1 # backprop into y. see http://cs231n.github.io/neural-networks-case-study/#grad if confused here
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
  return loss, dWxh, dWhh, dWhy, dbh, dby, hs[len(inputs)-1]

def sample(h, seed_ix, n):
  """ 
  sample a sequence of integers from the model 
  h is memory state, seed_ix is seed letter for first time step
  """
  x = np.zeros((vocab_size, 1))
  x[seed_ix] = 1
  ixes = []
  for t in xrange(n):
    h = np.tanh(np.dot(Wxh, x) + np.dot(Whh, h) + bh)
    y = np.dot(Why, h) + by
    p = np.exp(y) / np.sum(np.exp(y))
    ix = np.random.choice(range(vocab_size), p=p.ravel())
    #maxix = p.argmax()
    '''if ix != maxix:
      logging.warn("Not same ix in sampling. ix=%d, maxix=%d" % (ix, maxix))'''
    x = np.zeros((vocab_size, 1))
    x[ix] = 1
    ixes.append(ix)
    #x[maxix] = 1
    #ixes.append(maxix)
  return ixes

def gradCheck(inputs, target, hprev):
  global Wxh, Whh, Why, bh, by
  num_checks, delta = 1, 1e-5
  _, dWxh, dWhh, dWhy, dbh, dby, _ = lossFun(inputs, targets, hprev)
  for param,dparam,name in zip([Wxh, Whh, Why, bh, by], [dWxh, dWhh, dWhy, dbh, dby], ['Wxh', 'Whh', 'Why', 'bh', 'by']):
    s0 = dparam.shape
    s1 = param.shape
    assert s0 == s1, 'Error dims dont match: %s and %s.' % (`s0`, `s1`)
    #print name
    for i in xrange(num_checks):
      ri = int(random.uniform(0,param.size))
      # evaluate cost at [x + delta] and [x - delta]
      old_val = param.flat[ri]
      param.flat[ri] = old_val + delta
      cg0, _, _, _, _, _, _ = lossFun(inputs, targets, hprev)
      param.flat[ri] = old_val - delta
      cg1, _, _, _, _, _, _ = lossFun(inputs, targets, hprev)
      param.flat[ri] = old_val # reset old value for this parameter
      # fetch both numerical and analytic gradient
      grad_analytic = dparam.flat[ri]
      grad_numerical = (cg0 - cg1) / ( 2 * delta )
      rel_error = abs(grad_analytic - grad_numerical) / abs(grad_numerical + grad_analytic)
      logging.info('%f, %f => %e ' % (grad_numerical, grad_analytic, rel_error))
      #print '%f, %f => %e ' % (grad_numerical, grad_analytic, rel_error)
      # rel_error should be on order of 1e-7 or less

def validate(inputs):
  global smooth_loss
  loss = 0
  avg_loss = 0
  h = np.zeros((hidden_size, 1))
  len_line = len(inputs)
  if len_line <= 1:
    return None, None
  match = 0
  if args.byline:
    show_count = 5
  else:
    show_count = 100
  loss_add_count = 0
  for i in xrange(len_line - 1):
    x = np.zeros((vocab_size, 1))
    x[inputs[i]] = 1
    h = np.tanh(np.dot(Wxh, x) + np.dot(Whh, h) + bh)
    y = np.dot(Why, h) + by
    p = np.exp(y) / np.sum(np.exp(y))
    loss += -np.log(p[inputs[i + 1], 0]) # softmax (cross-entropy loss)

    if i % seq_length == 0:
      smooth_loss = smooth_loss * 0.999 + loss * 0.001
      if (i / seq_length) % show_count == 0:
        logging.info("iter %d, loss %f, smooth_loss %f" % (i / seq_length, loss, smooth_loss))
      avg_loss += loss
      loss_add_count += 1
      loss = 0

    #ix = np.random.choice(range(vocab_size), p=p.ravel())
    #ix = p.argmax()
    if p[inputs[i + 1]] > 5e-1: #p[ix] == p[inputs[i + 1]]:
      match += 1

  '''if loss != 0:
    avg_loss += loss
    loss_add_count += 1'''

  avg_loss /= loss_add_count 
    
  match_rate = float(match) / float(len_line - 1) * 100.0

  return avg_loss, match_rate
    
if args.train:
  if args.byline:
    epoch = 1
    while True:
      logging.info("epoch %d" %epoch)
      with io.open(args.input, mode = 'r', encoding = 'utf-8') as infile:
        for line in infile:
          loss, smooth_loss = train(line)
      epoch += 1

      if args.lossthreshold is not None and smooth_loss <= args.lossthreshold:
        logging.info('loss: %f, smooth loss: %f, exit' % (loss, smooth_loss))
        break
  
  else:
    epoch = 1
    while True:
      logging.info("epoch %d" % epoch)
      loss, smooth_loss = train(data)
      np.savez(args.save, wxh=Wxh, whh=Whh, why=Why, bh=bh, by=by)
      epoch += 1

      if args.lossthreshold is not None and smooth_loss <= args.lossthreshold:
        logging.info('loss: %f, smooth loss: %f, exit' % (loss, smooth_loss))
        break

elif args.generate:
  start_ch = random.randrange(0, vocab_size - 1) 
  hprev = np.zeros((hidden_size, 1))
  sample_ix = sample(hprev, start_ch, args.gensize)
  sample_char = [ix_to_char[ix] for ix in sample_ix]
  txt = ''.join(sample_char)
  logging.info("%s" % txt)

elif args.validate:
  if args.byline:
    low_match_rate_list = []
    high_avg_loss_list = []
    with io.open(args.validatefile, mode = 'r', encoding = 'utf-8') as infile:
      for line in infile:
        input_line = [char_to_ix[ch] for ch in line]
        avg_loss, match_rate = validate(input_line)

        if match_rate is not None and match_rate < 40.0:
          low_match_rate_list.append((match_rate, avg_loss, smooth_loss, line))

        if avg_loss is not None and avg_loss > 10.0:
          high_avg_loss_list.append((match_rate, avg_loss, smooth_loss, line))
        
        if match_rate is not None:
          logging.info("Accuracy %f%%, avg loss %f, smooth_loss %f: %s" % (match_rate, avg_loss, smooth_loss, line))
    low_match_rate_list_sorted = sorted(low_match_rate_list, key=itemgetter(0)) 
    high_avg_loss_list_sorted = sorted(high_avg_loss_list, key=itemgetter(1), reverse=True) 
    #logging.info("----BY MATCH RATE----")
    for item in low_match_rate_list_sorted:
      pass#logging.warn("Low match rate: acc %f%%, avg loss %f, smooth_loss %f, %s" % (item[0], item[1], item[2], item[3]))   
    #logging.info("\n\n\n\n\n\n----BY AVG LOSS----")
    for item in high_avg_loss_list_sorted:
      pass#logging.warn("High avg loss: acc %f%%, avg loss %f, smooth_loss %f, %s" % (item[0], item[1], item[2], item[3]))
  else:
    inputs = [char_to_ix[ch] for ch in data]
    avg_loss, match_rate = validate(inputs)
    if match_rate is not None:
      logging.info("Accuracy %f%%, avg loss %f, smooth_loss %f" % (match_rate, avg_loss, smooth_loss))

