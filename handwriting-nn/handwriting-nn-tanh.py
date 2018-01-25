"""
Minimal character-level Vanilla RNN model. Written by Andrej Karpathy (@karpathy)
BSD License
"""
import numpy as np
import logging
import argparse
import random
import hackthon_data as hd
import pdb

parser = argparse.ArgumentParser()
#parser.add_argument("-u", "--utf8", action="store_true", help="Input is UTF8 encoded")
parser.add_argument("-i", "--input", type=str, help="Text file as corpus")
parser.add_argument("-s", "--save", type=str, help="Model (parameters) to save")
parser.add_argument("-t", "--train", action="store_true", help="Trainig")
parser.add_argument("-l", "--load", type=str, help="Load trained model (parameters)")
parser.add_argument("-w", "--esize", type=int, default="200", help="Evaluation text size")
parser.add_argument("-e", "--evaluate", action="store_true", help="Evaluate model")
parser.add_argument("-r", "--lossthreshold", type=int, default="10", help="Exit training when reaching loss threshold. -1 no exit")
parser.add_argument("-g", "--logfile", type=str, help="Log file")
args = parser.parse_args()

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', datefmt='%Y-%m-%d %H:%M:%S', filename=args.logfile, filemode='w', level=logging.INFO)

# data I/O
#if args.utf8:
#data = open(args.input, 'r').read().decode("UTF-8") # should be simple plain text file
#else:
#data = open(args.input, 'r').read() # should be simple plain text file

#data = data.split()
#chars = list(set(data))

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
input_size = 14
hidden1_size = 120 # size of hidden layer of neurons
hidden2_size = 60
output_size = 10
#seq_length = 25 # number of steps to unroll the RNN for
learning_rate = 1e-2
reg_lambda = 1e-2

# model parameters
if args.load is None:
  Wxh1 = np.random.randn(hidden1_size, input_size) / np.sqrt(hidden1_size) # input to hidden
  Wh1h2 = np.random.randn(hidden2_size, hidden1_size) / np.sqrt(hidden2_size) # hidden to hidden
  Wh2y = np.random.randn(output_size, hidden2_size) / np.sqrt(output_size) # hidden to output
  #Wxh1 = np.random.normal(0, 1, [hidden1_size, input_size])
  #Wh1h2 = np.random.normal(0, 1, [hidden2_size, hidden1_size])
  #Wh2y = np.random.normal(0, 1, [output_size, hidden2_size])
  bxh1 = np.zeros((hidden1_size, 1)) # hidden bias
  bh1h2 = np.zeros((hidden2_size, 1))
  bh2y = np.zeros((output_size, 1)) # output bias
else:
  W = np.load(args.load)
  Wxh1 = W['wxh1']
  Wh1h2 = W['wh1h2']
  Wh2y = W['wh2y']
  bxh1 = W['bxh1']
  bh1h2 = W['bh1h2']
  bh2y = W['bh2y']

def backpropagation(inputs, targets):
  """
  inputs,targets are both list of integers.
  hprev is Hx1 array of initial hidden state
  returns the loss, gradients on model parameters, and last hidden state
  """
  #hs[-1] = np.copy(hprev)
  #loss = 0
  # forward pass
  #for t in xrange(len(inputs)):
  #xs = np.zeros((input_size, 1)) # encode in 1-of-k representation
  xs = inputs # input_size * batch_size
  #print("xs", xs.shape)
  #h1s = np.maximum((np.dot(Wxh1, xs) + bxh1), 0) # hidden state, hidden1_size * 1
  h1s = np.tanh(np.dot(Wxh1, xs) + bxh1) # hidden1_size * batch_size
  #print("h1s", h1s.shape)
  #h2s = np.maximum((np.dot(Wh1h2, h1s) + bh1h2), 0) # hidden2_size * 1
  h2s = np.tanh(np.dot(Wh1h2, h1s) + bh1h2) # hidden_2_size * batch_size
  #print("h2s", h2s.shape)
  ys = np.dot(Wh2y, h2s) + bh2y # output_size * batch_size
  #print("ys", ys.shape, ys)
  ps = np.exp(ys) / np.sum(np.exp(ys), axis = 0, keepdims=True) # probabilities for next chars, output_size * batch_size
  #print("ps", ps.shape, ps)
  #print("ps", ps)
    
  # cross-entropy loss
  log_loss = np.zeros(batch_size)
  #print("log_loss", log_loss.shape)
  for i in xrange(batch_size):
    log_loss[i] = -np.log(ps[targets[i], i])
  #print(log_loss)
  loss = np.sum(log_loss)
  #loss += reg_lambda / 2 * (np.sum(np.square(Wxh1)) + np.sum(np.square(Wh1h2)) + np.sum(np.square(Wh2y)))
  loss = 1.0 / batch_size * loss
    
  #print("loss", loss)
  
  # backward pass: compute gradients going backwards
  dWxh1, dWh1h2, dWh2y = np.zeros_like(Wxh1), np.zeros_like(Wh1h2), np.zeros_like(Wh2y)
  dbxh1, dbh1h2, dbh2y = np.zeros_like(bxh1), np.zeros_like(bh1h2), np.zeros_like(bh2y)
  #dhnext = np.zeros_like(hs[0])
  #for t in reversed(xrange(len(inputs))):
  dy = np.copy(ps) # output_size * batch_size
  for i in xrange(batch_size):
    dy[targets[i], i] -= 1 # backprop into y. see http://cs231n.github.io/neural-networks-case-study/#grad if confused here
  #print("dy", dy.shape, dy)
  dWh2y = np.dot(dy, h2s.T) # output_size * hidden2_size
  #print("dWh2y", dWh2y.shape)
  dbh2y = np.sum(dy, axis=1, keepdims=True) # output_size * 1
  #print("dbh2y", dbh2y.shape)
 
  dh2 = (1 - h2s * h2s) * np.dot(Wh2y.T, dy) # backprop through tanh nonlinearity, hidden2_size * batch_size 
  #print("dh2", dh2.shape)
  dWh1h2 = np.dot(dh2, h1s.T) # hidden2_size * hidden1_size 
  #print("dWh1h2", dWh1h2.shape)
  dbh1h2 = np.sum(dh2, axis=1, keepdims=True) # hidden2_size * 1
  #print("dbh1h2", dbh1h2.shape)

  dh1 = (1 - h1s * h1s) * np.dot(Wh1h2.T, dh2) # hidden1_size * batch_size
  #print("dh1", dh1.shape)
  dWxh1 = np.dot(dh1, xs.T) # hidden1_size * input_size
  #print("dWxh1", dWxh1.shape)
  dbxh1 = np.sum(dh1, axis=1, keepdims=True) # hidden1_size * 1 
  #print("dbxh1", dbxh1.shape)

  '''for dparam in [dWxh1, dWh1h2, dWh2y, dbxh1, dbh1h2, dbh2y]:
    np.clip(dparam, -5, 5, out=dparam) # clip to mitigate exploding gradients'''

  '''for param, dparam in zip([Wxh1, Wh1h2, Wh2y],
                           [dWxh1, dWh1h2, dWh2y]):
    dparam += reg_lambda * param'''
  
  for param, dparam in zip([Wxh1, Wh1h2, Wh2y, bxh1, bh1h2, bh2y], 
                           [dWxh1, dWh1h2, dWh2y, dbxh1, dbh1h2, dbh2y]):
    param += -learning_rate * dparam

  #print(loss, dWxh1, dWh1h2, dWh2y, dbxh1, dbh1h2, dbh2y)
  return loss

def evaluate(inputs, targets):
  xs = inputs # input_size * 1
  #print("xs", xs.shape)
  #h1s = np.maximum((np.dot(Wxh1, xs) + bxh1), 0) # hidden state, hidden1_size * 1
  h1s = np.tanh(np.dot(Wxh1, xs) + bxh1) # hidden1_size * 1
  #print("h1s", h1s.shape)
  #h2s = np.maximum((np.dot(Wh1h2, h1s) + bh1h2), 0) # hidden2_size * 1
  h2s = np.tanh(np.dot(Wh1h2, h1s) + bh1h2) # hidden_2_size * 1
  #print("h2s", h2s.shape)
  ys = np.dot(Wh2y, h2s) + bh2y # output_size * 1
  #print("ys", ys.shape, ys)
  #ps = np.exp(ys) / np.sum(np.exp(ys), axis = 0, keepdims=True) # probabilities for next chars, output_size * 1

  #print(np.argmax(ps), targets[0])
  if np.argmax(ys) == targets[0]:
    return 1
  else:
    return 0


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
      # rel_error should be on order of 1e-7 or less'''

if args.train:
  train_size = 6999
  batch_size = 10
  epoch_size = train_size / batch_size
  train_data = hd.hackthon_data()
  train_data.loaddata("Train.dat")
  epoch = 0
  while True:
    epoch += 1
    logging.info('epoch %d starts' % (epoch))
    for i in xrange(epoch_size):
      labels, vectors = train_data.getdata(batch_size)
      loss = backpropagation(vectors.T, labels)
    '''print "oldhprev", oldhprev, "hprev", hprev
    gradCheck(inputs, targets, oldhprev)
      smooth_loss = smooth_loss * 0.999 + loss * 0.001
      if i % 100 == 0: 
        logging.info('iter %d, loss: %f, smooth loss: %f' % (i, loss, smooth_loss)) #print 'iter %d, loss: %f' % (n, smooth_loss) # print progress
        #np.savez(args.save, wxh=Wxh1, wh1h2=Wh1h2, wh2y=Wh2y, bxh1=bxh1, bh1h2=bh1h2, bh2y=bh2y)
        if corpus_complete == True and args.lossthreshold != -1 and smooth_loss <= args.lossthreshold:
          logging.info('zhouy22: loss: %f, smooth loss: %f, exit' % (loss, smooth_loss))
          break'''
  
    logging.info('loss: %f' % (loss)) #print 'iter %d, loss: %f' % (n, smooth_loss) # print progress
    
    if epoch % 10 == 0:
      logging.info('Evaluating...')
      eval_size = 1095
      eval_data = hd.hackthon_data()
      eval_data.loaddata("test.dat")
      result = 0
      for i in xrange(eval_size):
        eval_labels, eval_vectors = eval_data.getdata(1)
        result += evaluate(eval_vectors.T, eval_labels)
      
      logging.info("Accuracy: %f%%" % (float(result) / float(eval_size) * 100.0))
      np.savez(args.save, wxh1=Wxh1, wh1h2=Wh1h2, wh2y=Wh2y, bxh1=bxh1, bh1h2=bh1h2, bh2y=bh2y)

        



elif args.evaluate:
  eval_size = 1095
  eval_data = hd.hackthon_data()
  eval_data.loaddata("test.dat")
  result = 0
  for i in xrange(eval_size):
    eval_labels, eval_vectors = eval_data.getdata(1)
    result += evaluate(eval_vectors.T, eval_labels)
      
  logging.info("Accuracy: %f%%" % (float(result) / float(eval_size) * 100.0))
