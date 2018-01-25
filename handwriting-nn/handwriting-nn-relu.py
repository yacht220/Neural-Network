import numpy as np
import logging
import argparse
import random
import hackthon_data as hd

parser = argparse.ArgumentParser()
parser.add_argument("--traindata", type=str, help="Training data")
parser.add_argument("--trainsize", type=int, help="Training data size")
parser.add_argument("--evaldata", type=str, help="Evaluation data")
parser.add_argument("--evalsize", type=int, help="Evaluation data size")
parser.add_argument("--save", type=str, help="Model (parameters) to save")
parser.add_argument("--train", action="store_true", help="Trainig")
parser.add_argument("--load", type=str, help="Load trained model (parameters)")
parser.add_argument("--evaluate", action="store_true", help="Evaluate model")
parser.add_argument("--lossthreshold", type=int, default="10", help="Exit training when reaching loss threshold. -1 no exit")
parser.add_argument("--logfile", type=str, help="Log file")
args = parser.parse_args()

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', datefmt='%Y-%m-%d %H:%M:%S', filename=args.logfile, filemode='w', level=logging.INFO)

input_size = 14
hidden1_size = 120 
hidden2_size = 60
output_size = 10
learning_rate = 1e-2
reg_lambda = 1e-2
input_scale = 1e-1

if args.load is None:
  Wxh1 = np.random.randn(hidden1_size, input_size) / np.sqrt(hidden1_size) # input to hidden
  Wh1h2 = np.random.randn(hidden2_size, hidden1_size) / np.sqrt(hidden2_size) # hidden to hidden
  Wh2y = np.random.randn(output_size, hidden2_size) / np.sqrt(output_size) # hidden to output
  bxh1 = np.zeros((hidden1_size, 1)) # hidden bias
  bh1h2 = np.zeros((hidden2_size, 1))
  bh2y = np.zeros((output_size, 1)) # output bias
else:
  W = np.load(args.load)
  Wxh1 = W['arr_0'].T
  bxh1 = W['arr_1'].T
  Wh1h2 = W['arr_2'].T
  bh1h2 = W['arr_3'].T
  Wh2y = W['arr_4'].T
  bh2y = W['arr_5'].T

def backpropagation(inputs, targets):
  #forward pass
  xs = inputs # input_size * batch_size
  h1s = np.maximum((np.dot(Wxh1, xs) + bxh1), 0) # hidden state, hidden1_size * batch_size
  #h1s = np.tanh(np.dot(Wxh1, xs) + bxh1) # hidden1_size * batch_size
  h2s = np.maximum((np.dot(Wh1h2, h1s) + bh1h2), 0) # hidden2_size * batch_size
  #h2s = np.tanh(np.dot(Wh1h2, h1s) + bh1h2) # hidden_2_size * batch_size
  ys = np.dot(Wh2y, h2s) + bh2y # output_size * batch_size
  ys = ys.astype(np.float128)
  
  # softmax
  logits_exp = np.exp(ys)
  ps = logits_exp / np.sum(logits_exp, axis = 0, keepdims=True)
    
  # cross-entropy loss
  log_loss = np.zeros(batch_size)
  for i in xrange(batch_size):
    log_loss[i] = -np.log(ps[targets[i], i])
  loss = np.sum(log_loss)
  loss += reg_lambda / 2 * (np.sum(np.square(Wxh1)) + np.sum(np.square(Wh1h2)) + np.sum(np.square(Wh2y)))
  loss = 1.0 / batch_size * loss
    
  # backward pass
  dWxh1, dWh1h2, dWh2y = np.zeros_like(Wxh1), np.zeros_like(Wh1h2), np.zeros_like(Wh2y)
  dbxh1, dbh1h2, dbh2y = np.zeros_like(bxh1), np.zeros_like(bh1h2), np.zeros_like(bh2y)
  dy = np.copy(ps) # output_size * batch_size
  for i in xrange(batch_size):
    dy[targets[i], i] -= 1 
  dy /= dy.shape[1] 
  dWh2y = np.dot(dy, h2s.T) # output_size * hidden2_size
  dbh2y = np.sum(dy, axis=1, keepdims=True) # output_size * 1
 
  #dh2 = (1 - h2s * h2s) * np.dot(Wh2y.T, dy) # hidden2_size * batch_size 
  dh2 = np.dot(Wh2y.T, dy)
  dh2[h2s <= 0] = 0
  dWh1h2 = np.dot(dh2, h1s.T) # hidden2_size * hidden1_size 
  dbh1h2 = np.sum(dh2, axis=1, keepdims=True) # hidden2_size * 1

  #dh1 = (1 - h1s * h1s) * np.dot(Wh1h2.T, dh2) # hidden1_size * batch_size
  dh1 = np.dot(Wh1h2.T, dh2)
  dh1[h1s <= 0] = 0
  dWxh1 = np.dot(dh1, xs.T) # hidden1_size * input_size
  dbxh1 = np.sum(dh1, axis=1, keepdims=True) # hidden1_size * 1 

  '''for dparam in [dWxh1, dWh1h2, dWh2y, dbxh1, dbh1h2, dbh2y]:
    np.clip(dparam, -5, 5, out=dparam) # clip to mitigate exploding gradients'''

  '''for param, dparam in zip([Wxh1, Wh1h2, Wh2y],
                           [dWxh1, dWh1h2, dWh2y]):
    dparam += reg_lambda * param'''
  
  for param, dparam in zip([Wxh1, Wh1h2, Wh2y, bxh1, bh1h2, bh2y], 
                           [dWxh1, dWh1h2, dWh2y, dbxh1, dbh1h2, dbh2y]):
    param += -learning_rate * dparam

  return loss

def evaluate(inputs, targets):
  xs = inputs 
  h1s = np.maximum((np.dot(Wxh1, xs) + bxh1), 0) 
  #h1s = np.tanh(np.dot(Wxh1, xs) + bxh1) 
  h2s = np.maximum((np.dot(Wh1h2, h1s) + bh1h2), 0) 
  #h2s = np.tanh(np.dot(Wh1h2, h1s) + bh1h2)
  ys = np.dot(Wh2y, h2s) + bh2y 
  #ps = np.exp(ys) / np.sum(np.exp(ys), axis = 0, keepdims=True) 

  if np.argmax(ys) == targets[0]:
    return 1
  else:
    return 0

if args.train:
  train_size = args.trainsize
  batch_size = 10
  epoch_size = train_size / batch_size
  train_data = hd.hackthon_data()
  train_data.loaddata(args.traindata)
  epoch = 0
  while True:
    epoch += 1
    logging.info('epoch %d' % (epoch))
    for i in xrange(epoch_size):
      labels, vectors = train_data.getdata(batch_size)
      loss = backpropagation(vectors.T * input_scale, labels)
  
    logging.info('loss: %f' % (loss)) 
    
    if epoch % 10 == 0:
      logging.info('Evaluating...')
      eval_size = args.evalsize
      eval_data = hd.hackthon_data()
      eval_data.loaddata(args.evaldata)
      result = 0
      for i in xrange(eval_size):
        eval_labels, eval_vectors = eval_data.getdata(1)
        result += evaluate(eval_vectors.T * input_scale, eval_labels)
      
      logging.info("Accuracy: %f%%" % (float(result) / float(eval_size) * 100.0))
      np.savez(args.save, Wxh1.T, bxh1.T, Wh1h2.T, bh1h2.T, Wh2y.T, bh2y.T)

elif args.evaluate:
  eval_size = args.evalsize
  eval_data = hd.hackthon_data()
  eval_data.loaddata(args.evaldata)
  result = 0
  for i in xrange(eval_size):
    eval_labels, eval_vectors = eval_data.getdata(1)
    result += evaluate(eval_vectors.T * input_scale, eval_labels)
      
  logging.info("Accuracy: %f%%" % (float(result) / float(eval_size) * 100.0))
