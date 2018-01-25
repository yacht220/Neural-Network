#!/usr/bin/python3
"""
Originated from minimal character-level Vanilla RNN model. Written by Andrej Karpathy (@karpathy)
BSD License

Revised by Yizhou Zhou (@yacht220) for study purpose.
"""
import numpy as np
import logging
import argparse
import random
from operator import itemgetter
import sys
import os
from jarvis.logbase import avamarlog
import re
import pdb
import multiprocessing
import time
import json

test_is_on = True

class RnnAvlog():
    def __init__(self, corpus, loadmodel = None, lossthreshold = None):
        self._loadCorpus(corpus)
        self._reset(lossthreshold)
        if loadmodel is not None:
            self._loadModel(loadmodel)
        
    def _reset(self, lossthreshold = None):
        self.loss_threshold = lossthreshold
        # hyperparameters
        self.hidden_size = 100 # size of hidden layer of neurons
        # number of steps to unroll the RNN for
        self.seq_length = 5
        self.learning_rate = 1e-1
        
        self.Wxh = np.random.randn(self.hidden_size, self.vocab_size)*0.01 # input to hidden
        self.Whh = np.random.randn(self.hidden_size, self.hidden_size)*0.01 # hidden to hidden
        self.Why = np.random.randn(self.vocab_size, self.hidden_size)*0.01 # hidden to output
        #self.Wxh = np.random.rand(self.hidden_size, self.vocab_size) # input to hidden
        #self.Whh = np.random.rand(self.hidden_size, self.hidden_size) # hidden to hidden
        #self.Why = np.random.rand(self.vocab_size, self.hidden_size) # hidden to output
        #self.Wxh = np.random.uniform(low = -1, high = 1, size = (self.hidden_size, self.vocab_size)) # input to hidden
        #self.Whh = np.random.uniform(low = -1, high = 1, size = (self.hidden_size, self.hidden_size)) # hidden to hidden
        #self.Why = np.random.uniform(low = -1, high = 1, size = (self.vocab_size, self.hidden_size)) # hidden to output

        self.bh = np.zeros((self.hidden_size, 1)) # hidden bias
        self.by = np.zeros((self.vocab_size, 1)) # output bias

        self.mWxh, self.mWhh, self.mWhy = np.zeros_like(self.Wxh), np.zeros_like(self.Whh), np.zeros_like(self.Why)
        self.mbh, self.mby = np.zeros_like(self.bh), np.zeros_like(self.by) # memory variables for Adagrad
        self.smooth_loss = -np.log(1.0/self.vocab_size)*self.seq_length # loss at iteration 0
    
    def _loadModel(self, model):
        W = np.load(model)
        self.Wxh = W['wxh']
        self.Whh = W['whh']
        self.Why = W['why']
        self.bh = W['bh']
        self.by = W['by']

    def _loadCorpus(self, corpus):
        self.corpus = corpus
        galp = avamarlog.AvamarLogParser()
        galp.setprefix('')
        galp.open(self.corpus)
        msgreader = avamarlog.AllMsgReader(parser=galp, removeuseless=True)
        data = msgreader.getallmsg()
        chars = sorted(list(set(data)))

        data_size, self.vocab_size = len(data), len(chars)
        logging.info('data has %d characters, %d unique.' % (data_size, self.vocab_size))
        self.char_to_ix = { ch:i for i,ch in enumerate(chars) }
        self.ix_to_char = { i:ch for i,ch in enumerate(chars) }

    def train(self, savemodel):
        epoch = 1
        talp = avamarlog.AvamarLogParser()
        talp.setprefix('')
        talp.open(self.corpus)
        logiter = avamarlog.MsgIterator(parser=talp, removeuseless=True)
        min_smooth_loss = 100
        while True:
            logging.info("epoch %d" %epoch)
            linecounter = 0
            for line, _, _ in logiter:
                linecounter += 1
                loss = self._backpropagation(line)
                if linecounter % 100  == 0:
                    logging.info('smooth loss: %f' % (self.smooth_loss)) #print 'iter %d, loss: %f' % (n, smooth_loss) # print progress
                    if self.smooth_loss < min_smooth_loss:
                        min_smooth_loss = self.smooth_loss
                        logging.info("Saving model...")
                        np.savez(savemodel, wxh=self.Wxh, whh=self.Whh, why=self.Why, bh=self.bh, by=self.by)
 
            talp.reset()
            epoch += 1

            if self.loss_threshold is not None and self.smooth_loss <= self.loss_threshold:
                logging.info('loss: %f, smooth loss: %f, exit' % (loss, self.smooth_loss))
                break

    def _backpropagation(self, text):
        loss = 0
        n = 0
        p = 0 # go from start of text
        hprev = np.zeros((self.hidden_size, 1)) # reset RNN memory
        show_count = 5
  
        # prepare inputs (we're sweeping from left to right in steps seq_length long)
        while p+self.seq_length+1 < len(text):
            inputs = [self.char_to_ix[ch] for ch in text[p:p+self.seq_length]]
            targets = [self.char_to_ix[ch] for ch in text[p+1:p+self.seq_length+1]]

            # forward seq_length characters through the net and fetch gradient
            loss, dWxh, dWhh, dWhy, dbh, dby, hprev = self._lossFun(inputs, targets, hprev)
            self.smooth_loss = self.smooth_loss * 0.999 + loss * 0.001

            # sample from the model now and then
            #if n % show_count == 0: 
                #sample_ix = self._sample(hprev, inputs[0], 200)
                #sample_char = [self.ix_to_char[ix] for ix in sample_ix]
                #txt = ''.join(sample_char)
                #logging.info('----\n %s \n----' % (txt))
                #logging.info('iter %d, loss: %f, smooth loss: %f' % (n, loss, self.smooth_loss)) #print 'iter %d, loss: %f' % (n, smooth_loss) # print progress
          
            # perform parameter update with Adagrad
            for param, dparam, mem in zip([self.Wxh, self.Whh, self.Why, self.bh, self.by], 
                                          [dWxh, dWhh, dWhy, dbh, dby], 
                                          [self.mWxh, self.mWhh, self.mWhy, self.mbh, self.mby]):
                mem += dparam * dparam
                param += -self.learning_rate * dparam / np.sqrt(mem + 1e-8) # adagrad update

            p += self.seq_length # move text pointer
            n += 1 # iteration counter

        return loss

    def _lossFun(self, inputs, targets, hprev):
        """
        inputs,targets are both list of integers.
        hprev is Hx1 array of initial hidden state
        returns the loss, gradients on model parameters, and last hidden state
        """
        xs, hs, ys, ps = {}, {}, {}, {}
        hs[-1] = np.copy(hprev)
        loss = 0
        # forward pass
        for t in range(len(inputs)):
            xs[t] = np.zeros((self.vocab_size,1)) # encode in 1-of-k representation
            xs[t][inputs[t]] = 1
            hs[t] = np.tanh(np.dot(self.Wxh, xs[t]) + np.dot(self.Whh, hs[t-1]) + self.bh) # hidden state
            #hs[t] = np.maximum((np.dot(self.Wxh, xs[t]) + np.dot(self.Whh, hs[t-1]) + self.bh), 0) 
            ys[t] = np.dot(self.Why, hs[t]) + self.by # unnormalized log probabilities for next chars
            #ys[t] = self._norm(ys[t])
            #ys[t] = ys[t].astype(np.float128)
            ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t])) # probabilities for next chars
            loss += -np.log(ps[t][targets[t],0]) # softmax (cross-entropy loss)
        # backward pass: compute gradients going backwards
        dWxh, dWhh, dWhy = np.zeros_like(self.Wxh), np.zeros_like(self.Whh), np.zeros_like(self.Why)
        dbh, dby = np.zeros_like(self.bh), np.zeros_like(self.by)
        dhnext = np.zeros_like(hs[0])
        for t in reversed(range(len(inputs))):
            dy = np.copy(ps[t])
            dy[targets[t]] -= 1 # backprop into y. see http://cs231n.github.io/neural-networks-case-study/#grad if confused here
            dWhy += np.dot(dy, hs[t].T)
            dby += dy
            dh = np.dot(self.Why.T, dy) + dhnext # backprop into h
            dhraw = (1 - hs[t] * hs[t]) * dh # backprop through tanh nonlinearity
            dbh += dhraw
            dWxh += np.dot(dhraw, xs[t].T)
            dWhh += np.dot(dhraw, hs[t-1].T)
            dhnext = np.dot(self.Whh.T, dhraw)
        for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
            np.clip(dparam, -5, 5, out=dparam) # clip to mitigate exploding gradients
        return loss, dWxh, dWhh, dWhy, dbh, dby, hs[len(inputs)-1]

    def _norm(self, matrix):
        mmin, mmax = matrix.min(), matrix.max()
        matrix = (matrix - mmin) / (mmax - mmin)
        return matrix

    def _sample(self, h, seed_ix, n):
        """ 
        sample a sequence of integers from the model 
        h is memory state, seed_ix is seed letter for first time step
        """
        x = np.zeros((self.vocab_size, 1))
        x[seed_ix] = 1
        ixes = []
        for t in range(n):
            h = np.tanh(np.dot(self.Wxh, x) + np.dot(self.Whh, h) + self.bh)
            y = np.dot(self.Why, h) + self.by
            p = np.exp(y) / np.sum(np.exp(y))
            ix = np.random.choice(range(self.vocab_size), p=p.ravel())
            x = np.zeros((self.vocab_size, 1))
            x[ix] = 1
            ixes.append(ix)
        return ixes

    def generate(self, gensize = 200):
        start_ch = random.randrange(0, self.vocab_size - 1) 
        hprev = np.zeros((self.hidden_size, 1))
        sample_ix = self._sample(hprev, start_ch, gensize)
        sample_char = [self.ix_to_char[ix] for ix in sample_ix]
        txt = ''.join(sample_char)
        logging.info("%s" % txt)

    def _validate(self, inputs):
        loss = 0
        avg_loss = 0
        h = np.zeros((self.hidden_size, 1))
        len_line = len(inputs)
        if len_line <= 1:
            return None, None
        match = 0
        show_count = 5
        loss_add_count = 0
        for i in range(len_line - 1):
            x = np.zeros((self.vocab_size, 1))
            x[inputs[i]] = 1
            h = np.tanh(np.dot(self.Wxh, x) + np.dot(self.Whh, h) + self.bh)
            y = np.dot(self.Why, h) + self.by
            p = np.exp(y) / np.sum(np.exp(y))
            loss += -np.log(p[inputs[i + 1], 0]) # softmax (cross-entropy loss)

            if i % self.seq_length == 0:
                #self.smooth_loss = self.smooth_loss * 0.999 + loss * 0.001
                if (i / self.seq_length) % show_count == 0:
                    pass #logging.info("iter %d, loss %f, smooth_loss %f" % (i / self.seq_length, loss, self.smooth_loss))
                avg_loss += loss
                loss_add_count += 1
                loss = 0

            if p[inputs[i + 1]] > 5e-1: #p[ix] == p[inputs[i + 1]]:
                match += 1

        avg_loss /= loss_add_count 
        match_rate = float(match) / float(len_line - 1) * 100.0

        return avg_loss, match_rate

    def _checkError(self, line):
        error = re.compile(r"Error <")
        fatal = re.compile(r"FATAL <")
        if error.search(line) is not None or fatal.search(line) is not None:
            return True
        else:
            return False

    def analyze(self, log, output = None):
        if output is not None:
            mout = avamarlog.MsgOutputter(output)
        malp = avamarlog.AvamarLogParser()
        if test_is_on:
            malp.setprefix('')
        malp.open(log)
        cpus = multiprocessing.cpu_count()
        #cpus = 1
        totalline = malp.calctotalline() 
        sizeperproc = 200
        tasks = int(totalline / sizeperproc)
        remainline = totalline % sizeperproc
        if remainline > 0:
            tasks += 1
        
        timestart = time.time()
        #os.system("taskset -p 0xff %d" % os.getpid())
        pool = multiprocessing.Pool(cpus)
        results = []
        nextstart = 1
        for i in range(tasks):
            if remainline > 0:
                result = pool.apply_async(func = self._analyzeSubproc, args = (log, nextstart, remainline))
                results.append(result)
                nextstart = nextstart + remainline
                remainline = 0
            else:
                result = pool.apply_async(func = self._analyzeSubproc, args = (log, nextstart, sizeperproc))
                results.append(result)
                nextstart = nextstart + sizeperproc

        pool.close()
        pool.join()
        timeend = time.time()
        elapse = timeend - timestart
        logging.info("elaspe: %s" % elapse)

        high_avg_loss_list = []
        error_list = []
        for result  in results:
            high_avg_loss_list.extend(result.get()[0])
            error_list.extend(result.get()[1])

        logging.info("\n\n\n\n\n\n----ERROR LOGS----")
        if output is not None:
            mout.outline("ERROR LOGS:\n")
        for item in error_list:
            logging.warn("line %s: %s" % (item[0], item[1]))
            if output is not None:
                mout.outline("[line %s] %s" % (item[0], item[1]))
        
        logging.info("\n\n\n\n\n\n----BY AVG LOSS----")
        if output is not None:
            mout.outline("\nSUSPICIOUS LOGS (MOST SUSPICIOUS SHOWS FIRST):\n")
            item_index = 0
        for item in sorted(high_avg_loss_list, key=itemgetter(1), reverse=True):
            logging.warn("High avg loss: acc %f%%, avg loss %f, line %s: %s" % (item[0], item[1], item[2], item[3]))
            if output is not None:
                item_index += 1
                mout.outline("[line %s] %s" % (item[2], item[3]))

    def _analyzeSubproc(self, log, start, size):
        #return start, size
        high_avg_loss_list = []
        error_list = []
        valp = avamarlog.AvamarLogParser()
        if test_is_on:
            valp.setprefix('')
        valp.open(log)
        logiter = avamarlog.MsgRangeIterator(startlinenum = start, size = size, parser = valp, removeuseless=True)
        lineindex = 0
        for line, rawline, linenum in logiter:
            lineindex += 1
            #if lineindex % 100 == 0:
                #logging.info("%d lines" % lineindex)
            if self._checkError(rawline):
                error_list.append((linenum, rawline))
            input_line = [self.char_to_ix[ch] for ch in line]
            avg_loss, match_rate = self._validate(input_line)
            if avg_loss is not None and avg_loss > 15:
                high_avg_loss_list.append((match_rate, avg_loss, linenum, rawline))

        return high_avg_loss_list, error_list

def analyze_suspicious_log(analyzelog, output):
    output_json = {}
    for log, tag in analyzelog:
        if tag == avamarlog.AvamarLogType.Avtar:
            print('#######################')
            logging.basicConfig(format='%(asctime)s %(process)d %(processName)s %(levelname)-8s %(message)s', datefmt='%Y-%m-%d %H:%M:%S', filename="suspiciousloganalyzer.log", filemode='w', level=logging.INFO)
            if test_is_on:
                rnnavlog = RnnAvlog(corpus = "avtar.log", loadmodel = "avtar.npz")
            else:
                rnnavlog = RnnAvlog(corpus = sys.path[0] + "/jarvis/suspicious_log_detector/avtar.log", loadmodel = sys.path[0] + "/jarvis/suspicious_log_detector/avtar.npz")
            analyzeout = "/".join(output.split("/")[:-1]) + "/" + log.split("/")[-1] + ".alz"
            rnnavlog.analyze(log = log, output = analyzeout)
            output_json[log] = analyzeout 

    avoutter = avamarlog.MsgOutputter(output)
    output_file = json.dumps(output_json)
    avoutter.outline(output_file)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, help="Text file as corpus")
    parser.add_argument("--save", type=str, help="Model (parameters) to save")
    parser.add_argument("--train", action="store_true", help="Trainig")
    parser.add_argument("--load", type=str, help="Load trained model (parameters)")
    parser.add_argument("--gensize", type=int, default="200", help="Generation text size")
    parser.add_argument("--generate", action="store_true", help="Generate text with current model")
    parser.add_argument("--lossthreshold", type=int, help="Exit training when reaching loss threshold. No exit if not set")
    parser.add_argument("--logfile", type=str, help="Log file")
    parser.add_argument("--analyze", action="store_true", help="Analyze log and extract suspicious lines")
    parser.add_argument("--analyzelogfile", type=str, help="Log file for analysis")
    args = parser.parse_args()

    logging.basicConfig(format='%(asctime)s %(process)d %(processName)s %(levelname) %(levelname)-8s %(message)s', datefmt='%Y-%m-%d %H:%M:%S', filename=args.logfile, filemode='w', level=logging.INFO)

    if args.train:
        if args.load is not None:
            modelpath = sys.path[0] + "/" + args.load
        else:
            modelpath = None
        rnnavlog = RnnAvlog(corpus = sys.path[0] + "/" + args.input, loadmodel = modelpath, lossthreshold = args.lossthreshold)
        rnnavlog.train(savemodel = args.save)

    elif args.generate:
        assert(args.load != None)
        rnnavlog = RnnAvlog(corpus = sys.path[0] + "/" + args.input, loadmodel = sys.path[0] + "/" + args.load)
        rnnavlog.generate(gensize = args.gensize)

    elif args.analyze:
        assert(args.load != None)
        rnnavlog = RnnAvlog(corpus = sys.path[0] + "/" + args.input, loadmodel = sys.path[0] + "/" + args.load)
        rnnavlog.analyze(log = args.analyzelogfile)

if __name__ == "__main__":
    main()
