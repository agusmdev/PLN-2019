"""Evaulate a language model using a test set.

Usage:
  eval.py -i <file>
  eval.py -h | --help

Options:
  -i <file>     Language model file.
  -h --help     Show this screen.
"""
from docopt import docopt
import pickle
import math

from languagemodeling.ngram import NGram
from nltk.corpus.reader.plaintext import PlaintextCorpusReader


if __name__ == '__main__':
    opts = docopt(__doc__)

    # load the model
    filename = opts['-i']
    f = open(filename, 'rb')
    model = pickle.load(f)
    f.close()

    # load the data
    # WORK HERE!! LOAD YOUR EVALUATION CORPUS
    sents = PlaintextCorpusReader(".", "test.txt").sents()

    # compute the cross entropy
    # WORK HERE!!
    log_prob = model.log_prob(sents)
    e = model.cross_entropy(sents)
    p = model.perplexity(sents)

    print('Log probability: {}'.format(log_prob))
    print('Cross entropy: {}'.format(e))
    print('Perplexity: {}'.format(p))
