"""Train a sequence tagger.

Usage:
  train.py [options] -o <file>
  train.py -h | --help

Options:
  -m <model>    Model to use [default: badbase]:
                  badbase: Bad baseline
                  base: Baseline
  -n <int>
  -o <file>     Output model file.
  -h --help     Show this screen.
"""
from docopt import docopt
import pickle
import dill

from tagging.ancora import SimpleAncoraCorpusReader
from tagging.baseline import BaselineTagger, BadBaselineTagger
from tagging.hmm import MLHMM
from tagging.classifier import ClassifierTagger


models = {
    'badbase': BadBaselineTagger,
    'base': BaselineTagger,
    'mlhmm': MLHMM,
    'cltagg': ClassifierTagger
}


if __name__ == '__main__':
    opts = docopt(__doc__)

    # load the data
    files = 'CESS-CAST-(A|AA|P)/.*\.tbf\.xml'
    corpus = SimpleAncoraCorpusReader('ancora/ancora-3.0.1es/', files)
    sents = list(corpus.tagged_sents())

    # train the model
    model_class = models[opts['-m']]
    n = opts["-n"]
    if n:
        model = model_class(int(n), sents)
    else:
        model = model_class(sents)

    # save it
    filename = opts['-o']
    f = open(filename, 'wb')
    # pickle.dump(model, f)
    dill.dump(model, f)
    f.close()
