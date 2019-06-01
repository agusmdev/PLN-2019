"""Evaulate a tagger.

Usage:
  eval.py -i <file> [-c]
  eval.py -h | --help

Options:

  -c            Show confusion matrix.
  -i <file>     Tagging model file.
  -h --help     Show this screen.
"""
from docopt import docopt
import pickle
import dill
import sys
from collections import defaultdict, Counter

from tagging.ancora import SimpleAncoraCorpusReader
import numpy as np
from collections import defaultdict
from sklearn.metrics import confusion_matrix
import numpy as np
from tqdm import tqdm
from tabulate import tabulate


class Evaluator:

    def __init__(self, sents, model):
        self.sents = sents
        self.model = model
        self.compute_analysis()

    def compute_analysis(self):
        hits, total, known_tags, unknown_tags = 0, 0, [], []
        y_true, y_pred, labels = [], [], []
        hits_known, hits_unknown = 0, 0

        for sent in tqdm(self.sents):
            word_sent, true_tags = zip(*sent)
            pred_tags = model.tag(word_sent)
            labels += true_tags

            hits += sum(mt == tt for mt, tt in zip(pred_tags, true_tags))
            known_tags += [not model.unknown(word) for word in word_sent]
            unknown_tags += [model.unknown(word) for word in word_sent]
            total += len(sent)

            y_true += [t for p, t in zip(pred_tags, true_tags)]
            y_pred += [p for p, t in zip(pred_tags, true_tags)]

        labels = list(set(labels))

        hits_unknown = sum(unk for yp, yt, unk in zip(y_pred, y_true, unknown_tags)
                           if yp == yt)
        hits_known = sum(kn for yp, yt, kn in zip(y_pred, y_true, known_tags)
                         if yp == yt)

        self.known_acc = (hits_known / sum(known_tags))*100
        self.unknown_acc = (hits_unknown / sum(unknown_tags))*100
        self.global_acc = float(hits) / total * 100

        top_labels = list(dict(Counter(y_pred).most_common(10)).keys())
        self.top_labels = top_labels

        ptag, ttag = zip(*[x for x in zip(y_pred, y_true)
                         if all(xi in top_labels for xi in x)])
        self.CM10 = confusion_matrix(ptag, ttag, top_labels)
        self.CM10_P = np.round((self.CM10/total*100), decimals=2)

    def print_results(self):
        print()
        print("Accuracy: {:2.2f}%".format(self.global_acc))
        print("Accuracy for known words: {:2.2f}%".format(self.known_acc))
        print("Accuracy for unknown words: {:2.2f}%".format(self.unknown_acc))
        print()

    def print_confusion_matrix(self):
        labels = self.top_labels
        table = self.CM10_P

        table = [list(map(str, list(row))) for row in table]
        table = [[labels[i]] + row for i, row in enumerate(table)]
        print(tabulate(table, labels, tablefmt='github'))


if __name__ == '__main__':
    opts = docopt(__doc__)

    # load the model
    filename = opts['-i']
    f = open(filename, 'rb')
    # model = dill.load(f)
    model = pickle.load(f)
    f.close()

    # load the data
    files = '3LB-CAST/.*\.tbf\.xml'
    corpus = SimpleAncoraCorpusReader('ancora/ancora-3.0.1es/', files)
    sents = list(corpus.tagged_sents())

    ev = Evaluator(sents, model)
    ev.print_results()
    ev.print_confusion_matrix()
