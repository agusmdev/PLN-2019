# https://docs.python.org/3/library/unittest.html
from unittest import TestCase

from tagging.classifier import feature_dict


class TestFeatureDict(TestCase):

    def test_feature_dict(self):
        sent = 'El gato come pescado .'.split()

        fdict = {
            'w': 'el',    # lower
            'wu': False,  # isupper
            'wt': True,   # istitle
            'wd': False,  # isdigit
            'nw': 'gato',
            'nwu': False,
            'nwt': False,
            'nwd': False,
            'pw': '<s>',
            'pwu': False,
            'pwt': False,
            'pwd': False,
        }

        self.assertEqual(feature_dict(sent, 0), fdict)

    def test_feature_dict2(self):
        sent = 'El Gato come pescado .'.split()

        fdict = {
            'w': 'el',    # lower
            'wu': False,  # isupper
            'wt': True,   # istitle
            'wd': False,  # isdigit
            'pw': '<s>',
            'nw': 'gato',
            'nwu': False,
            'nwt': True,
            'nwd': False,
            'pwu': False,
            'pwt': False,
            'pwd': False,
        }

        self.assertEqual(feature_dict(sent, 0), fdict)

    def test_feature_dict3(self):
        sent = 'El Gato come pescado .'.split()

        fdict = {
            'w': '.',    # lower
            'wu': False,  # isupper
            'wt': False,   # istitle
            'wd': False,  # isdigit
            'pw': 'pescado',
            'nw': '</s>',
            'nwu': False,
            'nwt': False,
            'nwd': False,
            'pwu': False,
            'pwt': False,
            'pwd': False,
        }

        self.assertEqual(feature_dict(sent, 4), fdict)

    def test_feature_dict_n_exception(self):
        sent = 'El Gato come pescado .'.split()
        self.assertRaises(IndexError, feature_dict, sent, 1, len(sent))
        self.assertRaises(IndexError, feature_dict, sent, 1, 0)

    def test_feature_dict_i_exception(self):
        sent = 'El Gato come pescado .'.split()
        self.assertRaises(IndexError, feature_dict, sent, len(sent), 3)
