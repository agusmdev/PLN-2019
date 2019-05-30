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

    def test_feature_dict_n_and_i(self):
        sent = 'El Gato come pescado .'.split()
        fdict1 = {
            'w': '.',
            'wu': False,
            'wt': False,
            'wd': False,
            'pw': 'pescado',
            'nw': '</s>',
            'pwu': False,
            'nwu': False,
            'pwt': False,
            'nwt': False,
            'pwd': False,
            'nwd': False
        }

        fdict2 = {'w': '</s>', 'wu': False, 'wt': False, 'wd': False}

        self.assertEqual(feature_dict(sent, 4, 200), fdict1)
        self.assertEqual(feature_dict([], 0, 0), fdict2)
