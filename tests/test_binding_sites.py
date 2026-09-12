#!/usr/bin/env python3

import unittest
from unittest.mock import patch

from collections import Counter

from sklearn.metrics import accuracy_score
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances

from src.binding_sites import (
    toint,
    get_features_and_labels,
    cluster_euclidean,
    cluster_hamming,
)


class TestToint(unittest.TestCase):

    def test_toint(self):
        self.assertEqual(toint("A"), 0, msg="Function toint is not functioning correctly for input 'A'!")
        self.assertEqual(toint("C"), 1, msg="Function toint is not functioning correctly for input 'C'!")
        self.assertEqual(toint("G"), 2, msg="Function toint is not functioning correctly for input 'G'!")
        self.assertEqual(toint("T"), 3, msg="Function toint is not functioning correctly for input 'T'!")


class TestGetFeaturesAndLabels(unittest.TestCase):

    def test_features_and_labels(self):
        A, y = get_features_and_labels("src/data.seq")
        n, m = A.shape
        self.assertEqual(n, y.shape[0], msg="Features and targets don't have the same number of rows!")
        self.assertEqual(n, 2000, msg="Incorrect number of samples!")
        self.assertEqual(m, 8, msg="Incorrect number of features!")
        d = Counter(A.flat)
        self.assertEqual(d[0], 6129, msg="Incorrect number of zeros (As)!")
        self.assertEqual(d[1], 2888, msg="Incorrect number of ones (Cs)!")
        self.assertEqual(d[2], 2912, msg="Incorrect number of twos (Gs)!")
        self.assertEqual(d[3], 4071, msg="Incorrect number of threes (Ts)!")
        self.assertEqual(sum(d.values()), 8 * 2000, msg="Incorrect total number of nucleotides!")


class TestClusterEuclidean(unittest.TestCase):

    def test_euclidean1(self):
        acc = cluster_euclidean("src/data.seq")
        self.assertAlmostEqual(acc, 0.9895, places=4,
                               msg="Function cluster_euclidean returned incorrect accuracy for file %s!" % "src/data.seq")

    def test_euclidean2(self):
        with patch("src.binding_sites.accuracy_score", side_effect=accuracy_score) as accs:
            cluster_euclidean("src/data.seq")
            accs.assert_called_once()

    def test_euclidean3(self):
        with patch("src.binding_sites.get_features_and_labels", side_effect=get_features_and_labels) as g:
            cluster_euclidean("src/data.seq")
            g.assert_called_once()

    def test_euclidean4(self):
        with patch("src.binding_sites.AgglomerativeClustering", side_effect=AgglomerativeClustering) as g:
            cluster_euclidean("src/data.seq")
            g.assert_called_once()
            args, kwargs = g.call_args
            if 'n_clusters' in kwargs:
                self.assertEqual(kwargs['n_clusters'], 2,
                                 "Incorrect argument value of n_clusters passed to AgglomerativeClustering!")
            elif len(args) > 0:
                    self.assertEqual(args[0], 2, "Incorrect argument value of n_clusters passed to AgglomerativeClustering!")
            self.assertIn('linkage', kwargs, msg="You did not give the 'linkage' parameter to AgglomerativeClustering!")
            self.assertEqual(kwargs['linkage'], "average",
                             "Incorrect argument value of 'linkage' passed to AgglomerativeClustering!")

            self.assertIn('affinity', kwargs, msg="You did not give the 'affinity' parameter to AgglomerativeClustering!")
            self.assertEqual(kwargs['affinity'], "euclidean",
                             "Incorrect argument value of 'affinity' passed to AgglomerativeClustering!")


class TestClusterHamming(unittest.TestCase):

    def test_hamming1(self):
        with patch("src.binding_sites.plt.show") as show:
            acc = cluster_hamming("src/data.seq")
            self.assertAlmostEqual(acc, 0.9985, places=4, msg="Function cluster_hamming returned incorrect accuracy for file %s!" % "src/data.seq")
            #show.assert_called_once()

    def test_hamming2(self):
        with patch("src.binding_sites.plt.show") as show:
            with patch("src.binding_sites.accuracy_score", side_effect=accuracy_score) as accs:
                acc = cluster_hamming("src/data.seq")
                accs.assert_called_once()

    def test_hamming3(self):
        with patch("src.binding_sites.plt.show") as show:
            with patch("src.binding_sites.get_features_and_labels", side_effect=get_features_and_labels) as g:
                with patch("src.binding_sites.pairwise_distances", side_effect=pairwise_distances) as ppd:
                    cluster_hamming("src/data.seq")
                    g.assert_called_once()
                    ppd.assert_called_once()
                    args, kwargs = ppd.call_args
                    self.assertIn('metric', kwargs, msg="You did not give the 'metric' parameter to pairwise_distances!")
                    self.assertEqual(kwargs['metric'], "hamming",
                             "Incorrect argument value of 'metric' passed to pairwise_distances!")

    def test_hamming4(self):
        with patch("src.binding_sites.plt.show") as show:
            with patch("src.binding_sites.AgglomerativeClustering", side_effect=AgglomerativeClustering) as g:
                cluster_hamming("src/data.seq")
                args, kwargs = g.call_args
                if 'n_clusters' in kwargs:
                    self.assertEqual(kwargs['n_clusters'], 2, "Incorrect argument value of n_clusters passed to AgglomerativeClustering!")
                elif len(args) > 0:
                    self.assertEqual(args[0], 2, "Incorrect argument value of n_clusters passed to AgglomerativeClustering!")
                self.assertIn('linkage', kwargs, msg="You did not give the 'linkage' parameter to AgglomerativeClustering!")
                self.assertEqual(kwargs['linkage'], "average", "Incorrect argument value passed to AgglomerativeClustering!")

                self.assertIn('affinity', kwargs, msg="You did not give the 'affinity' parameter to AgglomerativeClustering!")
                self.assertEqual(kwargs['affinity'], "precomputed", "Incorrect argument value passed to AgglomerativeClustering!")
                g.assert_called_once()


if __name__ == '__main__':
    unittest.main()
