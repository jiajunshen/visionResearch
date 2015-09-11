import caffe
import os
import deepdish as dd
import amitgroup as ag
import amitgroup.plot as gr
from __future__ import division, print_function, absolute_import
import numpy as np
import sys
from pnet.bernoullimm import BernoulliMM
import pnet
from sklearn.svm import LinearSVC


class CaffeToolKit:



    """
    Basic setup of the caffe environment.
    caffe mode has been set to GPU;
    """

