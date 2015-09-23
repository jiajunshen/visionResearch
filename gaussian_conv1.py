__author__ = 'jiajunshen'
import numpy as np
import caffe
import os
import deepdish as dd
import amitgroup as ag
import amitgroup.io
import random
import numpy
import lmdb
from sklearn.linear_model import SGDClassifier


if __name__ == "__main__":
    caffe.set_mode_gpu()
    os.chdir("/Users/jiajunshen/Documents/Research/caffe/")
    solver = caffe.get_solver("./examples/cifar10/cifar10_quick_solver_gaussian.prototxt")
    #Specify the directory of the conv1 locates
    filters = np.rollaxis(np.load("/Users/jiajunshen/Documents/Research/visionResearch/gmm_new.npy"),3,1)/10.0
    solver.net.params['conv1'][0].data[...] = filters
    solver.net.params['conv2'][1].data[...] = np.zeros(32)
    solver.net.params['conv3'][1].data[...] = np.zeros(64)
    solver.net.params['ip2'][1].data[...] = np.zeros(10)

    solver.step(4000)