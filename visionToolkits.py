from __future__ import division, print_function, absolute_import
import amitgroup as ag
import pnet
import caffe
import os
import deepdish as dd
import amitgroup as ag
import amitgroup.plot as gr
import numpy as np
import sys
from pnet.bernoullimm import BernoulliMM
import pnet
from sklearn.svm import LinearSVC


def generatePatches(data, regionSize = 32, patchSize = 5):
    finalPatches = []
    for i in range(data.shape[0]):
        counter = 0
        sampleData = zero_pad(data[i],2)
        while(counter<40):
            x = np.random.randint(low = 0, high=regionSize)
            y = np.random.randint(low = 0, high=regionSize)
            finalPatches.append(sampleData[x:x+patchSize,y:y+patchSize,:])
            counter+=1
    return finalPatches



def zero_pad(data, pad_size):
    data_shape = data.shape
    result_data = np.zeros((data_shape[0] + 2 * pad_size, data_shape[1] + 2 * pad_size, data_shape[2]))
    result_data[pad_size:pad_size + data_shape[0], pad_size:pad_size+data_shape[1],:] = data
    return result_data

def load_mean_cifar10():
    trainingData, trainingLabel = ag.io.load_cifar10("training")
    trainingData = trainingData * 255
    meanTraining = np.mean(trainingData, axis=0)
    trainingData = trainingData - meanTraining
    trainingData = np.rollaxis(trainingData, 3, 1)

    testingData, testingLabel = ag.io.load_cifar10("testing")
    testingData = testingData * 255
    testingData = testingData - meanTraining
    testingData = np.rollaxis(testingData, 3, 1)
    return trainingData, trainingLabel, testingData, testingLabel

def trainGaussian():
    training_seed = 1
    layers = [
        pnet.OrientedGaussianPartsLayer(32,4,(5,5),settings=dict(
            seed=training_seed,
            n_init = 2,
            samples_per_image=40,
            max_samples=100000,
            channel_mode='together'
            #covariance_type = ''
        )),
        pnet.PoolingLayer(shape=(8,8),strides=(2,2)),
        pnet.SVMClassificationLayer(C=1.0)
    ]
    net = pnet.PartsNet(layers)
    trainingData, trainingLabel, testingData, testingLabel = load_mean_cifar10()
    net.train(np.rollaxis(trainingData,1,4)/255.0, trainingLabel)
    return net



if __name__ == "__main__":
    trainedNet = trainGaussian()
    trainedNet.save("TrainedNetNewRotation2.npy")