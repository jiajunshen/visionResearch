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
import h5py
from sklearn.linear_model import SGDClassifier

os.chdir("/Users/jiajunshen/Documents/Research/caffe/")
caffe.set_mode_gpu()
solver = caffe.get_solver("./examples/cifar10/cifar10_quick_solver.prototxt")
trainingData = np.empty((50000, 3, 32, 32))
trainingLabel = np.empty(50000)
testingData = np.empty((10000, 3, 32, 32))
testingLabel = np.empty(10000)


def retrieveSolver(solverPath, solverStatePath):
    global solver
    solver = caffe.get_solver(solverPath)
    solver.net.copy_from(solverStatePath)

def zero_pad(data, pad_size):
    data_shape = data.shape
    print data_shape
    result_data = np.zeros((data_shape[0], data_shape[1], data_shape[2] + 2 * pad_size, data_shape[3] + 2 * pad_size))
    result_data[:, :, pad_size:pad_size + data_shape[2], pad_size:pad_size+data_shape[3]] = data
    return result_data

def trainNewSolver(trainSolverPath, saveToSolverPath):
    """
    Here we want to train a new solver with a certain features:
        1. All biased nodes are zero
        2. Construct a certain net and train
    :return: void. Save the solverstate to a directory
    """
    global solver
    solver = caffe.get_solver(trainSolverPath)
    solver.step(10000)


def retrieveParams(layerName):
    return np.array(solver.net.params[layerName][0].data)


def retrieveActivations(layerName):
    return np.array(solver.net.blobs[layerName].data)


def assignmentParams(layerName, inputArray):
    global solver
    solver.net.params[layerName].data[...] = np.array(inputArray).reshape(solver.net.params[layerName].data.shape)

def createLMDB(X, Y, dbName, order):
    N = X.shape[0]
    X = np.array(X, dtype=np.uint8)
    Y = np.array(Y, dtype=np.int64)
    arr = order
    map_size = dd.bytesize(X) * 2
    env = lmdb.open(dbName, map_size=map_size)

    for i in range(N):
        datum = caffe.proto.caffe_pb2.Datum()
        datum.channels = X.shape[1]
        datum.height = X.shape[2]
        datum.width = X.shape[3]
        datum.data = X[arr[i]].tobytes()  # or .tostring() if numpy < 1.9
        datum.label = int(Y[arr[i]])
        str_id = '{:08}'.format(i)

        with env.begin(write=True) as txn:
            # txn is a Transaction object
            # The encode is only essential in Python 3
            txn.put(str_id.encode('ascii'), datum.SerializeToString())


def createLMDBLabel(Y, dbName, order):
    N = Y.shape[0]
    Y = np.array(Y, dtype=np.int64)
    arr = order
    map_size = dd.bytesize(Y) * 100
    env = lmdb.open(dbName, map_size=map_size)

    for i in range(N):
        datum = caffe.proto.caffe_pb2.Datum()
        datum.channels = 1
        datum.height = 1
        datum.width = 1
        datum.data = np.zeros((1,1,1)).tobytes()  # or .tostring() if numpy < 1.9
        datum.label = int(Y[arr[i]])
        str_id = '{:08}'.format(i)

        with env.begin(write=True) as txn:
            # txn is a Transaction object
            # The encode is only essential in Python 3
            txn.put(str_id.encode('ascii'), datum.SerializeToString())

def createHDF5Label(Y, dbName, order):
    N = Y.shape[0]
    channel = 3;
    height = 1
    width = 1
    total_size = N * channel * height * width

    data = np.arange(total_size)
    data = data.reshape(N, channel, height, width)
    data = data.astype('float32')

    # We had a bug where data was copied into label, but the tests weren't
    # catching it, so let's make label 1-indexed.
    label = Y[order]
    label = label.astype('float32')
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # We add an extra label2 dataset to test HDF5 layer's ability
    # to handle arbitrary number of output ("top") Blobs.

    print data
    print label

    with h5py.File(script_dir + '/' + dbName+'.h5', 'w') as f:
        f['data1'] = data
        f['label'] = label

    with h5py.File(script_dir + '/' + dbName + '_gzip.h5', 'w') as f:
        f.create_dataset(
            'data1', data=data + total_size,
            compression='gzip', compression_opts=1
        )
        f.create_dataset(
            'label', data=label,
            compression='gzip', compression_opts=1
        )

    with open(script_dir + '/sample_data_list' + dbName + '.txt', 'w') as f:
        f.write(script_dir + '/' + dbName+'.h5\n')
        f.write(script_dir + '/' + dbName + '_gzip.h5\n')

def load_data():
    global trainingData
    global trainingLabel
    global testingData
    global testingLabel

    trainingData, trainingLabel = ag.io.load_cifar10("training")
    trainingData = trainingData * 255
    meanTraining = np.mean(trainingData, axis=0)
    trainingData = trainingData - meanTraining
    trainingData = np.rollaxis(trainingData, 3, 1)

    testingData, testingLabel = ag.io.load_cifar10("testing")
    testingData = testingData * 255
    testingData = testingData - meanTraining
    testingData = np.rollaxis(testingData, 3, 1)

def getActivations(layerName, setLabel, zeroPad = True):
    """
    This will provide the activations from a certain layer using either training data or testing data
    :ivar
        layerName: Name of the output layers
        setLabel: which dataset do you want to use. either training data or testing data.

    :returns
        activationArray.
    """
    activationArray = None
    rangeIndex = 500
    data = trainingData
    if setLabel == "testing":
        rangeIndex = 100
        data = testingData
    elif setLabel == "training":
        rangeIndex = 500
        data = trainingData

    for i in range(rangeIndex):
        solver.net.blobs['data'].data[...] = np.array(data[i * 100: (i + 1) * 100])
        solver.net.forward(start="conv1")
        if activationArray == None:
            activationArray = np.array(solver.net.blobs[layerName].data)
        else:
            activationArray = np.concatenate((activationArray, np.array(solver.net.blobs[layerName].data)))
    if(zeroPad):
        activationArray = zero_pad(data = activationArray, pad_size=2)
    #activationArray = activationArray.reshape(100 * rangeIndex, - 1)
    return activationArray

def generatePatches(activationArray, patchSize):
    n = activationArray.shape[0]
    w = activationArray.shape[2]
    l = activationArray.shape[3]
    patch_activationArray = []
    for k in range(n):
        for x in range(w - patchSize + 1):
            for y in range(l - patchSize + 1):
                patch_activationArray.append(activationArray[k, :, x:x+5, y:y+5])
    patch_activationArray = np.array(patch_activationArray)
    return patch_activationArray



def randomPartition(trainingLabel, testLabel):
    """
    randomly partition the groups into two groups.
    :ivar
        trainingLabel: the labels of the training data
        testLabel: the labels of the testing data
    :returns
        newTrainLabel
        newTestLabel
        sampledGroup
    """
    sampleNumber = np.random.randint(4, 7)
    sampledGroup = np.random.choice(range(10),size = sampleNumber,replace=False)
    newTestLabel = [testLabel[i] in sampledGroup for i in range(len(testLabel))]
    newTestLabel = np.array(newTestLabel,dtype=np.int)
    newTrainLabel = [trainingLabel[i] in sampledGroup for i in range(len(trainingLabel))]
    newTrainLabel = np.array(newTrainLabel,dtype=np.int)
    return newTrainLabel, newTestLabel


def batches(l, n):
    for i in xrange(0, len(l), n):
        yield l[i:i+n]

"""
An example of retrieveSolver from files
"""
if __name__ == "__main__":
    """
    Example of trainning new nets.
    """
    #trainSolverPath = "./examples/cifar10/cifar10_quick_solver_nofully.prototxt"
    #trainNewSolver(trainSolverPath, None)
    """
    Example of training the last layer with SVM
    """


    load_data()

    print("Preparing the activation from the last second conv layer")
    trainingActivation = getActivations("pool2", "training")# of size (50000, 32, 12, 12)
    testingActivation = getActivations("pool2", "testing")# of size (10000, 32, 12, 12)

    #preparing the random partitioned labels
    alteredTrainingLabels = np.zeros((100, 50000))
    alteredTestingLabels = np.zeros((100, 10000))
    patchTrainingLabels = np.zeros((100, 50000 * 64))
    patchTestingLabels = np.zeros((100, 10000 * 64))
    for i in range(100):
        alteredTrainingLabels[i], alteredTestingLabels[i] = randomPartition(trainingLabel, testingLabel)
    for i in range(50000 * 64):
        patchTrainingLabels[:, i] = alteredTrainingLabels[:, i//64]
    for i in range(10000 * 64):
        patchTestingLabels[:, i] = alteredTestingLabels[:, i//64]

    index = 0
    disablePatchLearning = 0
    if not disablePatchLearning:
        trainingPatches = generatePatches(activationArray=trainingActivation, patchSize=5)
        trainingPatchesMean = np.mean(trainingPatches,axis = 0)
        trainingPatches = trainingPatches - trainingPatchesMean
        testingPatches = generatePatches(activationArray=testingActivation, patchSize=5)
        testingPatches = testingPatches - trainingPatchesMean
        print trainingPatches.shape, patchTrainingLabels.shape, testingPatches, patchTestingLabels.shape
    N_train = trainingPatches.shape[0]
    N_test = testingPatches.shape[0]
    training_order = np.arange(N_train)
    testing_order = np.arange(N_test)
    np.random.shuffle(training_order)
    np.random.shuffle(testing_order)
    createLMDB(X = trainingPatches, Y = patchTrainingLabels[index], dbName="cifar10_patch_train_lmdb%d" %index, order = training_order)
    createLMDB(X = testingPatches, Y = patchTestingLabels[index], dbName="cifar10_patch_test_lmdb%d" %index, order = testing_order)

    #for index in range(100):
    #    print index
    #    createLMDBLabel(Y = patchTrainingLabels[index], dbName = "cifar10_patch_train_label_lmdb%d" %index)
    #    createLMDBLabel(Y = patchTestingLabels[index], dbName = "cifar10_patch_test_label_lmdb%d" %index)

    for index in range(100):
        print index
        createHDF5Label(Y = patchTrainingLabels[index], dbName="cifar10_patch_train_lmdb%d" %index, order = training_order)
        createHDF5Label(Y = patchTestingLabels[index], dbName="cifar10_patch_test_lmdb%d" %index, order = testing_order)

    """
    clf2 = SGDClassifier(loss='hinge',random_state = 0,verbose=True) # shuffle=True is useless here
    shuffledRange = range(len(trainingPatches))

    n_iter = 2
    for n in range(n_iter):
        random.shuffle(shuffledRange)
        shuffledX = [trainingPatches[i] for i in shuffledRange]
        shuffledY = [patchTrainingLabels[0][i] for i in shuffledRange]
        counter = 0
        for batch in batches(range(len(shuffledX)), 10000):
            print counter
            clf2.partial_fit(shuffledX[batch[0]:batch[-1]+1], shuffledY[batch[0]:batch[-1]+1], classes=numpy.unique([0,1]))
            counter+=1
    """




    #svmSolver = caffe.get_solver





