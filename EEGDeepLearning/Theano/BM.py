#!/usr/bin/env python

from __future__ import division
from enum import Enum
import numpy as np
#########################################

import six.moves.cPickle as pickle
import gzip
import os
import sys
import timeit

#import numpy

import theano
import theano.tensor as T


#########################################
Clamp = Enum('Clamp', 'VISIBLE_UNITS NONE INPUT_UNITS')

class Step:
    def __init__(self, temperature, epochs):
        self.temperature = temperature
        self.epochs = epochs

numInputUnits = 4
numOutputUnits = 4
numHiddenUnits = 2

numVisibleUnits = numInputUnits + numOutputUnits
numUnits = numVisibleUnits+numHiddenUnits

annealingSchedule = [Step(20.,2),
                     Step(15.,2),
                     Step(12.,2),
                     Step(10.,4)]

coocurranceCycle = Step(10.,10)

weights = np.zeros((numUnits,numUnits))
states = np.zeros(numUnits)
energy = np.zeros(numUnits)

connections = np.zeros((numUnits,numUnits), dtype=np.int)
for i in xrange(numInputUnits):
    for j in xrange(i+1,numInputUnits):
        connections[i,j] = 1
    for j in xrange(1,numHiddenUnits+1):
        connections[i,-j] = 1

for i in xrange(numOutputUnits):
    for j in xrange(i+1,numOutputUnits):
        connections[i+numInputUnits,j+numInputUnits] = 1
    for j in xrange(1,numHiddenUnits+1):
        connections[i+numInputUnits,-j] = 1

for i in xrange(numHiddenUnits,0,-1):
    for j in xrange(i-1,0,-1):
        connections[-i,-j] = 1

valid = np.nonzero(connections)
numConnections = np.size(valid[0])
connections[valid] = np.arange(1,numConnections+1)
connections = connections + connections.T - 1

def propagate(temperature, clamp):
    global energy, states, weights

    if clamp == Clamp.VISIBLE_UNITS:
        numUnitsToSelect = numHiddenUnits
    elif clamp == Clamp.NONE:
        numUnitsToSelect = numUnits
    else:
        numUnitsToSelect = numHiddenUnits + numOutputUnits

    for i in xrange(numUnitsToSelect):
        # Calculating the energy of a randomly selected unit
        unit=numUnits-np.random.randint(1,numUnitsToSelect+1)
        energy[unit] = np.dot(weights[unit,:], states)

        p = 1. / (1.+ np.exp(-energy[unit] / temperature))
        states[unit] = 1. if  np.random.uniform() <= p else 0

def anneal(annealingSchedule, clamp):
    for step in annealingSchedule:
        for epoch in xrange(step.epochs):
            propagate(step.temperature, clamp)

def sumCoocurrance(clamp):
    sums = np.zeros(numConnections)
    for epoch in xrange(coocurranceCycle.epochs):
        propagate(coocurranceCycle.temperature, clamp)
        for i in xrange(numUnits):
            if(states[i] == 1):
                for j in xrange(i+1,numUnits):
                    if(connections[i,j]>-1 and states[j] ==1):
                        sums[connections[i,j]] += 1
    return sums

def updateWeights(pplus, pminus):
    global weights
    for i in xrange(numUnits):
        for j in xrange(i+1,numUnits):
            if connections[i,j] > -1:
                index = connections[i,j]
                weights[i,j] += 2*np.sign(pplus[index] - pminus[index])
                weights[j,i] = weights[i,j]

def recall(pattern):
    global states

    # Setting pattern to recall
    states[0:numInputUnits] = pattern

    # Assigning random values to the hidden and output states
    states[-(numHiddenUnits+numOutputUnits):] = np.random.choice([0,1],numHiddenUnits+numOutputUnits)

    anneal(annealingSchedule, Clamp.INPUT_UNITS)

    return states[numInputUnits:numInputUnits+numOutputUnits]

def addNoise(pattern):
    probabilities = 0.8*pattern+0.05
    uniform = np.random.random(numVisibleUnits)
    return (uniform < probabilities).astype(int)



def load_data(dataset):
    ''' Loads the dataset

    :type dataset: string
    :param dataset: the path to the dataset (here MNIST)
    '''

    #############
    # LOAD DATA #
    #############

    # Download the MNIST dataset if it is not present
    data_dir, data_file = os.path.split(dataset)
    if data_dir == "" and not os.path.isfile(dataset):
        # Check if dataset is in the data directory.
        new_path = os.path.join(
            os.path.split(__file__)[0],
            "..",
            "data",
            dataset
        )
        if os.path.isfile(new_path) or data_file == 'mnist.pkl.gz':
            dataset = new_path

    if (not os.path.isfile(dataset)) and data_file == 'mnist.pkl.gz':
        from six.moves import urllib
        origin = (
            'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        )
        print('Downloading data from %s' % origin)
        urllib.request.urlretrieve(origin, dataset)

    print('... loading data')

    # Load the dataset
    with gzip.open(dataset, 'rb') as f:
        try:
            train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
        except:
            train_set, valid_set, test_set = pickle.load(f)
    # train_set, valid_set, test_set format: tuple(input, target)
    # input is a numpy.ndarray of 2 dimensions (a matrix)
    # where each row corresponds to an example. target is a
    # numpy.ndarray of 1 dimension (vector) that has the same length as
    # the number of rows in the input. It should give the target
    # to the example with the same index in the input.

    def shared_dataset(data_xy, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy
        shared_x = theano.shared(np.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(np.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
        return shared_x, T.cast(shared_y, 'int32')

    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval




def learn(patterns):
    global states, weights
    numPatterns = patterns.shape[0]
    trials=numPatterns*coocurranceCycle.epochs
    #weights = np.zeros((m,m))

    weights = np.zeros((10,10))

    for i in xrange(1800):
        # Positive phase
        pplus = np.zeros(numConnections)
        for pattern in patterns:

            # Setting visible units values (inputs and outputs)
            states[0:numVisibleUnits] = addNoise(pattern)

            # Assigning random values to the hidden units
            states[-numHiddenUnits:] = np.random.choice([0,1],numHiddenUnits)

            anneal(annealingSchedule, Clamp.VISIBLE_UNITS)
            pplus += sumCoocurrance(Clamp.VISIBLE_UNITS)
        pplus /= trials

        # Negative phase
        states = np.random.choice([0,1],numUnits)
        anneal(annealingSchedule, Clamp.NONE)
        pminus = sumCoocurrance(Clamp.NONE) / coocurranceCycle.epochs

        updateWeights(pplus,pminus)

datasets=load_data("C:\Users\Rakib\PycharmProjects\Theano\mnist.pkl.gz")
train_set_x, train_set_y = datasets[0]
test_set_x, test_set_y = datasets[2]

patterns = np.array([[1, 0, 0, 0, 1, 0, 0, 0],
                     [0, 1, 0, 0, 0, 1, 0, 0],
                     [0, 0, 1, 0, 0, 0, 1, 0],
                     [0, 0, 0, 1, 0, 0, 0, 1]])
learn(patterns)
print weights

#print recall(np.array([1, 0, 0, 0]))