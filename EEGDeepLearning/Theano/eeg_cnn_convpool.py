#!/usr/bin/env python
"""
Using convnets to classify EEG images into four categories corresponding to four
WM load levels.
Implementation using Lasagne module.
Input images are scaled by divining by 256. No baseline correction.
EEG images are generated by extracting FFT power within theta, alpha and beta frequency bands.
Duration of experiment is divided into 5 parts (700ms each).
EEG images corresponding to each part is fed into a
"""
#from __future__ import print_function
import time

import numpy as np
np.random.seed(1234)
import scipy.io
import theano
import theano.tensor as T

import lasagne
# from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
from lasagne.layers import Conv2DLayer, MaxPool2DLayer, InputLayer
from lasagne.layers import DenseLayer, ElemwiseMergeLayer, FlattenLayer
from lasagne.layers import ConcatLayer, ReshapeLayer, get_output_shape
from lasagne.layers import Conv1DLayer, DimshuffleLayer, LSTMLayer, SliceLayer
from utilsP import load_data, reformatInput

augment = False                                 # Flag for data augmentation
init_pars = False                               # Initialize parameters from convnet
#filename = 'EEG_images_32_timeWin'              # Name of the .mat file containing EEG images
filename ='EEG_images_32_flattened_locs'
filename_aug = 'EEG_images_32_timeWin_aug_pca'  # Name of the .mat file containing augmented EEG images
subjectsFilename = 'trials_subNums'             # Name of the .mat file containing trial/subject correspondence
model = 'mix'                                   # Model type selection
print('Model type is : {0}'.format(model))
num_epochs = 5                                  # Number of epochs for training
imSize = 32                                     # Size of the images
batch_size = 20                                 # Number of samples in each batch
nb_classes = 4                                  # Number of classes
numTimeWin = 7                                  # Number of time windows
GRAD_CLIP = 100                                 # Clipping value for gradient clipping in LSTM

# ##################### Build the neural network model #######################
# This script supports three types of models. For each one, we define a
# function that takes a Theano variable representing the input and returns
# the output layer of a neural network model built in Lasagne.

def build_cnn(input_var=None, W_init=None):
    """
    Builds a VGG style CNN network followed by a fully-connected layer and a softmax layer.
    input_var: Theano variable for input to the network
    outputs: pointer to the output of the last layer of network (softmax)
    :param input_var: theano variable as input to the network
    :param W_init: Initial weight values
    :return: a pointer to the output of last layer
    """

    weights = []        # Keeps the weights for all layers
    count = 0
    # If no initial weight is given, initialize with GlorotUniform
    if W_init is None:
        W_init = [lasagne.init.GlorotUniform()] * 7

    # Input layer
    network = InputLayer(shape=(None, 3, imSize, imSize),
                                        input_var=input_var)

    # CNN Stack 1
    network = Conv2DLayer(network, num_filters=32, filter_size=(3, 3),
                          W=W_init[count], pad='same')
    count += 1
    weights.append(network.W)
    network = Conv2DLayer(network, num_filters=32, filter_size=(3, 3),
                          W=W_init[count], pad='same')
    count += 1
    weights.append(network.W)
    network = Conv2DLayer(network, num_filters=32, filter_size=(3, 3),
                          W=W_init[count], pad='same')
    count += 1
    weights.append(network.W)
    network = Conv2DLayer(network, num_filters=32, filter_size=(3, 3),
                          W=W_init[count], pad='same')
    count += 1
    weights.append(network.W)
    network = MaxPool2DLayer(network, pool_size=(2, 2))

    # CNN Stack 2
    network = Conv2DLayer(network, num_filters=64, filter_size=(3, 3),
                          W=W_init[count], pad='same')
    count += 1
    weights.append(network.W)
    network = Conv2DLayer(network, num_filters=64, filter_size=(3, 3),
                          W=W_init[count], pad='same')
    count += 1
    weights.append(network.W)
    network = MaxPool2DLayer(network, pool_size=(2, 2))

    # CNN Stack 3
    network = Conv2DLayer(network, num_filters=128, filter_size=(3, 3),
                          W=W_init[count], pad='same')
    count += 1
    weights.append(network.W)
    network = MaxPool2DLayer(network, pool_size=(2, 2))

    return network, weights


def build_convpool_max(input_vars):
    """
    Builds the complete network with maxpooling layer in time.
    :param input_vars: list of EEG images (one image per time window)
    :return: a pointer to the output of last layer
    """
    convnets = []
    W_init = None

    # Build 7 parallel CNNs with shared weights
    for i in range(numTimeWin):
        if i == 0:
            convnet, W_init = build_cnn(input_vars[i])
        else:
            convnet, _ = build_cnn(input_vars[i], W_init)
        convnets.append(convnet)
    # convpooling using Max pooling over frames
    convpool = ElemwiseMergeLayer(convnets, theano.tensor.maximum)
    # A fully-connected layer of 512 units with 50% dropout on its inputs:
    convpool = DenseLayer(lasagne.layers.dropout(convpool, p=.5),
            num_units=512, nonlinearity=lasagne.nonlinearities.rectify)

    # And, finally, the output layer with 50% dropout on its inputs:
    convpool = lasagne.layers.DenseLayer(lasagne.layers.dropout(convpool, p=.5),
            num_units=nb_classes, nonlinearity=lasagne.nonlinearities.softmax)
    return convpool

def build_convpool_conv1d(input_vars):
    """
    Builds the complete network with 1D-conv layer to integrate time from sequences of EEG images.
    :param input_vars: list of EEG images (one image per time window)
    :return: a pointer to the output of last layer
    """
    convnets = []
    W_init = None
    # Build 7 parallel CNNs with shared weights
    for i in range(numTimeWin):
        if i == 0:
            convnet, W_init = build_cnn(input_vars[i])
        else:
            convnet, _ = build_cnn(input_vars[i], W_init)
        convnets.append(FlattenLayer(convnet))
    # at this point convnets shape is [numTimeWin][n_samples, features]
    # we want the shape to be [n_samples, features, numTimeWin]
    convpool = ConcatLayer(convnets)

    convpool = ReshapeLayer(convpool, ([0], numTimeWin, get_output_shape(convnets[0])[1]))
    convpool = DimshuffleLayer(convpool, (0, 2, 1))
    # convpool = ReshapeLayer(convpool, (-1, numTimeWin))

    # input to 1D convlayer should be in (batch_size, num_input_channels, input_length)
    convpool = Conv1DLayer(convpool, 64, 3)

    # A fully-connected layer of 512 units with 50% dropout on its inputs:
    convpool = DenseLayer(lasagne.layers.dropout(convpool, p=.5),
            num_units=512, nonlinearity=lasagne.nonlinearities.rectify)

    # And, finally, the output layer with 50% dropout on its inputs:
    convpool = DenseLayer(lasagne.layers.dropout(convpool, p=.5),
            num_units=nb_classes, nonlinearity=lasagne.nonlinearities.softmax)
    return convpool


def build_convpool_lstm(input_vars):
    """
    Builds the complete network with LSTM layer to integrate time from sequences of EEG images.
    :param input_vars: list of EEG images (one image per time window)
    :return: a pointer to the output of last layer
    """
    convnets = []
    W_init = None
    # Build 7 parallel CNNs with shared weights
    for i in range(numTimeWin):
        if i == 0:
            convnet, W_init = build_cnn(input_vars[i])
        else:
            convnet, _ = build_cnn(input_vars[i], W_init)
        convnets.append(FlattenLayer(convnet))
    # at this point convnets shape is [numTimeWin][n_samples, features]
    # we want the shape to be [n_samples, features, numTimeWin]
    convpool = ConcatLayer(convnets)
    # convpool = ReshapeLayer(convpool, ([0], -1, numTimeWin))

    convpool = ReshapeLayer(convpool, ([0], numTimeWin, get_output_shape(convnets[0])[1]))
    # Input to LSTM should have the shape as (batch size, SEQ_LENGTH, num_features)
    convpool = LSTMLayer(convpool, num_units=128, grad_clipping=GRAD_CLIP,
        nonlinearity=lasagne.nonlinearities.tanh)
    # After LSTM layer you either need to reshape or slice it (depending on whether you
    # want to keep all predictions or just the last prediction.
    # http://lasagne.readthedocs.org/en/latest/modules/layers/recurrent.html
    # https://github.com/Lasagne/Recipes/blob/master/examples/lstm_text_generation.py
    convpool = SliceLayer(convpool, -1, 1)      # Selecting the last prediction
    # A fully-connected layer of 256 units with 50% dropout on its inputs:
    convpool = DenseLayer(lasagne.layers.dropout(convpool, p=.5),
            num_units=256, nonlinearity=lasagne.nonlinearities.rectify)
    # We only need the final prediction, we isolate that quantity and feed it
    # to the next layer.

    # And, finally, the output layer with 50% dropout on its inputs:
    convpool = DenseLayer(lasagne.layers.dropout(convpool, p=.5),
            num_units=nb_classes, nonlinearity=lasagne.nonlinearities.softmax)
    return convpool

def build_convpool_mix(input_vars):
    """
    Builds the complete network with LSTM and 1D-conv layers combined
    to integrate time from sequences of EEG images.
    :param input_vars: list of EEG images (one image per time window)
    :return: a pointer to the output of last layer
    """
    convnets = []
    W_init = None
    # Build 7 parallel CNNs with shared weights
    for i in range(numTimeWin):
        if i == 0:
            convnet, W_init = build_cnn(input_vars[i])
        else:
            convnet, _ = build_cnn(input_vars[i], W_init)
        convnets.append(FlattenLayer(convnet))
    # at this point convnets shape is [numTimeWin][n_samples, features]
    # we want the shape to be [n_samples, features, numTimeWin]
    convpool = ConcatLayer(convnets)
    # convpool = ReshapeLayer(convpool, ([0], -1, numTimeWin))

    convpool = ReshapeLayer(convpool, ([0], numTimeWin, get_output_shape(convnets[0])[1]))
    reformConvpool = DimshuffleLayer(convpool, (0, 2, 1))

    # input to 1D convlayer should be in (batch_size, num_input_channels, input_length)
    conv_out = Conv1DLayer(reformConvpool, 64, 3)
    conv_out = FlattenLayer(conv_out)
    # Input to LSTM should have the shape as (batch size, SEQ_LENGTH, num_features)
    lstm = LSTMLayer(convpool, num_units=128, grad_clipping=GRAD_CLIP,
        nonlinearity=lasagne.nonlinearities.tanh)
    # After LSTM layer you either need to reshape or slice it (depending on whether you
    # want to keep all predictions or just the last prediction.
    # http://lasagne.readthedocs.org/en/latest/modules/layers/recurrent.html
    # https://github.com/Lasagne/Recipes/blob/master/examples/lstm_text_generation.py
    # lstm_out = SliceLayer(convpool, -1, 1)        # bypassing LSTM
    lstm_out = SliceLayer(lstm, -1, 1)

    # Merge 1D-Conv and LSTM outputs
    dense_input = ConcatLayer([conv_out, lstm_out])
    # A fully-connected layer of 256 units with 50% dropout on its inputs:
    convpool = DenseLayer(lasagne.layers.dropout(dense_input, p=.5),
            num_units=512, nonlinearity=lasagne.nonlinearities.rectify)
    # We only need the final prediction, we isolate that quantity and feed it
    # to the next layer.

    # And, finally, the 10-unit output layer with 50% dropout on its inputs:
    convpool = DenseLayer(convpool,
            num_units=4, nonlinearity=lasagne.nonlinearities.softmax)
    return convpool

# ############################# Batch iterator ###############################
# This is just a simple helper function iterating over training data in
# mini-batches of a particular size, optionally in random order. It assumes
# data is available as numpy arrays. For big datasets, you could load numpy
# arrays as memory-mapped files (np.load(..., mmap_mode='r')), or write your
# own custom data iteration function. For small datasets, you can also copy
# several changes in the main program, though, and is not demonstrated here.
# them to GPU at once for slightly improved performance. This would involve
########################## Borrowed from Lasagne example ####################

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    input_len = inputs.shape[1]
    assert input_len == len(targets)
    if shuffle:
        indices = np.arange(input_len)
        np.random.shuffle(indices)
    for start_idx in range(0, input_len - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[:, excerpt], targets[excerpt]


# ############################## Main program ################################

def main():
    # Load the dataset
    #R: 'EEG_images_32_flattened_locs'
    #R: EEG_images_32_timeWin_flattened_locs',
    # filename = 'EEG_images_32_timeWin'              # Name of the .mat file containing EEG images
    # filename_aug = 'EEG_images_32_timeWin_aug_pca'  # Name of the .mat file containing augmented EEG images
    # subjectsFilename = 'trials_subNums'
    print("Loading data...")
    data, labels = load_data(filename)
    labels=-1*np.transpose(labels)
    mat = scipy.io.loadmat(subjectsFilename, mat_dtype=True)
    subjNumbers = np.squeeze(mat['subjectNum'])     # subject IDs for each trial
    #### rakib

    mat = scipy.io.loadmat('Neuroscan_locs_orig')
    caploc=mat['A']
    new=[]
    for element in np.arange(0,caploc.shape[0]):
        #print "cc"
        new.append(caploc[element][0])
        new.append(caploc[element][1])
    locs=np.asanyarray(new).reshape(32,2)
    #print locs
    print 'location converted'

    ###

    # Create folds based on subject numbers (for leave-subject-out x-validation)
    fold_pairs = []
    if augment:
        # Aggregate augmented data and labels
        data_aug, labels_aug = load_data(filename_aug)
        data = np.concatenate((data, data_aug), axis=1)
        labels = np.vstack((labels, labels_aug))
        # Leave-Subject-Out cross validation
        for i in np.unique(subjNumbers):
            ts = subjNumbers == i
            tr = np.squeeze(np.nonzero(np.bitwise_not(ts)))         # Training indices
            ts = np.squeeze(np.nonzero(ts))
            # Include augmented training data
            tr = np.concatenate((tr, tr+subjNumbers.size))
            np.random.shuffle(tr)       # Shuffle indices
            np.random.shuffle(ts)
            fold_pairs.append((tr, ts))
    else:
        # Leave-Subject-Out cross validation
        for i in np.unique(subjNumbers):
            ts = subjNumbers == i
            tr = np.squeeze(np.nonzero(np.bitwise_not(ts)))
            ts = np.squeeze(np.nonzero(ts))
            np.random.shuffle(tr)       # Shuffle indices
            np.random.shuffle(ts)
            fold_pairs.append((tr, ts))

    # Initializing output variables
    validScores, testScores = [], []
    trainLoss = np.zeros((len(fold_pairs), num_epochs))
    validLoss = np.zeros((len(fold_pairs), num_epochs))
    validEpochAccu = np.zeros((len(fold_pairs), num_epochs))
    fold_pairs[:1]
    print "start for loop"
    for foldNum, fold in enumerate(fold_pairs):
        #print('Beginning fold {0} out of {1}'.format(foldNum+1, len(foldNum)))
        #print foldNum
        #print foldNum
        print "fold"
        print fold
        # Divide the dataset into train, validation and test sets
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = reformatInput(data, labels, fold)
        X_train = X_train.astype("float32", casting='unsafe')
        X_val = X_val.astype("float32", casting='unsafe')
        X_test = X_test.astype("float32", casting='unsafe')

        # trainMeans = [np.mean(X_train[:, :, i, :, :].flatten()) for i in range(X_train.shape[2])]
        # trainStds = [np.std(X_train[:, :, i, :, :].flatten()) for i in range(X_train.shape[2])]
        # for i in range(len(trainMeans)):
        #     X_train[:, :, i, :, :] = (X_train[:, :, i, :, :] - trainMeans[i]) / trainStds[i]
        #     X_val[:, :, i, :, :] = (X_val[:, :, i, :, :] - trainMeans[i]) / trainStds[i]
        #     X_test[:, :, i, :, :] = (X_test[:, :, i, :, :] - trainMeans[i]) / trainStds[i]
        # X_train = X_train / np.float32(256)
        # X_val = X_val / np.float32(256)
        # X_test = X_test / np.float32(256)

        # Prepare Theano variables for inputs and targets
        input_var = T.TensorType('floatX', ((False,) * 5))()        # Notice the () at the end
        target_var = T.ivector('targets')
        # Create neural network model (depending on first command line parameter)
        print("Building model and compiling functions...")
        # Building the appropriate model
        if model == '1dconv':
            network = build_convpool_conv1d(input_var)
        elif model == 'maxpool':
            network = build_convpool_max(input_var)
        elif model == 'lstm':
            network = build_convpool_lstm(input_var)
        elif model == 'mix':
            network = build_convpool_mix(input_var)

        # Initialize parameters with previously saved ones.
        if init_pars:
            with np.load('weigths_lasg{0}.npz'.format(foldNum)) as f:
                # Extract CNN parameters only (not the FC layers)
                param_values = [f['arr_%d' % i] for i in range(14)]
                layers = lasagne.layers.get_all_layers(network)
                lasagne.layers.set_all_param_values(layers[83], param_values)

        # Create a loss expression for training, i.e., a scalar objective we want
        # to minimize (for our multi-class problem, it is the cross-entropy loss):
        prediction = lasagne.layers.get_output(network)
        loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
        loss = loss.mean()
        # We could add some weight decay as well here, see lasagne.regularization.

        # Create update expressions for training, i.e., how to modify the
        # parameters at each training step. Here, we'll use Stochastic Gradient
        # Descent (SGD) with Nesterov momentum, but Lasagne offers plenty more.
        params = lasagne.layers.get_all_params(network, trainable=True)
        updates = lasagne.updates.adam(loss, params, learning_rate=0.001)

        # Create a loss expression for validation/testing. The crucial difference
        # here is that we do a deterministic forward pass through the network,
        # disabling dropout layers.
        test_prediction = lasagne.layers.get_output(network, deterministic=True)
        test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                                                                target_var)
        test_loss = test_loss.mean()
        # As a bonus, also create an expression for the classification accuracy:
        test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                          dtype=theano.config.floatX)

        # Compile a function performing a training step on a mini-batch (by giving
        # the updates dictionary) and returning the corresponding training loss:
        train_fn = theano.function([input_var, target_var], loss, updates=updates)

        # Compile a second function computing the validation loss and accuracy:
        val_fn = theano.function([input_var, target_var], [test_loss, test_acc])

        # Finally, launch the training loop.
        print("Starting training...")
        best_validation_accu = 0
        # We iterate over epochs:
        for epoch in range(num_epochs):
            # In each epoch, we do a full pass over the training data:
            train_err = 0
            train_batches = 0
            start_time = time.time()
            for batch in iterate_minibatches(X_train, y_train, batch_size, shuffle=False):
                inputs, targets = batch
                train_err += train_fn(inputs, targets)
                train_batches += 1

            # And a full pass over the validation data:
            val_err = 0
            val_acc = 0
            val_batches = 0
            for batch in iterate_minibatches(X_val, y_val, batch_size, shuffle=False):
                inputs, targets = batch
                err, acc = val_fn(inputs, targets)
                val_err += err
                val_acc += acc
                val_batches += 1
            av_train_err = train_err / train_batches
            av_val_err = val_err / val_batches
            av_val_acc = val_acc / val_batches
            # Then we print the results for this epoch:
            print("Epoch {} of {} took {:.3f}s".format(
                epoch + 1, num_epochs, time.time() - start_time))
            print("  training loss:\t\t{:.6f}".format(av_train_err))
            print("  validation loss:\t\t{:.6f}".format(av_val_err))
            print("  validation accuracy:\t\t{:.2f} %".format(av_val_acc * 100))

            trainLoss[foldNum, epoch] = av_train_err
            validLoss[foldNum, epoch] = av_val_err
            validEpochAccu[foldNum, epoch] = av_val_acc * 100

            if av_val_acc > best_validation_accu:
                best_validation_accu = av_val_acc

                # After training, we compute and print the test error:
                test_err = 0
                test_acc = 0
                test_batches = 0
                for batch in iterate_minibatches(X_test, y_test, batch_size, shuffle=False):
                    inputs, targets = batch
                    err, acc = val_fn(inputs, targets)
                    test_err += err
                    test_acc += acc
                    test_batches += 1

                av_test_err = test_err / test_batches
                av_test_acc = test_acc / test_batches
                print("Final results:")
                print("  test loss:\t\t\t{:.6f}".format(av_test_err))
                print("  test accuracy:\t\t{:.2f} %".format(av_test_acc * 100))
                # Dump the network weights to a file like this:
                np.savez('weights_lasg_{0}_{1}'.format(model, foldNum), *lasagne.layers.get_all_param_values(network))
        validScores.append(best_validation_accu * 100)
        testScores.append(av_test_acc * 100)
        print('-'*50)
        print("Best validation accuracy:\t\t{:.2f} %".format(best_validation_accu * 100))
        print("Best test accuracy:\t\t{:.2f} %".format(av_test_acc * 100))
    scipy.io.savemat('cnn_lasg_{0}_results'.format(model),
                     {'validAccu': validScores,
                      'testAccu': testScores,
                      'trainLoss': trainLoss,
                      'validLoss': validLoss,
                      'validEpochAccu': validEpochAccu
                      })


if __name__ == '__main__':
    main()