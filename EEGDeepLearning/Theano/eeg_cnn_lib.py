# from __future__ import print_function
import time

import numpy as np
np.random.seed(1234)

import math as m

import scipy.io
import theano
import theano.tensor as T

from scipy.interpolate import griddata
from scipy.misc import bytescale
from sklearn.preprocessing import scale
from utilsP import augment_EEG, cart2sph, pol2cart,reformatInput

import lasagne
# from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
from lasagne.layers import Conv2DLayer, MaxPool2DLayer, InputLayer
from lasagne.layers import DenseLayer, ElemwiseMergeLayer, FlattenLayer
from lasagne.layers import ConcatLayer, ReshapeLayer, get_output_shape
from lasagne.layers import Conv1DLayer, DimshuffleLayer, LSTMLayer, SliceLayer


def azim_proj(pos):
    """
    Computes the Azimuthal Equidistant Projection of input point in 3D Cartesian Coordinates.
    Imagine a plane being placed against (tangent to) a globe. If
    a light source inside the globe projects the graticule onto
    the plane the result would be a planar, or azimuthal, map
    projection.

    :param pos: position in 3D Cartesian coordinates
    :return: projected coordinates using Azimuthal Equidistant Projection
    """
    [r, elev, az] = cart2sph(pos[0], pos[1], pos[2])
    return pol2cart(az, m.pi / 2 - elev)


def gen_images(locs, features, nGridPoints, normalize=True,
               augment=False, pca=False, stdMult=0.1, n_components=2, edgeless=False):
    """
    Generates EEG images given electrode locations in 2D space and multiple feature values for each electrode

    :param loc: An array with shape [n_electrodes, 2] containing X, Y
                        coordinates for each electrode.
    :param features: Feature matrix as [n_samples, n_features+1]
                                Features are as columns.
                                Features corresponding to each frequency band are concatenated.
                                (alpha1, alpha2, ..., beta1, beta2,...)
    :param nGridPoints: Number of pixels in the output images
    :param normalize:   Flag for whether to normalize each feature over all samples
    :param augment:     Flag for generating augmented images
    :param pca:         Flag for PCA based data augmentation
    :param stdMult:     Standard deviation of noise for augmentation
    :param n_components:Number of components in PCA to retain for augmentation
    :param edgeless:    If True generates edgeless images by adding artificial channels
                        at four corners of the image with value = 0 (default=False).
    :return:            Tensor of size [samples, colors, W, H] containing generated
                        images.
    """

    feat_array_temp = []
    nElectrodes = locs.shape[0]     # Number of electrodes
    #R edit start
    features=features[:,0:features.shape[1]-1]
    #R edit end
    # Test whether the feature vector length is divisible by number of electrodes (last column is the labels)
    assert features.shape[1] % nElectrodes == 0
    n_colors = features.shape[1] / nElectrodes
    for c in range(n_colors):
        feat_array_temp.append(features[:, c * nElectrodes : nElectrodes * (c+1)])


    if augment:
        if pca:
            for c in range(n_colors):
                feat_array_temp[c] = augment_EEG(feat_array_temp[c], stdMult, pca=True, n_components=n_components)
        else:
            for c in range(n_colors):
                feat_array_temp[c] = augment_EEG(feat_array_temp[c], stdMult, pca=False, n_components=n_components)

    nSamples = features.shape[0]
    # Interpolate the values
    grid_x, grid_y = np.mgrid[
                     min(locs[:, 0]):max(locs[:, 0]):nGridPoints*1j,
                     min(locs[:, 1]):max(locs[:, 1]):nGridPoints*1j
                     ]
    temp_interp = []
    for c in range(n_colors):
        temp_interp.append(np.zeros([nSamples, nGridPoints, nGridPoints]))

    # Generate edgeless images
    if edgeless:
        min_x, min_y = np.min(locs, axis=0)
        max_x, max_y = np.max(locs, axis=0)
        locs = np.append(locs, np.array([[min_x, min_y], [min_x, max_y],[max_x, min_y],[max_x, max_y]]),axis=0)
        for c in range(n_colors):
            feat_array_temp[c] = np.append(feat_array_temp[c], np.zeros((nSamples, 4)), axis=1)

    for i in xrange(nSamples):
        for c in range(n_colors):
            temp_interp[c][i, :, :] = griddata(locs, feat_array_temp[c][i, :], (grid_x, grid_y),
                                    method='cubic', fill_value=np.nan)
        print 'Interpolating {0}/{1}\r'.format(i+1, nSamples),

    for c in range(n_colors):
        if normalize:
            temp_interp[c][~np.isnan(temp_interp[c])] = \
                scale(temp_interp[c][~np.isnan(temp_interp[c])])
        temp_interp[c] = np.nan_to_num(temp_interp[c])

    return np.swapaxes(np.asarray(temp_interp), 0, 1)     # swap axes to have [samples, colors, W, H]


def build_cnn(input_var=None, W_init=None, n_layers=(4, 2, 1), n_filters_first=32, imSize=32):
    """
    Builds a VGG style CNN network followed by a fully-connected layer and a softmax layer.
    Stacks are separated by a maxpool layer. Number of kernels in each layer is twice
    the number in previous stack.
    input_var: Theano variable for input to the network
    outputs: pointer to the output of the last layer of network (softmax)

    :param input_var: theano variable as input to the network
    :param n_layers: number of layers in each stack. An array of integers with each
                    value corresponding to the number of layers in each stack.
                    (e.g. [4, 2, 1] == 3 stacks with 4, 2, and 1 layers in each.
    :param n_filters_first: number of filters in the first layer
    :param W_init: Initial weight values
    :param imSize: Size of the image
    :return: a pointer to the output of last layer
    """

    weights = []        # Keeps the weights for all layers
    count = 0
    # If no initial weight is given, initialize with GlorotUniform
    if W_init is None:
        W_init = [lasagne.init.GlorotUniform()] * sum(n_layers)

    # Input layer
    network = InputLayer(shape=(None, 3, imSize, imSize),
                                        input_var=input_var)
    # network = InputLayer(shape=(None,None, 3, imSize, imSize),  ###add a none
    #                                     input_var=input_var)

    for i, s in enumerate(n_layers):
        for l in range(s):
            network = Conv2DLayer(network, num_filters=n_filters_first * (2 ** i), filter_size=(3, 3),
                          W=W_init[count], pad='same')
            count += 1
            weights.append(network.W)
    network = MaxPool2DLayer(network, pool_size=(2, 2))


    return network, weights



def build_cnn_R(input_var=None, W_init=None, n_layers=(4, 2, 1), n_filters_first=32, imSize=32):
    """
    Builds a VGG style CNN network followed by a fully-connected layer and a softmax layer.
    Stacks are separated by a maxpool layer. Number of kernels in each layer is twice
    the number in previous stack.
    input_var: Theano variable for input to the network
    outputs: pointer to the output of the last layer of network (softmax)

    :param input_var: theano variable as input to the network
    :param n_layers: number of layers in each stack. An array of integers with each
                    value corresponding to the number of layers in each stack.
                    (e.g. [4, 2, 1] == 3 stacks with 4, 2, and 1 layers in each.
    :param n_filters_first: number of filters in the first layer
    :param W_init: Initial weight values
    :param imSize: Size of the image
    :return: a pointer to the output of last layer
    """

    weights = []        # Keeps the weights for all layers
    count = 0
    # If no initial weight is given, initialize with GlorotUniform
    if W_init is None:
        W_init = [lasagne.init.GlorotUniform()] * sum(n_layers)

    # Input layer
    network = InputLayer(shape=(None, 3, imSize, imSize),
                                        input_var=input_var)
    # network = InputLayer(shape=(None,None, 3, imSize, imSize),  ###add a none
    #                                     input_var=input_var)

    for i, s in enumerate(n_layers):
        for l in range(s):
            network = Conv2DLayer(network, num_filters=n_filters_first * (2 ** i), filter_size=(3, 3),
                          W=W_init[count], pad='same')
            count += 1
            weights.append(network.W)
    network = MaxPool2DLayer(network, pool_size=(2, 2))
    # convpooling using Max pooling over frames
    convpool = ElemwiseMergeLayer(network, theano.tensor.maximum)
    # A fully-connected layer of 512 units with 50% dropout on its inputs:
    convpool = DenseLayer(lasagne.layers.dropout(convpool, p=.5),
            num_units=512, nonlinearity=lasagne.nonlinearities.rectify)

    # And, finally, the output layer with 50% dropout on its inputs:
    convpool = lasagne.layers.DenseLayer(lasagne.layers.dropout(convpool, p=.5),
            num_units=nb_classes, nonlinearity=lasagne.nonlinearities.softmax)
    return convpool



def build_convpool_max(input_vars, nb_classes):
    """
    Builds the complete network with maxpooling layer in time.

    :param input_vars: list of EEG images (one image per time window)
    :param nb_classes: number of classes
    :return: a pointer to the output of last layer
    """
    convnets = []
    numTimeWin = input_vars.ndim
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

def build_convpool_conv1d(input_vars, nb_classes):
    """
    Builds the complete network with 1D-conv layer to integrate time from sequences of EEG images.

    :param input_vars: list of EEG images (one image per time window)
    :param nb_classes: number of classes
    :return: a pointer to the output of last layer
    """
    numTimeWin = input_vars.ndim
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


def build_convpool_lstm(input_vars, nb_classes, GRAD_CLIP=100):
    """
    Builds the complete network with LSTM layer to integrate time from sequences of EEG images.

    :param input_vars: list of EEG images (one image per time window)
    :param nb_classes: number of classes
    :param GRAD_CLIP:  the gradient messages are clipped to the given value during
                        the backward pass.
    :return: a pointer to the output of last layer
    """
    convnets = []
    numTimeWin = input_vars.ndim
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

def build_convpool_mix(input_vars, nb_classes, GRAD_CLIP=100):
    """
    Builds the complete network with LSTM and 1D-conv layers combined

    :param input_vars: list of EEG images (one image per time window)
    :param nb_classes: number of classes
    :param GRAD_CLIP:  the gradient messages are clipped to the given value during
                        the backward pass.
    :return: a pointer to the output of last layer
    """
    convnets = []
    #numTimeWin = input_vars.ndim
    numTimeWin = input_vars.ndim #rakib
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
            num_units=nb_classes, nonlinearity=lasagne.nonlinearities.softmax)
    return convpool

### add Rakib:
def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    #input_len = inputs.shape[1]
    input_len = inputs.shape[0] #Rakib
    assert input_len == len(targets)
    if shuffle:
        indices = np.arange(input_len)
        np.random.shuffle(indices)
    for start_idx in range(0, input_len - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]

###iterate_minibatches



if __name__ == '__main__':
#def main():

    #input_var = T.TensorType('floatX', ((False,) * 5))()        # Notice the () at the end
    #target_var = T.ivector('targets')

    ####################################################

    #Rakib
    augment = False                                 # Flag for data augmentation
    init_pars = False                               # Initialize parameters from convnet
    #filename = 'EEG_images_32_timeWin'              # Name of the .mat file containing EEG images
    filename ='Valnence'
    filename_aug = 'EEG_images_32_timeWin_aug_pca'  # Name of the .mat file containing augmented EEG images
    subjectsFilename = 'trials_subNums'             # Name of the .mat file containing trial/subject correspondence
    model = 'CNN'                                   # Model type selection
    print('Model type is : {0}'.format(model))
    num_epochs = 5                                  # Number of epochs for training
    imSize = 32                                     # Size of the images
    batch_size = 20                                 # Number of samples in each batch
    nb_classes = 2                                  # Number of classes
    numTimeWin = 7                                  # Number of time windows
    GRAD_CLIP = 100                                 # Clipping value for gradient clipping in LSTM
    #print("Loading data...")
    #data, labels = load_data(filename)
    #labels=-1*np.transpose(labels)

   # #star uncommentin
   #
    mat = scipy.io.loadmat(subjectsFilename, mat_dtype=True)
    subjNumbers = np.squeeze(mat['subjectNum'])     # subject IDs for each trial
    #### rakib

    #mat = scipy.io.loadmat('Neuroscan_locs_orig')
    mat = scipy.io.loadmat('C:\Users\Rakib\PycharmProjects\Theano\Sample data\Neuroscan_locs_orig')
    caploc=mat['A']
    new=[]
    for element in np.arange(0,caploc.shape[0]):
        #print "cc"
        new.append(caploc[element][0])
        new.append(caploc[element][1])
    #locs=np.asanyarray(new).reshape(32,2)
    locs=np.asanyarray(new).reshape(32,2)
    #print locs
    print 'location converted'

    mat = scipy.io.loadmat(subjectsFilename, mat_dtype=True)
    subjNumbers = np.squeeze(mat['subjectNum'])
    #data load

    mat1 = scipy.io.loadmat('Valence')
    data=mat1['data']
    #data=data[:,0:data.shape[1]-1]
    images = gen_images(locs,
                        data,
                        32, augment=True, pca=True, n_components=2)

    feature=data[:,0:data.shape[1]-1]
    labels=data[:,data.shape[1]-1]
    print images.ndim
    print images.shape
    data=images

    #end

    # input_var = T.TensorType('floatX', ((False,) * 4))()        # Notice the () at the end
    # target_var = T.ivector('targets')
    # images = gen_images(np.random.rand(32, 2),
    #                     np.random.rand(100, 129),
    #                     32, augment=True, pca=True, n_components=2)
    # data=images
    # labels=np.repeat([0,1],100)
    # subjNumbers=np.repeat(np.arange(1,11),10)

    ######################################################



    # images = gen_images(np.random.rand(10, 2),
    #                     np.random.rand(100, 31),
    #                     16, augment=True, pca=True, n_components=2)
    #network = build_cnn(input_var[0])
    #network = build_convpool_max(input_var, 2)
    #network = build_convpool_conv1d(input_var, 3)
    #network = build_convpool_lstm(input_var, 3, 90)
    #network = build_convpool_mix(input_var, 3, 90)
    #print 'Done!'
#def gen_images(locs, features, nGridPoints, normalize=True, augment=False, pca=False, stdMult=0.1, n_components=2, edgeless=False):

#######################################################################################################################################

#Rakib edit


    fold_pairs = []
    if augment:
        # Aggregate augmented data and labels
        #data_aug, labels_aug = load_data(filename_aug)
        #data = np.concatenate((data, data_aug), axis=1)
        #labels = np.vstack((labels, labels_aug))
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
        #input_var = T.TensorType('floatX', ((False,) * 5))()        # Notice the () at the end
        # input_var = T.TensorType('floatX', ((False,) *4))()   #rakib
        input_var = T.tensor4('floatX')
        target_var = T.ivector('targets')
        # Create neural network model (depending on first command line parameter)
        print("Building model and compiling functions...")
        # Building the appropriate model
        if model == '1dconv':
            network = build_convpool_conv1d(input_var,2)
        elif model == 'maxpool':
            network = build_convpool_max(input_var,2)
        elif model == 'lstm':
            network = build_convpool_lstm(input_var)
        elif model == 'mix':
            network = build_convpool_mix(input_var,2)# change
        elif model == 'CNN':
            network, _ = build_cnn(input_var)# R : change: build_cnn(input_var=None, W_init=None, n_layers=(4, 2, 1), n_filters_first=32, imSize=32)
            # 'CNN': return network, weights
        print "ok"
        print 'ok'
        # # convpooling using Max pooling over frames
        # convpool = ElemwiseMergeLayer(network, theano.tensor.maximum)
        # A fully-connected layer of 512 units with 50% dropout on its inputs:
        convpool = DenseLayer(lasagne.layers.dropout(network, p=.5),
                num_units=512, nonlinearity=lasagne.nonlinearities.rectify)

        # And, finally, the output layer with 50% dropout on its inputs:
        convpool = lasagne.layers.DenseLayer(lasagne.layers.dropout(convpool, p=.5),
                num_units=nb_classes, nonlinearity=lasagne.nonlinearities.softmax)

        network=convpool

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

        print '1 ok'
        print '1 ok'
        # Compile a function performing a training step on a mini-batch (by giving
        # the updates dictionary) and returning the corresponding training loss:
        train_fn = theano.function([input_var, target_var], loss, updates=updates)####find solution ...problem..............

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
            ##
            #input_len = X_train.shape[1]
            #assert input_len == len(y_train)
            #print X_train.shape[1]
            print X_train.shape[0] #Rakib
            print len(y_train)
            print 'before batch'
            for batch in iterate_minibatches(X_train, y_train, batch_size, shuffle=False):
                inputs, targets = batch
                print "ok before function"
                train_err += train_fn(inputs, targets)
                train_batches += 1
            print 'Batch finish'
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
            print 'ok'

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

