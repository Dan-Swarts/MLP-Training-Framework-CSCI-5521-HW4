import numpy as np

# You can play with these parameters
batch_size = 64
epochs = 2

# the number of classes
num_classes = 10

# input image dimensions
img_rows, img_cols = 8, 8

# needed activation functions:
leaky_ReLU = lambda x: x if x >= 0 else 0.01 * x
sigmoid = lambda x: 1 / (1 + np.exp(-x))

"""
    Modified version of ReadNormalizedOptdigitsDataset, which returns only the training and validation datasets

    Parameters:
    -----------
    training_filename: path to training data

    validation_filename: path to validation data

    Returns: the normalized data and y values
    --------
"""
def ReadNormalizedTrainingData(training_filename, validation_filename):

    # read files
    training_data = np.loadtxt(training_filename, delimiter=',')
    validation_data = np.loadtxt(validation_filename, delimiter=',')

    # prepare training data output
    X_trn = training_data[:, 0:-1]
    y_trn = training_data[:, -1].astype(np.int8)
    
    # prepare validation data output
    X_val = validation_data[:, 0:-1]
    y_val = validation_data[:, -1].astype(np.int8)
    

    # mean and std
    mu = np.mean(X_trn, axis=0)
    s = np.std(X_trn, axis=0)
    
    # normalize data
    X_trn_norm = (X_trn - np.tile(mu, (np.shape(X_trn)[0], 1))) / np.tile(s, (np.shape(X_trn)[0], 1))
    X_val_norm = (X_val - np.tile(mu, (np.shape(X_val)[0], 1))) / np.tile(s, (np.shape(X_val)[0], 1))
    
    # fix nan
    X_trn_norm[np.isnan(X_trn_norm)]=0
    X_val_norm[np.isnan(X_val_norm)]=0
    
    return X_trn_norm, y_trn, X_val_norm, y_val

"""
    Modified version of ReadNormalizedOptdigitsDataset, which returns only the testing dataset

    Parameters:
    -----------
    test_filename: path to testing data

    Returns: the normalized data and y values
    --------
"""
def ReadNormalizedTestingData(test_filename):

    # read files
    test_data = np.loadtxt(test_filename, delimiter=',')
    
    # prepare test data output
    X_tst = test_data[:, 0:-1]
    y_tst = test_data[:, -1].astype(np.int8)

    # mean and std
    mu = np.mean(X_tst, axis=0)
    s = np.std(X_tst, axis=0)
    
    # normalize data
    X_tst_norm = (X_tst - np.tile(mu, (np.shape(X_tst)[0], 1))) / np.tile(s, (np.shape(X_tst)[0], 1))
    
    # fix nan
    X_tst_norm[np.isnan(X_tst_norm)]=0
    
    return X_tst_norm, y_tst

"""
    Changes the classification to be compatible with the output vector

    Parameters:
    -----------
    r_value: unsigned integer. Specifies which of the classifications an input belongs to.

    K: unsigned integer. Specifies how many classifications exist within the model

    Returns: array of size K, where everything is a zero exept the index of the classification, which is a 1.
    --------
"""
def reshape_expected_value(r_value,K):
    output = np.zeros(shape=K)
    output[r_value] = 1
    return output

"""
    Update the weights in the inner and outer layer of the MLP.

    Parameters:
    -----------
    L: scalar. Determines sensitivity to change

    W: D by H array of weights. Contains the weights of each input to each perceptron

    V: H by K array of weights. Contains the weights of each hidden step to each output

    Z: batch_size by H array of floats. Contains the intermediary values for each batch input

    R: batch_size array of floats. Contains the true classifications for each input of the batch

    Y: batch_size by K array of floats. Contains the calculated classifications for each input of the batch

    X: batch_size by 

    Returns: updated weights W and V
    --------
"""
# update the weights of a batch according to the error function provided
def update_weights(L,W,V,Z,R,Y,X):

    # derive variables:
    H = W.shape[1] # H is the number of perceptrons in the hidden layer
    D = X.shape[1] # D is the size of an input vector
    K = V.shape[1] # K is the number of outputs, aka the number of classifications
    batch_size = R.shape[0] # batch_size is the number of inputs in this batch

    # store changes to the weights:
    delta_weight = np.zeros(shape=(D+1, H),dtype=np.float32)

    # update all weights for each datapoint in the batch:
    for t in range(batch_size):
        # reshapes the true output from a scalar classification to a vector for compatability with the output vector
        r_vector = reshape_expected_value(R[t],K)

        for h in range(H):
            for d in range(D):
                # first calculate the summation:
                sum = 0
                for k in range(K):
                    sum += (r_vector[k] - Y[t][k]) * V[h][k]

                # then determine leaky-ness:
                dot = np.dot(W[1:,h],X[t])
                if (dot < 0):
                    # finally, add the change
                    delta_weight[k][h] += 0.01 * L * sum * X[t][d] 
                else:
                    delta_weight[k][h] += L * sum * X[t][d] * Z[t][h]

    # looking at the average change over the entire batch
    delta_weight /= batch_size
    # update the inner weights
    W += delta_weight

    # now update outer weights: 
    delta_v = np.zeros(shape=(H+1, K),dtype=np.float32)

    for t in range(batch_size):
        r_vector = reshape_expected_value(R[t],K)
        for k in range(K):
            for h in range(H):
                delta_v[h][k] += L * (r_vector[k] - Y[t][k]) * Z[t][h]

    # looking at the average change over the entire batch
    delta_v /= batch_size
    # update the inner weights
    V += delta_v

    return W,V

"""
    Find the error between expected and calculated values. uses binary error for each of K classifications.

    Parameters:
    -----------
    R: batch_size array of floats. Contains the true classifications for each input

    Y: batch_size by K array of floats. Contains the calculated classifications for each input of the batch

    Returns: an error value for each classifications 
    --------
"""
def find_error(R,Y):
    # derive variables:
    K = Y.shape[1] 
    batch_size = R.shape[0]
    error = np.zeros(shape=K)

    # Sums error for each input in the batch: 
    for t in range(batch_size):
        r_vector = reshape_expected_value(R[t],K)
        for k in range(K):
            error[k] += r_vector[k] * np.log(Y[t][k]) + (1 - r_vector[k]) * np.log(1 - Y[t][k])

    # finds average error
    error /= batch_size
    return error