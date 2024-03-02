import numpy as np
import helpers

def MLPTrain(train_data, val_data, K, H):

    # load and normalize data
    x_train, y_train, x_valid, y_valid \
    = helpers.ReadNormalizedTrainingData(
        train_data,
        val_data
    )

    # D stores the size of an input 
    D = x_train.shape[1]
    # N stores the number of inputs:
    N = x_train.shape[0]
    # create space to store intermediat values: 
    Z = np.zeros(shape=(N,H),dtype=np.float32)
    # randomly initialize input weights: 
    W = np.random.uniform(low=-0.01, high=0.01, size=(D+1, H))
    # randomly initialize hidden weights:
    V = np.random.uniform(low=-0.01, high=0.01, size=(H+1, K))

    batch_size = helpers.batch_size

    # create space to store the outputs of every batch (needed for updating the weights):
    Y = np.zeros(shape=(batch_size,K),dtype=np.float32)
    b = 0

    num_epochs = helpers.epochs

    for epoch in range(num_epochs):
        for i in range(0, N, batch_size):
            # clarify variables within the domain of the batch
            x_batch = x_train[i:i+batch_size]

            b = 0
            for index, x_t in enumerate(x_batch):
                t = index + i

                # calculate intermideate values
                for h in range(H):
                    dot = np.dot(x_t,W[1:,h]) + W[0,h]
                    Z[t][h] = helpers.leaky_ReLU(dot)
                
                # calculate output values:
                for k in range(K):
                    dot = np.dot(Z[t],V[1:,k]) + V[0,k]
                    Y[index][k] = helpers.sigmoid(dot)
                b += 1

            x_batch = x_train[i:i+b]
            r_batch = y_train[i:i+b]
            z_batch = Z[i:i+b]

            error = helpers.find_error(r_batch,Y)
            print(f"Testing Error: \n{error} Average: {np.mean(error)}")
            # the values for the batch have been found. now update the weights accordingly: 
            W,V = helpers.update_weights(0.5,W,V,z_batch,r_batch,Y,x_batch)


    # Validation: 

    # temporary z storage for validation purposes
    z_valid = np.zeros(shape=(H)) 

    for i in range(0, len(x_valid), batch_size):
        # clarify variables within the domain of the batch
        x_batch = x_valid[i:i+batch_size]

        b = 0
        for index, x_t in enumerate(x_batch):

            # calculate intermideate values
            for h in range(H):
                dot = np.dot(x_t,W[1:,h]) + W[0,h]
                z_valid[h] = helpers.leaky_ReLU(dot)
            
            # calculate output values:
            for k in range(K):
                dot = np.dot(z_valid,V[1:,k]) + V[0,k]
                Y[index][k] = helpers.sigmoid(dot)
            b += 1

        x_batch = x_train[i:i+b]
        r_batch = y_train[i:i+b]
        z_batch = Z[i:i+b]

        error = helpers.find_error(r_batch,Y)
        print(f"Testing Error: \n{error} Average: {np.mean(error)}")

    return Z,W,V


