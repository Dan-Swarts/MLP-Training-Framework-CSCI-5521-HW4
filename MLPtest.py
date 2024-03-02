import numpy as np
import helpers
import MLPtrain

def MLPTest(test_data, W, V):

    # load and normalize data
    x_test, y_test \
    = helpers.ReadNormalizedTestingData(
        test_data
    )

    # H is the number of intermediate values:
    H = W.shape[1]
    # K is the number of outputs:
    K = V.shape[1]
    # D stores the size of an input 
    D = x_test.shape[1]
    # N stores the number of inputs:
    N = x_test.shape[0]
    # create space to store intermediat values: 
    Z = np.zeros(shape=(N,H),dtype=np.float32)

    batch_size = helpers.batch_size

    average = 0.0

    # create space to store the outputs:
    Y = np.zeros(shape=(batch_size,K),dtype=np.float32)

    for i in range(0, N, batch_size):
        # clarify variables within the domain of the batch
        x_batch = x_test[i:i+batch_size]

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

        x_batch = x_test[i:i+b]
        r_batch = y_test[i:i+b]

        error = helpers.find_error(r_batch,Y)
        print(f"Testing Error: \n{error} Average: {np.mean(error)}")

        average += np.mean(error)

    average /= (N / batch_size)

    print(f"\nFINAL AVERAGE ON TESTING DATA: \n{average}")

    return Z
