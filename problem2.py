import numpy as np
import helpers
import MLPtrain
import MLPtest

Z,W,V = MLPtrain.MLPTrain("optdigits_train.txt","optdigits_valid.txt",10,18)

Z = MLPtest.MLPTest("optdigits_test",W,V)