import h5py
import numpy as np

def load_data():
    train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

def process_data(img_data):
    imgNum=img_data.shape[0]
    vectorImg=img_data.reshape(imgNum,-1).T
    vectorImg=vectorImg/255.0   #important to use 255.0 to make it float
    return vectorImg

def relu(Z):
    A = np.maximum(0,Z)
    assert(A.shape == Z.shape)
    return A

def partial_relu(Z):
    Z[Z<=0]=0
    Z[Z>0]=1
    pZ=Z
    assert(pZ.shape == Z.shape)
    return pZ

def sigmoid(Z):
    A=1/(1+np.exp(-Z))
    assert(A.shape==Z.shape)
    return A

def partial_sigmoid(Z):
    A=sigmoid(Z)
    pZ=A*(1-A)
    assert(pZ.shape == Z.shape)
    return pZ
