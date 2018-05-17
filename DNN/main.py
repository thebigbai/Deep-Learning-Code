import time
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage

from tools import *
from DNN import *

if __name__ == '__main__':
    iteration_num=2400
    learning_rate=0.0075

    #load data
    train_x_orig,train_y,test_x_orig,test_y, classes=load_data()
    train_x=process_data(train_x_orig)
    test_x=process_data(test_x_orig)

    #set up NN structure, 5 layers (1 input, 3 hidden and 1 output)
    nn_layers=[train_x.shape[0],20,7,5,1]
    nn_actication=["relu","relu","relu","sigmoid"]
    
    param=init_param(nn_layers) #init W and b
    costs=[]

    for i in range(iteration_num):
        A_cache,Z_cache=forward(train_x,param)  
        cost=compute_cost(A_cache,train_y)
        grads=backward(train_y,A_cache,Z_cache,param)
        param=update_param(param,grads,learning_rate)

        if i%100 == 0:
            print ("iterate:",i," cost: ",cost)
            costs.append(cost)

    accracy_train=predict(train_x,train_y,param)
    accracy_test=predict(test_x,test_y,param)
    f=open("param.txt","a+")
    keys=param.keys()
    for key in keys:
        shape=param[str(key)].shape
        value=param[str(key)]
        f.write(str(key)+"\n")
        f.write(str(shape)+"\n")
        f.write(str(value)+"\n")
        f.write("#===============================================#\n")
    f.close()

    print ("accuracy on training set: ",accracy_train)
    print ("accuracy on test set: ",accracy_test)


        
       

       