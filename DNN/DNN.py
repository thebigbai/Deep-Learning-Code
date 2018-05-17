import time
import numpy as np


from tools import *


def init_param(nn_layers):
    np.random.seed(1)
    param={}

    for i in range(1,len(nn_layers)):
        param["W"+str(i)]=np.random.randn(nn_layers[i],nn_layers[i-1])/np.sqrt(nn_layers[i-1])
        param["b"+str(i)]=np.zeros((nn_layers[i],1))

        assert(param["W"+str(i)].shape==(nn_layers[i],nn_layers[i-1]))
        assert(param["b"+str(i)].shape==(nn_layers[i],1))


    return param

def linear_forward(A,W,b):
    Z=np.dot(W,A)+b
    return Z

def activate_forward(Z,mode):
    if mode=="relu":
        A=relu(Z)
    if mode =="sigmoid":
        A=sigmoid(Z)
    
    return A

def forward(X,param):
    A_cache={}
    Z_cache={}
    L=len(param)//2
    A=X
    A_cache["A0"]=A
    for i in range(L):
        if i+1 == L: 
            W=param["W"+str(i+1)]
            b=param["b"+str(i+1)]
            Z=linear_forward(A,W,b)
            A=activate_forward(Z,"sigmoid")
        else:
            W=param["W"+str(i+1)]
            b=param["b"+str(i+1)]
            Z=linear_forward(A,W,b)
            A=activate_forward(Z,"relu")
        
        A_cache["A"+str(i+1)]=A
        Z_cache["Z"+str(i+1)]=Z
    
    return A_cache,Z_cache

def compute_cost(A_cache,Y):
    L=len(A_cache)-1
    A=A_cache["A"+str(L)]
    m=Y.shape[1]

    cost=(-1.0/m)*np.sum(Y*np.log(A)+(1-Y)*np.log(1-A)) # use 1.0 is very important
    #cost = (1./m) * (-np.dot(Y,np.log(A).T) - np.dot(1-Y, np.log(1-A).T))
    
    return cost

def activate_backward(dA,Z,mode):
    if mode=="sigmoid":
        pZ=partial_sigmoid(Z)
        dZ=dA*pZ
    if mode=="relu":
        pZ=partial_relu(Z)
        dZ=dA*pZ
    
    return dZ

def linear_backward(dZ,A_pre,W):
    m=dZ.shape[1]
    dW=np.dot(dZ,A_pre.T)/m
    dB=np.sum(dZ,axis=1,keepdims=True)/m
    dA=np.dot(W.T,dZ)

    assert(dA.shape==A_pre.shape)
    assert(dW.shape==W.shape)

    return dA,dW,dB



def backward(Y,A_cache,Z_cache,param):
    #print Z_cache.keys()
    L=len(A_cache)-1
    A=A_cache["A"+str(L)]
    assert(Y.shape==A.shape)
    dA=-((Y/A)-((1-Y)/(1-A)))
    grads={}
    grads["dA"+str(L)]=dA
    #dA = - (np.divide(Y, A) - np.divide(1 - Y, 1 - A))
    
    for i in reversed(range(L)):
        if i+1==L:
            Z=Z_cache["Z"+str(i+1)]
            dZ=activate_backward(dA,Z,"sigmoid")
            A_pre=A_cache["A"+str(i)]
            W=param["W"+str(i+1)]
            dA,dW,dB=linear_backward(dZ,A_pre,W)
        else:
            Z=Z_cache["Z"+str(i+1)]
            dZ=activate_backward(dA,Z,"relu")
            A_pre=A_cache["A"+str(i)]
            W=param["W"+str(i+1)]
            dA,dW,dB=linear_backward(dZ,A_pre,W)
        
        grads["dW"+str(i+1)]=dW
        grads["dB"+str(i+1)]=dB
        grads["dA"+str(i+1)]=dA

    return grads

def update_param(param,grads,learning_rate):
    L=len(param)//2

    for i in range(L):
        param["W"+str(i+1)]=param["W"+str(i+1)]-learning_rate*grads["dW"+str(i+1)]
        param["b"+str(i+1)]=param["b"+str(i+1)]-learning_rate*grads["dB"+str(i+1)]
    
    return param


def predict(X,Y,param):
    m=X.shape[1]
    p = np.zeros((1,m))
    A_cache,Z_cache=forward(X,param)
    L=len(param)//2
    prob=A_cache["A"+str(L)]

    postive=0
    negtive=0
    true=0
    false=0
    for i in range(0,prob.shape[1]):
        if prob[0,i] > 0.5:
            p[0,i] = 1
        else:
            p[0,i] = 0
    
    accuracy=np.sum((p == Y)/m)

    return accuracy




            


    

        

    
    
    