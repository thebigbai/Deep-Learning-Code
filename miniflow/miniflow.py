# -*- coding: utf-8 -*-
"""
Created on Thu Apr 04 15:15:49 2019

@author: Y. Wang
"""
import numpy as np


class Neuron:
    def __init__(self,inbound_neurons=[]):
        #the list that this neuron recieves values
        self.inbound_neurons=inbound_neurons
        #the list that this neuron passes values
        self.outbound_neurons=[]
        # the value that this nueron will output
        self.value=None
        #gradients
        self.gradients={}
        
        for n in self.inbound_neurons:
            n.outbound_neurons.append(self)
            
    def forward(self):
        raise NotImplemented
        #return NotImplemented
    
    def backward(self):
        raise NotImplemented

class Input_Neuron(Neuron):
    def __init__(self):
        Neuron.__init__(self)
    
    def forward(self,value=None):
        if value is not None:
            self.value=value
    
    def backward(self):
        self.gradients = {self: 0}
        #input neuron do nothing
        for n in self.outbound_neurons:
            grad=n.gradients[self]
            self.gradients[self]+=grad
    

class Add_Neuron(Neuron):
    def __init__(self,*x):
        Neuron.__init__(self,list(x))
    
    def forward(self):
        self.value=0
        for i in self.inbound_neurons:
            self.value+=i.value
        
        
class Mul_Neuron(Neuron):
    def __init__(self,*x):
        Neuron.__init__(self,list(x))
    
    def forward(self):
        self.value=1
        for i in self.inbound_neurons:
            self.value=self.value*i.value

# calc wx+b            
class Linear_Neuron(Neuron):
    def __init__(self,x,w,b):
        Neuron.__init__(self,[x,w,b])
        
    def forward(self):
        x_vector=self.inbound_neurons[0].value
        w_vector=self.inbound_neurons[1].value
        b=self.inbound_neurons[2].value
        """
        x is m by n matrix,the m is num of samples, 
          the n is the num of node in this layer
        w is n by k matrix, the n is the num of node in this layer,
          the k is the num of node in next layer
        """
        assert x_vector.shape[1]==w_vector.shape[0]

        Sum=np.dot(x_vector,w_vector)+b
        self.value=Sum
        
    def backward(self):
        #init to 0
        self.gradients={n: np.zeros_like(n.value) for n in self.inbound_neurons}

        for n in self.outbound_neurons:
            grad=n.gradients[self]
            """
            L=WX+b
            dL/dx=dW
            dL/dx=dX
            dL/db=1
            obviously, grad.shape == L.shape
            X.shape(n,m) where n is the num of node in this layer, m is sample nums
            W.shape(m,k) where k is the num of node in next layer             
            So, L.shape==(n,k)
            when backward, the matrix should keep the same as forward process
            so (n,k) dot (k,m)--->the W need to be transposed
            (m,n) dot (n,k) --->the X need to be transposed
            """
            self.gradients[self.inbound_neurons[0]]+=np.dot(grad,self.inbound_neurons[1].value.T)
            self.gradients[self.inbound_neurons[1]]+=np.dot(self.inbound_neurons[0].value.T,grad)
            self.gradients[self.inbound_neurons[2]]+=np.sum(grad,axis=0,keepdims=False)
            #personly think here shoule be divided by sample nums. using np.mean instead of np.sum
        

class Sigmoid_Neuron(Neuron):
    def __init__(self,z):
        Neuron.__init__(self,[z])
        
    def sigmoid(self,z):
        return 1./(1.+np.exp(-z))
        
    def forward(self):
        z=self.inbound_neurons[0].value
        a=self.sigmoid(z)
        self.value=a
    
    def backward(self):
        #init to 0
        self.gradients={n: np.zeros_like(n.value) for n in self.inbound_neurons}
        """
        sigmiod'(x)=sigmoid(x)(1-sigmoid(x))
        """
        for n in self.outbound_neurons:
            #grad is the gradient from backpropagation of next layer
            grad=n.gradients[self]
            self.gradients[self.inbound_neurons[0]]+=self.value*(1-self.value)*grad
        
        
#cost function
class MSE(Neuron):
    def __init__(self,y,a):
        Neuron.__init__(self,[y,a])
    
    def forward(self):
        #convert to col vector
        y=self.inbound_neurons[0].value.reshape(-1,1)
        a=self.inbound_neurons[1].value.reshape(-1,1)
        self.m=y.shape[0]  
        
        assert y.shape==a.shape

        self.diff=y-a
        self.value=1./self.m*(np.sum(self.diff**2))
        
    def backward(self):
        """
        C=1/m*(y-a)**2
        dC/dy=2/m*(y-a)
        dC/da=-2/m(y-a)
        """
        self.gradients = {n:np.zeros_like(n.value) for n in self.inbound_neurons}
        self.gradients[self.inbound_neurons[0]]=(2./self.m)*self.diff
        self.gradients[self.inbound_neurons[1]]=(-2./self.m)*self.diff
        
        
        
        
        
        
        