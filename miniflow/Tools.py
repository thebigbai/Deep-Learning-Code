# -*- coding: utf-8 -*-
"""
Created on Mon Apr 08 13:01:01 2019

@author: THICV
"""
from miniflow import *



# Kahn's Algorithm    
def topological_sort(feed_dict):
    input_neurons=[n for n in feed_dict.keys()]  
    Graph={}
    #neurons=input_neurons
    neurons=[n for n in input_neurons]
    
    #constrcut the graph
    while len(neurons)>0:
        n=neurons.pop(0) #pop the 1st node
        if n not in Graph:
            Graph[n]={"in":set(),"out":set()}   
        for m in n.outbound_neurons:
            if m not in Graph:
                Graph[m]={"in":set(),"out":set()}
            Graph[n]["out"].add(m)
            Graph[m]["in"].add(n)
            neurons.append(m)
    
    L=[]
    S=set(input_neurons)
    while len(S)>0:
        n=S.pop()
        if isinstance(n,Input_Neuron):
            n.value=feed_dict[n]
        L.append(n)
        
        #record the predecessor and successor
        for m in n.outbound_neurons:
            Graph[n]["out"].remove(m)
            Graph[m]["in"].remove(n)
            
            if len(Graph[m]["in"])==0:
                S.add(m)  
    return L


def forward_pass(output,graph):
    for i in graph:
        i.forward()
    output.forward()
    return output.value
    
    
def forward_and_backward(graph):
    for n in graph:
        n.forward()
    #print n.value
    
    for n in graph[::-1]:
        n.backward()
    #print "gradients: ",n.gradients
        
def update_param(trainables,learning_rate=1e-2):
    for t in trainables:
        #print t.gradients
        partial=t.gradients[t]
        t.value-=learning_rate*partial
        
        
        
        
        
        
        
        
        