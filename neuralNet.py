import random as rnd
import numpy as np 
import time
print("hello world :) . . . watch out")

RELU=1
SIGMOID=0
#choosing sigmoid to be the default because otherwise a neural net of all relu would
#be a linear function

def relu(x):
	return max(np.zeros(len(x)),x)
	#no need to worry about vectors thanks to broadcasting

def sig(x):
	return 1/(1+np.exp(-x))
	#we love broadcasting

def neuralNetInit(neuronsInlayers):
	#that neurons in layers will carry a list of the count of neurons in each layer
	#neuronsInLayers carries tuples throughout, the first member being the neuron count per layer
	#and the second member to denote the activation function
	#e.g.: [(4,0),(5,0),(3,1)]
	neuralNet = {}
	for i in range(len(neuronsInLayers)-1):
		neuralNet['Weight'+str(i)] = np.rnd.randn(neuronsInlayers[i+1][0],neuronsInlayers[i][0])*.00714 #lucky number
		#making the weights matrix0
		neuralNet['bias'+str(i)] = np.zeros(neuronsInlayers[i+1][0],1)
		#making bias vectors
		neuralNet['act'+str(i)] = neuronsInlayers[i+1][1]
		#storing preferred activation function
	neuralNet['nLayers'] = len(neuronsInlayers) #not bad to have the # layers easily available
	return neuralNet

def forwardFeed(net, inputVec, keepZ=False):
	acc = inputVec
	record = [] #will keep track of all computation outputs after weights and biases before activation
	for i in range(net['nLayers']-1):
		acc = net['Weight'+str(i)]*acc + net['bias'+str(i)] 
		record.append(acc)
		if net['act'+str(i)] == RELU:
			acc = relu(acc)
		else:
			acc = sig(acc)
	return acc



