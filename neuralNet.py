import random as rnd
import numpy as np 
import time
print("hello world :) . . . watch out")

RELU=1
SIGMOID=0
#choosing sigmoid to be the default because otherwise a neural net of all relu would
#be a linear function

def relu(x):
	return max(0,x)

def sig(x):
	return 1/(1+np.exp(-x))

def neuralNetInit(inputs, neuronsInlayers):
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
		neuralNet['activation'+str(i)] = neuronsInlayers[i+1][1]
		#storing preferred activation function
	return neuralNet




