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

def derivAct(x, choice):
	if choice==RELU:#derivative of Relu defined to be 0 at X<0, .5 @ x=0, 1 @ x>0
		return np.where(x<np.zeros(len(x)),0,np.where(x==np.zeros(len(x)),.5,1))
	else: #derivative of sigmoid
		return (sig(x)*(1-sig(x)))

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
	record = [acc] #will keep track of all computation outputs after weights and biases before activation
	for i in range(net['nLayers']-1):
		acc = net['Weight'+str(i)]*acc + net['bias'+str(i)] 
		record.append(acc)
		if net['act'+str(i)] == RELU:
			acc = relu(acc)
		else:
			acc = sig(acc)
	if keepZ:
		return acc, np.array(record) #only returns the Z records if learning, not when simply predicting
	else:
		return acc

def cost(yHat, y): #defining cost as the sum of losses
	return (1 / y.shape[1]) * sum([yHatI - yI for yHatI, yI in zip(yHat, y)])

def derivs(net, record):
	LAYERS = net['nLayers']
	dlL = 1/LAYERS
	daL = dlL#derivative of loss w respect to aL is 1
	dzL = daL*derivAct(record[LAYERS-1],0)
	dbL = 1/LAYERS*np.sum(dzL, axis=1, keepdims=True)
	dwL = 1/LAYERS*dzL*np.transpose(net['act'+str(L-1)])
