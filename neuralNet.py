import random 

def dotProd(uno, dos):
	sum = 0
	for i,j in zip(uno,dos):
		sum+= i*j
	return sum

def reLu(input):
	#Rectified Linear Unit, easier derivative than sigmoid
	return max(0,input) 

def derReLu(input):
	if input > 0:
		return 1
	elif input <0:
		return 0
	else:
		return .5


class Neuron:
	def __init__(self, nIn=0):
		self.weights = [(random.random()*2)-1 for _ in range(nIn)]
		self.bias = (random.random()*2)-1
		self.output = None
	def calc(self, inputs):
		self.output = reLu(dotProd(inputs, self.weights) + self.bias) 
		#output is dot product of inputs and weights for neuron + neuron's bias
		return self.output

class Layer:
	def __init__(self, nNeur, nIn):
		self.neurons = [Neuron(nIn) for _ in range(nNeur)]
		self.output = None
	def calc(self, inVect = []):
		self.output = [i.calc(inVect) for i in self.neurons]
		return self.output
	def dadRelu(self, prevOutput):
		return [1] * len(self.neurons)
	def dzdb(self, prevOutput):
		return [1] * len(self.neurons)
	def dzdw(self, prevOutput):
		return prevOutput
	def dzda(self,prevOutput): 
		#the derivative of a linear function multiplying a vector by a matrix with respect to said vector
		#is the matrix transposed
		return [[self.neurons[i].weights[j] for i in range(len(self.neurons))] for j in range(len(self.neurons[0].weights))]


class NNet:
	def __init__(self, neursLayers):
		self.layers = [Layer(neursLayers[i], neursLayers[i-1]) for i in range(len(neursLayers))[1:]]
		self.nInputs = neursLayers[0]
	def fit(self, inVects=[], corrects=[], epochs=1):
		for _ in range(epochs):
			for inVect, goal in zip(inVects, corrects):
				inputVector = [u for u in inVect]
				acc = [u for u in inVect]
				for i in self.layers:
					acc = i.predict(acc)
				self._learn(inputVector, acc, 0.0625, goal)
	def predict(self, inVect = []):
		for i in self.layers:
			inVect = i.calc(inVect) 
		return inVect
	def _learn(self, inputVector, outputVector, step, goal):
		for i in goal:
			self.output


			