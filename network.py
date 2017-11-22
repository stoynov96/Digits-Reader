import numpy as np
import random

class Network(object):

	def __init__(self, layer_sizes):
		self.num_layers = len(layer_sizes)
		self.layer_sizes = layer_sizes
		self.biases = [np.random.randn(layer_size, 1) for layer_size in layer_sizes[1:]]
		self.weights = [np.random.randn(layer_size, prev_layer_size) for layer_size, prev_layer_size in zip(layer_sizes[1:], layer_sizes[:-1])]

	def feedforward(self, activation):
		"""
		Feeds a given input through the matrix and returns the resulting output vector
		"""
		for w,b in zip(self.weights, self.biases):
			activation = sigmoid(np.dot(w,activation) + b)
		return activation

	def SGD(self, training_data, epochs, mini_batch_size, learning_rate, test_data):
		"""
		Applies Stochastic Gradient Descent to learn over a specified number of epochs.
		Prints detailed progress report if test data is provided
		"""

		training_data = list(training_data)
		train_data_length = len(training_data)

		if test_data:
			test_data = list(test_data)

		for epoch in range(epochs):
			random.shuffle(training_data)
			for i in range(0, train_data_length, mini_batch_size):
				mini_batch = training_data[i:i+mini_batch_size]
				self.update_mini_batch(mini_batch, learning_rate)

			if test_data:
				print ( "Epoch {0}: {1} / {2} correct".format(epoch, self.evaluate(test_data), len(test_data)) )
			else:
				print ( "Epoch {0} completed...".format(epoch) )

	def update_mini_batch(self, mini_batch, learning_rate):
		"""
		Updates a mini batch based on a small step from a gradient descent towards a local minimum of the cost function
		"""
		mini_batch_size = len(mini_batch)

		total_delta_b = [np.zeros(b.shape) for b in self.biases]
		total_delta_w = [np.zeros(w.shape) for w in self.weights]

		for x,y in mini_batch:
			delta_b, delta_w = self.backprop(x,y)
			total_delta_b = [tdb+db for tdb,db in zip(total_delta_b, delta_b)]
			total_delta_w = [tdw+dw for tdw,dw in zip(total_delta_w, delta_w)]

		self.biases = [b - (learning_rate/mini_batch_size)*tdb for b,tdb in zip(self.biases, total_delta_b)]
		self.weights = [w - (learning_rate/mini_batch_size)*tdw for w,tdw in zip(self.weights, total_delta_w)]


	def backprop(self, x, y):
		# Initialize
		gradient_w = [np.zeros(w.shape) for w in self.weights]
		gradient_b = [np.zeros(b.shape) for b in self.biases]

		# Feed forward
		activation = x
		activations = [x]
		zs = []

		for w,b in zip(self.weights, self.biases):
			z = np.dot(w,activation) + b
			zs.append(z)
			activation = sigmoid(z)
			activations.append(activation)

		# Calculate last layer errors
		error = self.cost_derivative(activation, y) * sigmoid_prime(z)
		gradient_b[-1] = error
		gradient_w[-1] = np.dot(error, activations[-2].transpose())

		# print(activations)

		# Backpropagade the error

		for layer in range(2, self.num_layers):
			error = np.dot(self.weights[-layer+1].transpose(), error) * sigmoid_prime(zs[-layer])

			# Get biases and weights gradients from the calculated errors
			gradient_b[-layer] = error
			gradient_w[-layer] = np.dot(error, activations[-layer-1].transpose())

		# Return gradients
		return gradient_b, gradient_w


	def evaluate(self, test_data):
		"""
		Returns the number of digits in the test data for which the network's guess corresponded with the correct output
		"""
		# Get a test_results list of tuples - the network's guess and correct output for each test case in test_data
		test_results = [ (np.argmax(self.feedforward(x)), y) for x,y in test_data ]

		# Return the number of times the network's guess was identical to the correct output for each case in test_data (correct outputs now transferred to test_results)
		return sum(int(x == y) for x,y in test_results)

	def cost_derivative(self, output_activations, y):
		"""
		Returns a cost derivative vector for a cost function C = 1/2 sum( (y - A)^2 )
		Note: returns a cost derivative for a single training example
		"""
		return (output_activations - y)


def sigmoid(z):
	return 1.0/(1.0 + np.exp(-z))

def sigmoid_prime(z):
	sig = sigmoid(z)
	return sig * (1-sig)


