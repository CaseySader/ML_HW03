from sklearn.datasets import fetch_openml
from sklearn.metrics import classification_report
import numpy as np
import sys
import getopt

def sigmoid(z):
	s = 1. / (1. + np.exp(-z))
	return s

def compute_loss(Y, Y_hat):

	L_sum = np.sum(np.multiply(Y, np.log(Y_hat)))
	m = Y.shape[1]
	loss_average = -(1./m) * L_sum
	return loss_average

def forward_propagation(X, params):

	cache = {}

	cache["Z1"] = np.matmul(params["W1"], X) + params["b1"]
	cache["A1"] = sigmoid(cache["Z1"])
	cache["Z2"] = np.matmul(params["W2"], cache["A1"]) + params["b2"]
	cache["A2"] = np.exp(cache["Z2"]) / np.sum(np.exp(cache["Z2"]), axis=0)
	return cache

def backward_propagation(X, Y, params, cache, m_batch):

	dZ2 = cache["A2"] - Y
	dW2 = (1./m_batch) * np.matmul(dZ2, cache["A1"].T)
	db2 = (1./m_batch) * np.sum(dZ2, axis=1, keepdims=True)

	dA1 = np.matmul(params["W2"].T, dZ2)
	dZ1 = dA1 * sigmoid(cache["Z1"]) * (1 - sigmoid(cache["Z1"]))
	dW1 = (1./m_batch) * np.matmul(dZ1, X.T)
	db1 = (1./m_batch) * np.sum(dZ1, axis=1, keepdims=True)

	gradients = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}
	return gradients

def main(args):
	# initialize hyperparameters to values
	dataset = ""
	n_h = 64
	learning_rate = 0.1
	batch_size = 128
	epochs = 10

	# read in arguments and assign values if necessary
	try:
		opts,args=getopt.getopt(args,"d:n:r:e:b:",[])
	except getopt.GetoptError:
		print '\npython NeuralNetwork.py -d <dataset> -n <num_hidden_layer_nodes> -r <learning_rate> -e <epochs> -b <batch_size>', \
			  '\n\toptions for <dataset> are "-d mnist_784" or "-d ionosphere"\n'
		sys.exit(2)
	for opt,arg in opts:
		if opt=='-d':
			dataset=arg
		elif opt=='-n':
			n_h=int(arg)
		elif opt=='-r':
			learning_rate=float(arg)
		elif opt=='-e':
			epochs=int(arg)
		elif opt=='-b':
			batch_size=int(arg)

	# check that datasets are one of the two we set this up for
	if (dataset != 'mnist_784') and (dataset != 'ionosphere'):
		print '\nERROR: use -d to specify dataset\n\toptions are "-d mnist_784" or "-d ionosphere"\n'
		sys.exit(2)

	# load dataset from openml website
	print "Loading dataset"
	loaded_data = fetch_openml(dataset, version=1, cache=True)

	print "Feature Extraction"
	# get target labels for printing out
	target_labels = np.unique(loaded_data["target"])

	# convert labels to integers
	if dataset == 'ionosphere':
		loaded_data.target = np.where(loaded_data.target == 'b', 0, 1)
	elif dataset == 'mnist_784':
		loaded_data.target = loaded_data.target.astype(np.int8)

	# X is features, y is labels
	X, y = loaded_data["data"], loaded_data["target"]

	# scale
	if dataset == 'mnist_784':
		X = X / 255

	# one-hot encode labels
	digits = len(np.unique(y))
	examples = y.shape[0]
	y = y.reshape(1, examples)
	Y_new = np.eye(digits)[y.astype('int32')]
	Y_new = Y_new.T.reshape(digits, examples)

	# split, reshape, shuffle
	m = int(0.7*examples)
	m_test = X.shape[0] - m
	X_train, X_test = X[:m].T, X[m:].T
	Y_train, Y_test = Y_new[:,:m], Y_new[:,m:]
	shuffle_index = np.random.permutation(m)
	X_train, Y_train = X_train[:, shuffle_index], Y_train[:, shuffle_index]

	np.random.seed(17)

	# hyperparameters
	n_x = X_train.shape[0]
	beta = .9
	batches = -(-m // batch_size)

	# initialization
	params = { "W1": np.random.randn(n_h, n_x) * np.sqrt(1. / n_x),
			   "b1": np.zeros((n_h, 1)) * np.sqrt(1. / n_x),
			   "W2": np.random.randn(digits, n_h) * np.sqrt(1. / n_h),
			   "b2": np.zeros((digits, 1)) * np.sqrt(1. / n_h) }

	V_dW1 = np.zeros(params["W1"].shape)
	V_db1 = np.zeros(params["b1"].shape)
	V_dW2 = np.zeros(params["W2"].shape)
	V_db2 = np.zeros(params["b2"].shape)

	# train on number of epochs
	print "Beginning training"
	for i in range(epochs):

		permutation = np.random.permutation(X_train.shape[1])
		X_train_shuffled = X_train[:, permutation]
		Y_train_shuffled = Y_train[:, permutation]

		for j in range(batches):

			begin = j * batch_size
			end = min(begin + batch_size, X_train.shape[1] - 1)
			X = X_train_shuffled[:, begin:end]
			Y = Y_train_shuffled[:, begin:end]
			m_batch = end - begin

			cache = forward_propagation(X, params)
			gradients = backward_propagation(X, Y, params, cache, m_batch)

			V_dW1 = (beta * V_dW1 + (1. - beta) * gradients["dW1"])
			V_db1 = (beta * V_db1 + (1. - beta) * gradients["db1"])
			V_dW2 = (beta * V_dW2 + (1. - beta) * gradients["dW2"])
			V_db2 = (beta * V_db2 + (1. - beta) * gradients["db2"])

			params["W1"] = params["W1"] - learning_rate * V_dW1
			params["b1"] = params["b1"] - learning_rate * V_db1
			params["W2"] = params["W2"] - learning_rate * V_dW2
			params["b2"] = params["b2"] - learning_rate * V_db2

		cache = forward_propagation(X_train, params)
		train_loss = compute_loss(Y_train, cache["A2"])
		cache = forward_propagation(X_test, params)
		test_loss = compute_loss(Y_test, cache["A2"])
		print "Epoch {}: training loss = {}, test loss = {}".format(i+1 ,train_loss, test_loss)

	print "Training done."

	# Make predictions on testing set
	cache = forward_propagation(X_test, params)
	predictions = np.argmax(cache["A2"], axis=0)
	labels = np.argmax(Y_test, axis=0)

	print "\nTest Data Statistics:\n\n", classification_report(predictions, labels, target_names=target_labels)


if __name__ == "__main__":
   main(sys.argv[1:])
