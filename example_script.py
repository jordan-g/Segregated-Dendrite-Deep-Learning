"""
Example script creating a network & training it on MNIST.
"""

import deep_learning as dl

# load MNIST -- assuming the original MNIST binary files are in the current folder
try:
	x_train, x_test, t_train, t_test = dl.load_MNIST()
except:
	exit()

# silence 80% of feedback weights
dl.use_sparse_feedback = True

# create the network
net = dl.Network(n=(500, 10))

# set training parameters
f_etas = (0.21, 0.21)
b_etas = None
n_epochs = 60
n_training_examples = 60000
save_experiment = True

# train the network
net.train(f_etas, b_etas, n_epochs, n_training_examples, save_experiment)