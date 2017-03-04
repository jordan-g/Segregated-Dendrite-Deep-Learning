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

# set training parameters
f_etas = (0.3, 0.01)
b_etas = None
n_epochs = 10
n_training_examples = 60000

# create the network
net = dl.Network(n=(500, 10))

# train the network
net.train(f_etas, b_etas, n_epochs, n_training_examples, save_simulation=True, simulations_folder="Simulations", folder_name="Example Simulation")

# re-load the saved simulation & network
net, f_etas, b_etas, n_training_examples = dl.load_simulation(last_epoch=9, folder_name="Example Simulation", simulations_folder="Simulations")

# train the network for another 10 epochs
net.train(f_etas, b_etas, n_epochs, n_training_examples, save_simulation=True, simulations_folder="Simulations", folder_name="Example Simulation")