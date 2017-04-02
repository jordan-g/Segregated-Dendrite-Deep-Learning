"""
Example script creating a network & training it on MNIST.

Copyright (C) 2017 Jordan Guerguiev

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

import deep_learning as dl

# load MNIST -- assuming the original MNIST binary files are in the current folder
try:
	x_train, t_train, x_test, t_test = dl.load_MNIST()
except:
	exit()

# silence 80% of feedback weights
dl.use_sparse_feedback = True

# set training parameters
f_etas = (0.21, 0.21)
b_etas = None
n_epochs = 10
n_training_examples = 60000

# create the network
net = dl.Network(n=(500, 10))

# train the network
net.train(f_etas, b_etas, n_epochs, n_training_examples, save_simulation=True, simulations_folder="Simulations", folder_name="Example Simulation")

# re-load the saved simulation & network
net, f_etas, b_etas, n_training_examples = dl.load_simulation(latest_epoch=9, folder_name="Example Simulation", simulations_folder="Simulations")

# train the network for another 10 epochs
net.train(f_etas, b_etas, n_epochs, n_training_examples, save_simulation=True, simulations_folder="Simulations", folder_name="Example Simulation")