"""
Example script creating a network & training it on MNIST.
"""

import deep_learning as dl

# load MNIST -- assuming the original MNIST binary files are in the current folder
try:
    x_train, x_test, t_train, t_test = dl.load_MNIST()
except:
    exit()

# -- No spiking feedback, 100 output units -- #

dl.use_rand_phase_lengths  = True  # use random phase lengths (chosen from Wald distribution)
dl.use_conductances        = True  # use conductances between dendrites and soma
dl.use_broadcast           = True  # use broadcast (ie. feedback to all layers comes from output layer)
dl.use_spiking_feedback    = False # use spiking feedback

dl.use_symmetric_weights   = False # enforce symmetric weights
dl.noisy_symmetric_weights = False # add noise to symmetric weights

dl.use_sparse_feedback     = False # use sparse feedback weights
dl.update_backward_weights = False # update backward weights
dl.use_backprop            = False # use error backpropagation
dl.record_backprop_angle   = False # record angle b/w hidden layer error signals and backprop-generated error signals
dl.use_apical_conductance  = False # use attenuated conductance from apical dendrite to soma

# create the network
net = dl.Network(n=(500, 100, 100))

# set training parameters
f_etas = (0.12, 0.23, 0.23)
b_etas = None
n_epochs = 60
n_training_examples = 60000
save_experiment = True

# -- Spiking feedback, 100 output units -- #

# train the network
net.train(f_etas, b_etas, n_epochs, n_training_examples, save_experiment)

dl.use_rand_phase_lengths  = True  # use random phase lengths (chosen from Wald distribution)
dl.use_conductances        = True  # use conductances between dendrites and soma
dl.use_broadcast           = True  # use broadcast (ie. feedback to all layers comes from output layer)
dl.use_spiking_feedback    = True  # use spiking feedback

dl.use_symmetric_weights   = False # enforce symmetric weights
dl.noisy_symmetric_weights = False # add noise to symmetric weights

dl.use_sparse_feedback     = False # use sparse feedback weights
dl.update_backward_weights = False # update backward weights
dl.use_backprop            = False # use error backpropagation
dl.record_backprop_angle   = False # record angle b/w hidden layer error signals and backprop-generated error signals
dl.use_apical_conductance  = False # use attenuated conductance from apical dendrite to soma

# create the network
net = dl.Network(n=(500, 100, 100))

# set training parameters
f_etas = (0.12, 0.23, 0.23)
b_etas = None
n_epochs = 60
n_training_examples = 60000
save_experiment = True

# train the network
net.train(f_etas, b_etas, n_epochs, n_training_examples, save_experiment)