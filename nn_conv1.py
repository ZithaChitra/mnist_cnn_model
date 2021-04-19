from typing import Dict
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Flatten


def nn_conv1(
	net_args: Dict,
	io_shapes: Dict
):
	"""
	Build a simple keras convnet model.
	Parameters:
	----------
	net_config: Dict
		A dictionary with keys 'shapes' and 'hyperparams'.
		'shapes' is a dictionary containing the input and
		output shape.
		'hyperparameters' is a dictionary containing keys
		representing other hyperparameters of the model such
		as activation_fn, dropout_amount, etc. 
	"""
	# input_shape = io_shapes.get["input_shape"]
	# output_shape = io_shapes.get["output_shape"]
	config = net_args

	model = Sequential()
	model.add(Conv2D(config["layer_1_size"], (5, 5), activation='relu', input_shape=(28, 28, 1)))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Conv2D(config["layer_2_size"], (5, 5), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(config["dropout"]))
	model.add(Flatten())
	model.add(Dense(config["hidden_layer_size"], activation='relu'))
	model.add(Dense(10, activation='softmax'))

	return model





