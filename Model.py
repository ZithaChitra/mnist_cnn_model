""" 
A model is a combination of the neural net and the
data used to train it.
"""
from pathlib import Path
from typing import Callable, Dict
from keras import Model as KerasModel
from util_yaml import yaml_dump, yaml_loader
from keras.optimizers import SGD

DIRNAME = Path(__file__).parents[1].resolve() / "weights"


class Model():
	def __init__(
		self,
		dataset_cls: type,
		network_fn: Callable[..., KerasModel],
		net_args: Dict,
		net_io_shapes: Dict,
		dataset_args: Dict = None,
	):
		self.name = f"{self.__class__.__name__}_{dataset_cls.__name__}_{network_fn.__name__}"

		if dataset_args == None:
			dataset_args = {}
		self.data = dataset_cls()
		exp_conf = yaml_loader("experiment.yaml")
		exp_conf.get("dataset")["dataset_args"]["img_width"] = self.data.img_width
		exp_conf.get("dataset")["dataset_args"]["img_height"] = self.data.img_height
		yaml_dump("experiment.yaml", exp_conf)

		
		self.net_args = net_args
		self.network = network_fn(net_args, net_io_shapes)

	
	@property
	def weights_filename(self)->str:
		DIRNAME.mkdir(parent=True, exist_ok=True)
		return str(DIRNAME / f"{self.name}_weights.h5")

	
	def fit(
		self,
		callbacks
	):
		config = self.net_args

		sgd = SGD(lr=config["learn_rate"], decay=config["decay"], momentum=config["momentum"],
          nesterov=True)

		# if callbacks in None:
		# 	callbacks = []

		self.network.compile(
			loss="categorical_crossentropy",
			optimizer=sgd,
			metrics=["accuracy"]
		)

		self.network.fit(
			self.data.X_train, self.data.y_train,
			validation_data=(self.data.X_test, self.data.y_test),
			epochs = config["epochs"],
			callbacks=callbacks
		)
