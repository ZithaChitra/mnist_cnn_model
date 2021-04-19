from typing import Dict
import importlib
from util_yaml import yaml_loader
import click
import wandb
from wandb.keras import WandbCallback
import mlflow
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, ColSpe


@click.command()
@click.argument("experiment-config", type=click.Path(exists=True), default="experiment.yaml")
@click.option("--hidden_layer_size", default=128)
@click.option("--layer_1_size", default=16)
@click.option("--layer_2_size", default=32)
@click.option("--decay", default=1.0e-06)
@click.option("--dropout", default=0.2)
@click.option("--epochs", default=20)
@click.option("--learn_rate", default=0.01)
@click.option("--momentum", default=0.9)
def main(
	experiment_config,
	hidden_layer_size: int,
	layer_1_size: int,
	layer_2_size: int,
	decay: float,
	dropout: float,
	epochs: int,
	learn_rate: float,
	momentum: float
):
	""" Update values in experiment configuration file  """
	exp_config = yaml_loader("experiment.yaml")
	proj_name = exp_config.get("project_name")
	net_name = exp_config.get("network")["name"]
	
	net_args = exp_config.get("network")["net_args"]
	net_args["hidden_layer_size"] = hidden_layer_size
	net_args["layer_1_size"] = layer_1_size
	net_args["layer_2_size"] = layer_2_size
	net_args["decay"] = decay
	net_args["dropout"] = dropout
	net_args["epochs"] = epochs
	net_args["learn_rate"] = learn_rate
	net_args["momentum"] = momentum

	net_io_shapes = exp_config.get("network")["io_shapes"]
	
	dataset_cls = exp_config.get("dataset")["name"]
	dataset_args = exp_config.get("dataset")["dataset_args"]

	model = exp_config.get("model")

	wandb.login()
	train(
		proj_name, 
		model, 
		dataset_cls, 
		net_name,
		net_args,
		net_io_shapes,
		dataset_args
	)




def train(
	proj_name: str,
	Model: str,
	dataset_cls: str,
	net_fn: str,
	net_args: Dict,
	net_io_shapes: Dict,
	dataset_args: Dict,

):
	""" Train Function """

	dataset_module = importlib.import_module(f"{dataset_cls}")
	dataset_cls_ = getattr(dataset_module, dataset_cls)

	network_module = importlib.import_module(f"{net_fn}")
	network_fn_ = getattr(network_module, net_fn)

	model_module = importlib.import_module(f"{Model}")
	model_cls_ = getattr(model_module, Model)


	config = {
		"model": Model,
		"dataset_cls": dataset_cls,
		"net_fn": net_fn,
		"net_args": net_args,
		"net_io_shapes": net_io_shapes,
		"dataset_args": dataset_args
	}


	with wandb.init(project=proj_name, config=config):
		""""""
		config = wandb.config
		model = model_cls_(dataset_cls_, network_fn_, net_args, net_io_shapes, dataset_args)
		labels = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
          "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
		callback = [WandbCallback(data_type="images", labels=labels)]

		model.fit(callbacks=callback)
		



if __name__ == "__main__":
	main()





















