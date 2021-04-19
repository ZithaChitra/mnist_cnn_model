from keras.datasets import fashion_mnist
from keras.utils import np_utils
import numpy as np

class Mnist():
	"""
	Wrapper class for the mnist dataset.
	"""
	def __init__(self, img_width: int=28, img_height: int=28):
		self.img_width = img_width
		self.img_height = img_height
		(self.X_train, self.y_train), (self.X_test, self.y_test) = self.preprocess()



	def load_or_generate(self):
		"""
		Load data from local file system if it's 
		already there, otherwise download it.
		This implementstion always downloads 
		the data and doesn't do checks.
		"""
		return fashion_mnist.load_data()


	def split_data(self, data: np.ndarray, proportion: float = 0.2):
		"""
		Split data into two parts using the given proportion
		"""
		pass


	def preprocess(self):
		""" Preprocess data """
		(X_train, y_train), (X_test, y_test) = self.load_or_generate()
		X_train = X_train.astype(np.float32) / 255
		X_train = X_train.reshape(X_train.shape[0], self.img_width, self.img_height, 1)
		X_test = X_test.astype(np.float32) / 255
		X_test = X_test.reshape(X_test.shape[0], self.img_width, self.img_height, 1)

		# one hot encode outputs
		y_train = np_utils.to_categorical(y_train)
		y_test = np_utils.to_categorical(y_test)
		return (X_train, y_train), (X_test, y_test)
