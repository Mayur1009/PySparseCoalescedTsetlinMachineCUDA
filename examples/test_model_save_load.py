import pickle
from PySparseCoalescedTsetlinMachineCUDA.tm import MultiClassConvolutionalTsetlinMachine2D
import numpy as np
from time import time
from bz2 import BZ2File

from keras.api.datasets import mnist


if __name__ == "__main__":
	(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
	X_train = np.where(X_train.reshape((X_train.shape[0], 28 * 28)) > 75, 1, 0)
	X_test = np.where(X_test.reshape((X_test.shape[0], 28 * 28)) > 75, 1, 0)

	tm = MultiClassConvolutionalTsetlinMachine2D(
		number_of_clauses=2500,
		T=3125,
		s=10.0,
		dim=(28, 28, 1),
		patch_dim=(10, 10),
	)

	epochs = 1
	print(f"\nAccuracy over {epochs} epochs:\n")
	for i in range(epochs):
		start_training = time()
		tm.fit(X_train, Y_train, epochs=1, incremental=True)
		stop_training = time()

		start_testing = time()
		result = 100 * (tm.predict(X_test) == Y_test).mean()
		stop_testing = time()

		print(
			f"#{i + 1} | Accuracy: {result:.4f}% | Training Time: {stop_training - start_training:.4f}s, Testing Time: {stop_testing - start_testing:.2f}s"
		)

	# Save the model
	state_dict = tm.save()
	with BZ2File("model.tm", "wb") as file:
		pickle.dump(state_dict, file)

	# Load the model
	with BZ2File("model.tm", "rb") as file:
		loaded_dict = pickle.load(file)

	new_tm = MultiClassConvolutionalTsetlinMachine2D(
		number_of_clauses=2500,
		T=3125,
		s=10.0,
		dim=(28, 28, 1),
		patch_dim=(10, 10),
	)
	new_tm.load(loaded_dict)

	# Test the loaded model
	for i in range(epochs):
		start_training = time()
		new_tm.fit(X_train, Y_train, epochs=1, incremental=True)
		stop_training = time()

		start_testing = time()
		result = 100 * (new_tm.predict(X_test) == Y_test).mean()
		stop_testing = time()

		print(
			f"#{i + 1} | Accuracy: {result:.4f}% | Training Time: {stop_training - start_training:.4f}s, Testing Time: {stop_testing - start_testing:.2f}s"
		)

