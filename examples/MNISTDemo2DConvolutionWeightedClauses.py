from PySparseCoalescedTsetlinMachineCUDA.tm import MultiClassConvolutionalTsetlinMachine2D

import numpy as np
from time import time

from keras.api.datasets import mnist

if __name__ == "__main__":
	(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
	X_train = np.where(X_train.reshape((X_train.shape[0], 28 * 28)) > 75, 1, 0)
	X_test = np.where(X_test.reshape((X_test.shape[0], 28 * 28)) > 75, 1, 0)

	tm = MultiClassConvolutionalTsetlinMachine2D(
		number_of_clauses=2500,
		T=3125,
		s=10,
		dim=(28, 28, 1),
		patch_dim=(10, 10),
	)

	for i in range(5):
		start_training = time()
		tm.fit(X_train, Y_train, epochs=1, incremental=True)
		stop_training = time()

		start_testing = time()
		result_test = 100 * (tm.predict(X_test) == Y_test).mean()
		stop_testing = time()

		result_train = 100 * (tm.predict(X_train) == Y_train).mean()

		print(
			f"Epoch {i + 1} | Train Time: {stop_training - start_training:.2f}s, Test Time: {stop_testing - start_testing:.2f}s | Train Accuracy: {result_train:.4f}, Test Accuracy: {result_test:.4f}"
		)
