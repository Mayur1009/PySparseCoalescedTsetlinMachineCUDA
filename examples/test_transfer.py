from time import time

import numpy as np
from keras.api.datasets import mnist
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import matplotlib.pyplot as plt
from PySparseCoalescedTsetlinMachineCUDA.tm import MultiClassConvolutionalTsetlinMachine2D, MultiClassTsetlinMachine

if __name__ == "__main__":
	(X_train, Y_train_org), (X_test, Y_test_org) = mnist.load_data()

	X_train = np.where(X_train.reshape((X_train.shape[0], 28 * 28)) > 75, 1, 0)
	X_test = np.where(X_test.reshape((X_test.shape[0], 28 * 28)) > 75, 1, 0)
	Y_train, Y_test = Y_train_org, Y_test_org

	train_target_ind = np.logical_or(Y_train == 1, Y_train == 7)
	test_target_ind = np.logical_or(Y_test == 1, Y_test == 7)

	train_source_ind = np.logical_not(train_target_ind)
	test_source_ind = np.logical_not(test_target_ind)

	X_train_target = X_train[train_target_ind][:50]
	Y_train_target = Y_train[train_target_ind][:50]
	X_train_source = X_train[train_source_ind]
	Y_train_source = Y_train[train_source_ind]

	X_test_target = X_test[test_target_ind]
	Y_test_target = Y_test[test_target_ind]
	X_test_source = X_test[test_source_ind]
	Y_test_source = Y_test[test_source_ind]

	z = np.unique(Y_train_target)
	for i, v in enumerate(z):
		Y_train_target[Y_train_target == v] = i
		Y_test_target[Y_test_target == v] = i

	z = np.unique(Y_train_source)
	for i, v in enumerate(z):
		Y_train_source[Y_train_source == v] = i
		Y_test_source[Y_test_source == v] = i

	print("Training on source")

	tm_source = MultiClassTsetlinMachine(
		1000,
		10000,
		8,
		grid=(16 * 13, 1, 1),
		block=(128, 1, 1),
	)

	train_acc_source = []
	test_acc_source = []
	for i in range(2):
		start_training = time()
		tm_source.fit(X_train_source, Y_train_source, epochs=1, incremental=True)
		stop_training = time()

		start_testing = time()
		preds = tm_source.predict(X_test_source)
		stop_testing = time()
		result_test = accuracy_score(Y_test_source, preds)

		preds = tm_source.predict(X_train_source)
		result_train = accuracy_score(Y_train_source, preds)

		train_acc_source.append(result_train)
		test_acc_source.append(result_test)

		print(
			"%d %.2f %.2f %.2f %.2f"
			% (i, result_train, result_test, stop_training - start_training, stop_testing - start_testing)
		)

	plt.figure()
	plt.title("Source")
	plt.plot(train_acc_source, label="train")
	plt.plot(test_acc_source, label="test")
	plt.ylim(0.5, 1.05)
	plt.grid()

	tm_target = MultiClassTsetlinMachine(
		1000,
		10000,
		8,
		grid=(16 * 13, 1, 1),
		block=(128, 1, 1),
	)

	train_acc_target = []
	test_acc_target = []

	for i in range(40):
		start_training = time()
		tm_target.fit(X_train_target, Y_train_target, epochs=1, incremental=True)
		stop_training = time()

		start_testing = time()
		preds = tm_target.predict(X_test_target)
		stop_testing = time()
		result_test = accuracy_score(Y_test_target, preds)

		preds = tm_target.predict(X_train_target)
		result_train = accuracy_score(Y_train_target, preds)

		train_acc_target.append(result_train)
		test_acc_target.append(result_test)

		print(
			"%d %.2f %.2f %.2f %.2f"
			% (i, result_train, result_test, stop_training - start_training, stop_testing - start_testing)
		)

	plt.figure()
	plt.title("Target")
	plt.plot(train_acc_target, label="train")
	plt.plot(test_acc_target, label="test")
	plt.ylim(0.5, 1.05)
	plt.grid()

	source_state = tm_source.save()

	tm_transfer = MultiClassTsetlinMachine(
		1010,
		10000,
		8,
		grid=(16 * 13, 1, 1),
		block=(128, 1, 1),
	)

	tm_transfer.transfer(source_state, 2)

	ff = np.zeros((1, 1010), dtype=np.uint32)
	for i in range(10):
		ff[0, 1000 + i] = 1

	tm_transfer.freeze_clauses(ff)

	states = tm_transfer.get_ta_states()
	print(f'{states.shape=}')
	breakpoint()

	train_acc_transfer = []
	test_acc_transfer = []
	for i in range(20):
		start_training = time()
		tm_transfer.fit(X_train_target, Y_train_target, epochs=1, incremental=True)
		stop_training = time()

		start_testing = time()
		preds = tm_transfer.predict(X_test_target)
		stop_testing = time()
		result_test = accuracy_score(Y_test_target, preds)

		preds = tm_transfer.predict(X_train_target)
		result_train = accuracy_score(Y_train_target, preds)

		train_acc_transfer.append(result_train)
		test_acc_transfer.append(result_test)

		print(
			"%d %.2f %.2f %.2f %.2f"
			% (i, result_train, result_test, stop_training - start_training, stop_testing - start_testing)
		)

	tm_transfer.unfreeze_clauses()

	for i in range(20):
		start_training = time()
		tm_transfer.fit(X_train_target, Y_train_target, epochs=1, incremental=True)
		stop_training = time()

		start_testing = time()
		preds = tm_transfer.predict(X_test_target)
		stop_testing = time()
		result_test = accuracy_score(Y_test_target, preds)

		preds = tm_transfer.predict(X_train_target)
		result_train = accuracy_score(Y_train_target, preds)

		train_acc_transfer.append(result_train)
		test_acc_transfer.append(result_test)

		print(
			"%d %.2f %.2f %.2f %.2f"
			% (i, result_train, result_test, stop_training - start_training, stop_testing - start_testing)
		)

	plt.figure()
	plt.title("Transfer")
	plt.plot(train_acc_transfer, label="train")
	plt.plot(test_acc_transfer, label="test")
	# plt.ylim(0.5, 1.05)
	plt.grid()

	plt.show()
