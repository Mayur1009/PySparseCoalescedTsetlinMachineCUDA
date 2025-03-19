from PySparseCoalescedTsetlinMachineCUDA.tm import MultiClassConvolutionalTsetlinMachine2D
import numpy as np
from time import time
from medmnist.dataset import OCTMNIST
from sklearn.metrics import accuracy_score, roc_auc_score

if __name__ == "__main__":
	octmnist = OCTMNIST("train")
	X_train = octmnist.imgs.astype(np.uint32)
	Y_train = octmnist.labels.squeeze().astype(np.uint32)

	octmnist = OCTMNIST("test")
	X_test = octmnist.imgs.astype(np.uint32)
	Y_test = octmnist.labels.squeeze().astype(np.uint32)

	ch = 8

	out = np.zeros((*X_train.shape, ch))
	for j in range(ch):
		t1 = (j + 1) * 255 / (ch + 1)
		out[:, :, :, j] = (X_train >= t1) & 1
	X_train = np.array(out)
	X_train = X_train.reshape((X_train.shape[0], -1))

	out = np.zeros((*X_test.shape, ch))
	for j in range(ch):
		t1 = (j + 1) * 255 / (ch + 1)
		out[:, :, :, j] = (X_test >= t1) & 1
	X_test = np.array(out)
	X_test = X_test.reshape((X_test.shape[0], -1))

	T = 25250
	tm = MultiClassConvolutionalTsetlinMachine2D(
		number_of_clauses=31500,
		T=T,
		s=49.5,
		q=2,
		dim=(28, 28, 8),
		patch_dim=(7, 7),
		grid=(16 * 13, 1, 1),
		block=(128, 1, 1),
	)

	for i in range(30):
		start_training = time()
		tm.fit(X_train, Y_train, epochs=1, incremental=True)
		stop_training = time()

		start_testing = time()
		preds, cs = tm.predict(X_test, return_class_sums=True)
		stop_testing = time()

		acc = accuracy_score(Y_test, preds)
		prob = (np.clip(cs, -T, T) + T) / (2 * T)
		prob = prob / np.sum(prob, axis=1, keepdims=True)
		auc = roc_auc_score(Y_test, prob, average="macro", multi_class="ovr")

		print(
			f"Epoch {i + 1} | Train Time: {stop_training - start_training:.2f}s, Test Time: {stop_testing - start_testing:.2f}s | Accuracy: {acc:.4f}, AUC: {auc:.4f}"
		)
