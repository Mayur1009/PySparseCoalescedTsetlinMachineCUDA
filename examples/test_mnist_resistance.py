from matplotlib.animation import ArtistAnimation
from matplotlib.axes import Axes
from PySparseCoalescedTsetlinMachineCUDA.tm import MultiClassTsetlinMachine
import numpy as np
from time import time
import matplotlib.pyplot as plt
import seaborn as sns
from keras.api.datasets import mnist

if __name__ == "__main__":
	(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
	X_train = np.where(X_train.reshape((X_train.shape[0], 28 * 28)) > 75, 1, 0)
	X_test = np.where(X_test.reshape((X_test.shape[0], 28 * 28)) > 75, 1, 0)

	# X_train = X_train[:10000]
	# Y_train = Y_train[:10000]
	# X_test = X_test[:10000]
	# Y_test = Y_test[:10000]

	tm = MultiClassTsetlinMachine(2500, 3125, 1.0, sr=15.0, r=0.5)

	batch_size = 100
	epochs = 10
	all_states = []

	for i in range(epochs):
		for batch in range(0, X_train.shape[0], batch_size):
			start_training = time()
			tm.fit(X_train[batch : batch + batch_size], Y_train[batch : batch + batch_size], epochs=1, incremental=True)
			stop_training = time()
			states = tm.get_ta_states()
			all_states.append(states)
		start_testing = time()
		result = 100 * (tm.predict(X_test) == Y_test).mean()
		stop_testing = time()
		print("#%d Accuracy: %.2f%%" % (i + 1, result))

	# shape: (Number of frames, num_clauses, 28 * 28 * 2)
	all_states = np.array(all_states).squeeze()
	print(f"{all_states.shape=}")

	cmap = sns.color_palette("Spectral", as_cmap=True)
	fig, axs = plt.subplots(1, 2, squeeze=False, layout="compressed", sharey=True, figsize=(8, 8))
	ax0: Axes = axs[0, 0]
	ax1: Axes = axs[0, 1]
	ax0.set_title("Positive Literals")
	ax1.set_title("Negative Literals")
	for ax in axs.ravel():
		ax.axis("off")

	ims = []
	for f in range(all_states.shape[0]):
		im0 = ax0.imshow(all_states[f, :, : 28 * 28], vmin=0, vmax=255, cmap=cmap)
		im1 = ax1.imshow(all_states[f, :, 28 * 28 :], vmin=0, vmax=255, cmap=cmap)
		if f == 0:
			ax0.imshow(all_states[f, :, : 28 * 28], vmin=0, vmax=255, cmap=cmap)
			ax1.imshow(all_states[f, :, 28 * 28 :], vmin=0, vmax=255, cmap=cmap)
			cbar = fig.colorbar(im1, ax=ax1)

		ims.append([im0, im1])

	anim = ArtistAnimation(fig, ims, blit=True, repeat_delay=1000)

	anim.save(
		"test_mnist_resistance_r0.5.mp4",
		progress_callback=lambda i, n: print(f"Saving frame {i}/{len(ims)}", end="\r", flush=True),
		dpi=120,
		fps=240,
	)
