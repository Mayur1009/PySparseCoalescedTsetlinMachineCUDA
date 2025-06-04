# %%
from PySparseCoalescedTsetlinMachineCUDA.tm import MultiClassConvolutionalTsetlinMachine2D
import numpy as np
from time import time
from keras.datasets import mnist
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns

# %%
# Load MNIST dataset
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
X_train = np.where(X_train.reshape((X_train.shape[0], 28 * 28)) > 75, 1, 0)
X_test = np.where(X_test.reshape((X_test.shape[0], 28 * 28)) > 75, 1, 0)


# %%
# The Model
tm_s1 = MultiClassConvolutionalTsetlinMachine2D(
	number_of_clauses=2500,
	T=3125,
	s=1,
	dim=(28, 28, 1),
	patch_dim=(10, 10),
	grid=(256, 1, 1),
	block=(128, 1, 1),
)

# %%
# Train and test the model
epochs = 5
for i in range(epochs):
	start_training = time()
	tm_s1.fit(X_train, Y_train, epochs=1, incremental=True)
	stop_training = time()

	start_testing = time()
	result = 100 * (tm_s1.predict(X_test) == Y_test).mean()
	stop_testing = time()

	print(
		f"""Epoch: {i + 1} | Accuracy: {result:.2f}% | Training Time: {stop_training - start_training:.2f}s | Testing Time: {stop_testing - start_testing:.2f}s""",
	)

# %%
# Get clauses
clauses_s1 = tm_s1.get_literals()

# %%
sim_s1 = cosine_similarity(clauses_s1, clauses_s1)


# %%
# The Model
tm_s2 = MultiClassConvolutionalTsetlinMachine2D(
	number_of_clauses=2500,
	T=3125,
	s=1,
	dim=(28, 28, 1),
	patch_dim=(10, 10),
	grid=(256, 1, 1),
	block=(128, 1, 1),
)

# %%
# Train and test the model
epochs = 5
for i in range(epochs):
	start_training = time()
	tm_s2.fit(X_train, Y_train, epochs=1, incremental=True)
	stop_training = time()

	start_testing = time()
	result = 100 * (tm_s2.predict(X_test) == Y_test).mean()
	stop_testing = time()

	print(
		f"""Epoch: {i + 1} | Accuracy: {result:.2f}% | Training Time: {stop_training - start_training:.2f}s | Testing Time: {stop_testing - start_testing:.2f}s""",
	)

# %%
# Get clauses
clauses_s2 = tm_s2.get_literals()

# %%
sim_s2 = cosine_similarity(clauses_s2, clauses_s2)


# %%
fig, ax = plt.subplots(1, 2, figsize=(10, 10), layout="compressed")
sns.heatmap(sim_s1, ax=ax[0], vmin=0, vmax=1, annot=False, cbar=False)
ax[0].set_title("Similarity Matrix for Model s=1")
sns.heatmap(sim_s2, ax=ax[1], cbar=True, vmin=0, vmax=1, annot=False)
ax[1].set_title("Similarity Matrix for Model s=10")

plt.show()

# %%


