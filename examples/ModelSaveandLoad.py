import pickle
from time import time

import numpy as np
from keras.api.datasets import mnist
from sklearn.metrics import accuracy_score

from PySparseCoalescedTsetlinMachineCUDA.tm import MultiOutputConvolutionalTsetlinMachine2D

clauses_1 = int(2000)
s = 10.0
T_1 = int(clauses_1 * 0.8)

epochs = 1

patch_size = 10

(X_train, Y_train_org), (X_test, Y_test_org) = mnist.load_data()

X_train = np.where(X_train.reshape((X_train.shape[0], 28 * 28)) > 75, 1, 0)
X_test = np.where(X_test.reshape((X_test.shape[0], 28 * 28)) > 75, 1, 0)

groups = 10
# Y_train = np.empty((Y_train_org.shape[0], groups), dtype=np.uint32)
# random_grouping = []
# for group in range(groups):
#     random_grouping.append(np.random.choice(10, size=5, replace=False))
#     Y_train[:, group] = np.where(np.isin(Y_train_org, random_grouping[-1]), 1, 0)
#
# Y_test = np.empty((Y_test_org.shape[0], groups), dtype=np.uint32)
# for group in range(groups):
#     Y_test[:, group] = np.where(np.isin(Y_test_org, random_grouping[group]), 1, 0)


Y_train = np.zeros((Y_train_org.shape[0], groups), dtype=np.uint32)
for i in range(Y_train_org.shape[0]):
    Y_train[i, Y_train_org[i]] = 1

Y_test = np.zeros((Y_test_org.shape[0], groups), dtype=np.uint32)
for i in range(Y_test_org.shape[0]):
    Y_test[i, Y_test_org[i]] = 1

f = open("mnist_%.1f_%d_%d_%d.txt" % (s, clauses_1, T_1, patch_size), "w+")

tm = MultiOutputConvolutionalTsetlinMachine2D(clauses_1, T_1, s, (28, 28, 1), (patch_size, patch_size), q=5)

for i in range(epochs):
    start_training = time()
    tm.fit(X_train, Y_train, epochs=1, incremental=True)
    stop_training = time()

    start_testing = time()
    preds = tm.predict(X_test)
    result_test = accuracy_score(Y_test, preds)
    stop_testing = time()

    preds = tm.predict(X_train)
    result_train = accuracy_score(Y_train, preds)

    print(
        "%d %.2f %.2f %.2f %.2f"
        % (i, result_train, result_test, stop_training - start_training, stop_testing - start_testing)
    )
    print(
        "%d %.2f %.2f %.2f %.2f"
        % (i, result_train, result_test, stop_training - start_training, stop_testing - start_testing),
        file=f,
    )
    f.flush()
f.close()

print("Prediction before saving...")
print(f'{accuracy_score(Y_test, tm.predict(X_test))=}')

state = tm.get_state()
with open("model.tm", "wb") as f:
    pickle.dump(state, f)


tm2 = MultiOutputConvolutionalTsetlinMachine2D(clauses_1, T_1, s, (28, 28, 1), (patch_size, patch_size), q=5)

with open("model.tm", "rb") as f:
    state_loaded = pickle.load(f)

tm2.set_state(state_loaded)

print("Prediction after loading...")
print(f'{accuracy_score(Y_test, tm2.predict(X_test))=}')

