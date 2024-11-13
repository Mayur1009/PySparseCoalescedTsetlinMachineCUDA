from time import time

import matplotlib.pyplot as plt
import numpy as np
from keras.api.datasets import mnist
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from PySparseCoalescedTsetlinMachineCUDA.tm import MultiOutputConvolutionalTsetlinMachine2D

clauses_1 = int(2500)
s = 10.0
T_1 = int(3125)

epochs = 1

patch_size = 10

(X_train, Y_train_org), (X_test, Y_test_org) = mnist.load_data()

X_train = np.where(X_train.reshape((X_train.shape[0], 28 * 28)) > 75, 1, 0)
X_test = np.where(X_test.reshape((X_test.shape[0], 28 * 28)) > 75, 1, 0)

classes = 10

Y_train = np.zeros((Y_train_org.shape[0], classes), dtype=np.uint32)
for i in range(Y_train_org.shape[0]):
    Y_train[i, Y_train_org[i]] = 1

Y_test = np.zeros((Y_test_org.shape[0], classes), dtype=np.uint32)
for i in range(Y_test_org.shape[0]):
    Y_test[i, Y_test_org[i]] = 1

tm = MultiOutputConvolutionalTsetlinMachine2D(clauses_1, T_1, s, (28, 28, 1), (patch_size, patch_size), q=5)

for i in range(epochs):
    start_training = time()
    tm.fit(X_train, Y_train, epochs=1, incremental=True)
    stop_training = time()

    start_testing = time()
    preds = tm.predict(X_test)
    result_test = accuracy_score(Y_test, preds)
    stop_testing = time()

    print("%d %.2f %.2f %.2f" % (i, result_test, stop_training - start_training, stop_testing - start_testing))


X = X_train[:2]
Y = Y_train_org[:2]
literals = tm.get_literals()

M, N = tm.dim[0] - tm.patch_dim[0] + 1, tm.dim[1] - tm.patch_dim[1] + 1

num_loc_lits = M - 1 + N - 1
half_lits = tm.number_of_features // 2
grp_ids = tm.group_ids

print(f"{literals.shape=}")

for c in range(10):
    weights = tm.get_weights()[c]
    patch_weights = tm.get_patch_weights()[c].reshape((tm.number_of_clauses, M, N))

    avg_clause_img = np.zeros((2, 3, 28, 28))

    for ci in tqdm(range(tm.number_of_clauses)):
        pos_lits = (
            literals[grp_ids[c], ci, num_loc_lits:half_lits].reshape((tm.patch_dim[0], tm.patch_dim[1])).astype(np.int8)
        )
        neg_lits = (
            literals[grp_ids[c], ci, half_lits + num_loc_lits :]
            .reshape((tm.patch_dim[0], tm.patch_dim[1]))
            .astype(np.int8)
        )

        tpos = np.zeros((28, 28))
        tneg = np.zeros((28, 28))
        teff = np.zeros((28, 28))
        pws = np.abs(patch_weights[ci].reshape(M * N))
        th = np.quantile(pws, 0.95)

        for m in range(M):
            for n in range(N):
                if np.abs(patch_weights[ci, m, n]) > th:
                    tpos[m : m + tm.patch_dim[0], n : n + tm.patch_dim[1]] += pos_lits * patch_weights[ci, m, n]
                    tneg[m : m + tm.patch_dim[0], n : n + tm.patch_dim[1]] += neg_lits * patch_weights[ci, m, n]
                    teff[m : m + tm.patch_dim[0], n : n + tm.patch_dim[1]] += (pos_lits - neg_lits).astype(
                        int
                    ) * patch_weights[ci, m, n]

        polarity = (weights[ci] > 0) & 1

        avg_clause_img[polarity, 0] = avg_clause_img[polarity, 0] + (
            tpos * abs(weights[ci]) - avg_clause_img[polarity, 0]
        ) / (tm.number_of_clauses + 1)
        avg_clause_img[polarity, 1] = avg_clause_img[polarity, 1] + (
            tneg * abs(weights[ci]) - avg_clause_img[polarity, 1]
        ) / (tm.number_of_clauses + 1)
        avg_clause_img[polarity, 2] = avg_clause_img[polarity, 2] + (
            teff * abs(weights[ci]) - avg_clause_img[polarity, 2]
        ) / (tm.number_of_clauses + 1)

    fig, axs = plt.subplots(2, 3, squeeze=False, layout="compressed")
    for p in range(2):
        for t in range(3):
            axs[p - 1, t].imshow(avg_clause_img[p, t])

plt.show()
