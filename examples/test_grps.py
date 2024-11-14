from time import time

import numpy as np
from keras.api.datasets import mnist
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import matplotlib.pyplot as plt
from PySparseCoalescedTsetlinMachineCUDA.tm import MultiClassConvolutionalTsetlinMachine2D

clauses_1 = int(1000)
s = 10
T_1 = int(clauses_1 * 0.8)

epochs = 1

patch_size = 10

(X_train, Y_train_org), (X_test, Y_test_org) = mnist.load_data()

X_train = np.where(X_train.reshape((X_train.shape[0], 28 * 28)) > 75, 1, 0)
X_test = np.where(X_test.reshape((X_test.shape[0], 28 * 28)) > 75, 1, 0)

classes = 10

Y_train, Y_test = Y_train_org, Y_test_org
# Y_train = np.zeros((Y_train_org.shape[0], classes), dtype=np.uint32)
# for i in range(Y_train_org.shape[0]):
#     Y_train[i, Y_train_org[i]] = 1
#
# Y_test = np.zeros((Y_test_org.shape[0], classes), dtype=np.uint32)
# for i in range(Y_test_org.shape[0]):
#     Y_test[i, Y_test_org[i]] = 1


tm = MultiClassConvolutionalTsetlinMachine2D(
    clauses_1,
    T_1,
    s,
    (28, 28, 1),
    (patch_size, patch_size),
    group_ids=[],
    weight_update_factor=[5, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    state_inc_factor=[5, 1, 1, 1, 1, 1, 1, 1, 1, 1],
)

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


literals = tm.get_literals()
M, N = tm.dim[0] - tm.patch_dim[0] + 1, tm.dim[1] - tm.patch_dim[1] + 1
num_loc_lits = M - 1 + N - 1
half_lits = tm.number_of_features // 2

print(f"{literals.shape=}")

grp_ids = tm.group_ids

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


# X = X_train[:2]
# Y = Y_train_org[:2]
# literals = tm.get_literals()
# print(f"{literals.shape=}")
# weights = tm.get_weights()
# print(f"{weights.shape=}")
# patch_outputs = tm.transform_patchwise(X)
# grp_ids = tm.group_ids
#
# M, N = tm.dim[0] - tm.patch_dim[0] + 1, tm.dim[1] - tm.patch_dim[1] + 1
# patch_outputs = np.array(patch_outputs.todense()).reshape((2, tm.number_of_groups, tm.number_of_clauses, M, N))
#
# num_loc_lits = M - 1 + N - 1
# half_lits = tm.number_of_features // 2
#
# print(f"{literals.shape=}")
# print(f"{patch_outputs.shape=}")
#
# for e in range(2):
#     print(f"{Y[e]=}")
#     c = Y[e]
#     tot_active = 0
#     po = 0
#
#     pimg = np.zeros((28, 28))
#     nimg = np.zeros((28, 28))
#     eimg = np.zeros((28, 28))
#     pwimg = np.zeros((28, 28))
#     nwimg = np.zeros((28, 28))
#     ewimg = np.zeros((28, 28))
#     for ci in range(tm.number_of_clauses):
#         pos_lits = literals[grp_ids[c], ci, num_loc_lits:half_lits].reshape((tm.patch_dim[0], tm.patch_dim[1]))
#         neg_lits = literals[grp_ids[c], ci, half_lits + num_loc_lits :].reshape((tm.patch_dim[0], tm.patch_dim[1]))
#
#         tpos = np.zeros((28, 28))
#         tneg = np.zeros((28, 28))
#
#         w = weights[Y[e], ci]
#         if w >= 0:
#             tot_active += 1
#
#             for m in range(M):
#                 for n in range(N):
#                     if patch_outputs[e, grp_ids[c], ci, m, n] == 1:
#                         po += 1
#                         tpos[m : m + tm.patch_dim[0], n : n + tm.patch_dim[1]] += pos_lits
#                         tneg[m : m + tm.patch_dim[0], n : n + tm.patch_dim[1]] += neg_lits
#
#             # tpos = (tpos > 0) & 1
#             # tneg = (tneg > 0) & 1
#
#             pimg += tpos
#             nimg += tneg
#             eimg += tpos - tneg
#             pwimg += tpos * w
#             nwimg += tneg * w
#             ewimg += (tpos - tneg) * w
#
#     print(f"{tot_active=}")
#     print(f"{po=}")
#     tot_active = tot_active if tot_active > 0 else 1
#     # pimg /= tot_active
#     # nimg /= tot_active
#     # eimg /= tot_active
#     # pwimg /= tot_active
#     # nwimg /= tot_active
#     # ewimg /= tot_active
#
#     fig, axs = plt.subplots(2, 4, squeeze=False, layout="compressed")
#
#     axs[0, 0].imshow(X[e].reshape(28, 28))
#     axs[0, 1].imshow(pimg)
#     axs[0, 2].imshow(nimg)
#     axs[0, 3].imshow(eimg)
#     axs[1, 1].imshow(pwimg)
#     axs[1, 2].imshow(nwimg)
#     axs[1, 3].imshow(ewimg)
# plt.show()
