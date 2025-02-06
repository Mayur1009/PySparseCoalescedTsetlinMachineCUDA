import numpy as np
from time import time
from keras.datasets import mnist
from PySparseCoalescedTsetlinMachineCUDA.tm import MultiClassConvolutionalTsetlinMachine2D
from scipy.sparse import lil_matrix
from skimage.util import view_as_windows
from sklearn.feature_extraction.text import CountVectorizer
from skimage.util import view_as_windows
import argparse
from scipy.sparse import csr_matrix, csc_matrix, lil_matrix

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

X_train = np.where(X_train > 75, 1, 0).astype(np.uint32)
X_test = np.where(X_test > 75, 1, 0).astype(np.uint32)
Y_train = Y_train.astype(np.uint32)
Y_test = Y_test.astype(np.uint32)

def default_args(**kwargs):
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", default=250, type=int)
    parser.add_argument("--number-of-clauses", default=1000, type=int)
    parser.add_argument("--T", default=1000, type=int)
    parser.add_argument("--s", default=1.0, type=float)
    parser.add_argument("--max-included-literals", default=32, type=int)

    args = parser.parse_args()
    for key, value in kwargs.items():
        if key in args.__dict__:
            setattr(args, key, value)
    return args

args = default_args()

f = open("mnist_%.1f_%d_%d_%d_%d.txt" % (args.s, args.number_of_clauses, args.T, args.hypervector_bits, args.hypervector_size), "w+")

patch_size = 3
dim = 28 - patch_size + 1

number_of_nodes = dim * dim

# Produces hypervector codes

symbols = patch_size*patch_size
hypervector_size = symbols

indexes = np.arange(hypervector_size, dtype=np.uint32)
encoding = np.zeros(symbols, dtype=np.uint32)
for i in range(symbols):
    encoding[i] = i

X_train_tokenized = np.zeros((X_train.shape[0], 26, 26, hypervector_size), dtype=np.uint32)
for i in range(X_train.shape[0]):
    if i % 1000 == 0:
        print(i, X_train.shape[0])

    windows = view_as_windows(X_train[i,:,:], (patch_size, patch_size))
    for q in range(windows.shape[0]):
        for r in range(windows.shape[1]):
            patch = windows[q,r].reshape(-1).astype(np.uint32)
            X_train_tokenized[i, q, r, :] = 0
            for k in patch.nonzero()[0]:
                X_train_tokenized[i, q, r,:][encoding[k]] = 1

print("Training data produced")

X_test_tokenized = np.zeros((X_test.shape[0], 26, 26, hypervector_size), dtype=np.uint32)
for i in range(X_test.shape[0]):
    if i % 1000 == 0:
        print(i, X_test.shape[0])

    windows = view_as_windows(X_test[i,:,:], (patch_size, patch_size))
    for q in range(windows.shape[0]):
        for r in range(windows.shape[1]):
            patch = windows[q,r].reshape(-1).astype(np.uint32)
            X_test_tokenized[i, q, r, :] = 0
            for k in patch.nonzero()[0]:
                X_test_tokenized[i, q, r,:][encoding[k]] = 1

print("Testing data produced")

# Starts training on the visual tokens encoded as hypervectors

X_train = csr_matrix(X_train_tokenized.reshape(X_train.shape[0], -1))
X_test = csr_matrix(X_test_tokenized.reshape(X_test.shape[0], -1))

tm = MultiClassConvolutionalTsetlinMachine2D(args.number_of_clauses, args.T, args.s, (dim, dim, args.hypervector_size), (1, 1), max_included_literals=args.max_included_literals)

for epoch in range(args.epochs):
    start_training = time()
    tm.fit(X_train, Y_train, incremental=True, epochs=1)
    stop_training = time()

    start_testing = time()
    result_test = 100*(tm.predict(X_test) == Y_test).mean()
    stop_testing = time()

    result_train = 100*(tm.predict(X_train) == Y_train).mean()

    print("%d %.2f %.2f %.2f %.2f" % (epoch, result_test, result_train, stop_training-start_training, stop_testing-start_testing))
    print("%d %.2f %.2f %.2f %.2f" % (epoch, result_test, result_train, stop_training-start_training, stop_testing-start_testing), file=f)
    f.flush()
f.close()