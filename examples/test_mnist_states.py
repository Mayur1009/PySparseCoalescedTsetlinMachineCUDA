from PySparseCoalescedTsetlinMachineCUDA.tm import MultiClassTsetlinMachine
import numpy as np
from time import time
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.animation import FuncAnimation
from keras.api.datasets import mnist

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

X_train = np.where(X_train.reshape((X_train.shape[0], 28 * 28)) > 75, 1, 0)
X_test = np.where(X_test.reshape((X_test.shape[0], 28 * 28)) > 75, 1, 0)


tm = MultiClassTsetlinMachine(2500, 3125, 10.0, sr=1000.0)

batch_size = 60000
all_states = []
for i in range(10):
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

# all_states = np.array(all_states).squeeze()
# print(f"{all_states.shape=}")
#
#
# fig, axs = plt.subplots(1, 2, squeeze=False, layout="compressed", figsize=[10, 10])
# s = 0
#
# def frame_gen(frame):
#     sns.heatmap(frame[:, : 28 * 28], vmin=0, vmax=255, ax=axs[0, 0], cbar=False)
#     sns.heatmap(frame[:, 28 * 28 : 28 * 28 * 2], vmin=0, vmax=255, ax=axs[0, 1], cbar=False)
#
#
# anim = FuncAnimation(fig, frame_gen, frames=[all_states[x] for x in range(all_states.shape[0])], interval=1000 // 12)
# plt.show()
