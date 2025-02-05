from PySparseCoalescedTsetlinMachineCUDA.tm import MultiClassTsetlinMachine
import numpy as np

np.random.seed(42)


def create_data(num_samples_per_pattern, noise):
    X = np.zeros((num_samples_per_pattern * 4, 2), dtype=np.uint32)
    Y = np.zeros(num_samples_per_pattern * 4)

    X[1 * num_samples_per_pattern : 2 * num_samples_per_pattern, 1] = 1
    X[2 * num_samples_per_pattern : 3 * num_samples_per_pattern, 0] = 1
    X[3 * num_samples_per_pattern : 4 * num_samples_per_pattern, :] = 1
    Y[1 * num_samples_per_pattern : 3 * num_samples_per_pattern] = 1.0

    add_noise = np.random.random(num_samples_per_pattern * 4) <= noise
    Y[np.argwhere(add_noise).ravel()] -= 1
    Y[np.argwhere(add_noise).ravel()] *= -1
    Y = Y.astype(np.uint32)

    # shuffle
    indices = np.random.permutation(num_samples_per_pattern * 4)

    return X[indices], Y[indices]


if __name__ == "__main__":
    for noise in [0.2]:
        X_train, Y_train = create_data(4000, noise)
        X_test, Y_test = create_data(2000, noise)

        tm = MultiClassTsetlinMachine(number_of_clauses=4, T=100, s=1.0)
        for epoch in range(10):
            tm.fit(X_train, Y_train, epochs=1, incremental=True)

            pred = tm.predict(X_test)
            acc = np.mean(pred == Y_test)
            print(f"{acc=}")

        # states = np.array(tm.history["ta_state"])
        # states = states.squeeze()
        # np.save(f"states-before-{noise}.npy", states)
