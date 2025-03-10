#include <curand_kernel.h>

extern "C" {
__global__ void encode(unsigned int *X_indptr, unsigned int *X_indices, unsigned int *encoded_X, int e, int dim_x,
                       int dim_y, int dim_z, int patch_dim_x, int patch_dim_y, int append_negated, int class_features) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    int number_of_features =
        class_features + patch_dim_x * patch_dim_y * dim_z + (dim_x - patch_dim_x) + (dim_y - patch_dim_y);
    int number_of_patches = (dim_x - patch_dim_x + 1) * (dim_y - patch_dim_y + 1);

    int number_of_ta_chunks;
    if (append_negated) {
        number_of_ta_chunks = (((2 * number_of_features - 1) / 32 + 1));
    } else {
        number_of_ta_chunks = (((number_of_features - 1) / 32 + 1));
    }

    unsigned int *indices = &X_indices[X_indptr[e]];
    int number_of_indices = X_indptr[e + 1] - X_indptr[e];

    for (int k = 0; k < number_of_indices; ++k) {
        int y = indices[k] / (dim_x * dim_z);
        int x = (indices[k] % (dim_x * dim_z)) / dim_z;
        int z = (indices[k] % (dim_x * dim_z)) % dim_z;

        for (int patch = index; patch < number_of_patches; patch += stride) {
            int patch_coordinate_y = patch / (dim_x - patch_dim_x + 1);
            int patch_coordinate_x = patch % (dim_x - patch_dim_x + 1);

            if ((y < patch_coordinate_y) || (y >= patch_coordinate_y + patch_dim_y) || (x < patch_coordinate_x) ||
                (x >= patch_coordinate_x + patch_dim_x)) {
                continue;
            }

            int p_y = y - patch_coordinate_y;
            int p_x = x - patch_coordinate_x;

            int patch_pos = class_features + (dim_y - patch_dim_y) + (dim_x - patch_dim_x) + p_y * patch_dim_x * dim_z +
                            p_x * dim_z + z;

            int chunk_nr = patch_pos / 32;
            int chunk_pos = patch_pos % 32;
            encoded_X[patch * number_of_ta_chunks + chunk_nr] |= (1U << chunk_pos);

            if (append_negated) {
                int chunk_nr = (patch_pos + number_of_features) / 32;
                int chunk_pos = (patch_pos + number_of_features) % 32;
                encoded_X[patch * number_of_ta_chunks + chunk_nr] &= ~(1U << chunk_pos);
            }
        }
    }
}

__global__ void restore(unsigned int *X_indptr, unsigned int *X_indices, unsigned int *encoded_X, int e, int dim_x,
                        int dim_y, int dim_z, int patch_dim_x, int patch_dim_y, int append_negated,
                        int class_features) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    int number_of_features =
        class_features + patch_dim_x * patch_dim_y * dim_z + (dim_x - patch_dim_x) + (dim_y - patch_dim_y);
    int number_of_patches = (dim_x - patch_dim_x + 1) * (dim_y - patch_dim_y + 1);

    int number_of_ta_chunks;
    if (append_negated) {
        number_of_ta_chunks = (((2 * number_of_features - 1) / 32 + 1));
    } else {
        number_of_ta_chunks = (((number_of_features - 1) / 32 + 1));
    }

    unsigned int *indices = &X_indices[X_indptr[e]];
    int number_of_indices = X_indptr[e + 1] - X_indptr[e];

    for (int k = 0; k < number_of_indices; ++k) {
        int y = indices[k] / (dim_x * dim_z);
        int x = (indices[k] % (dim_x * dim_z)) / dim_z;
        int z = (indices[k] % (dim_x * dim_z)) % dim_z;

        for (int patch = index; patch < number_of_patches; patch += stride) {
            int patch_coordinate_y = patch / (dim_x - patch_dim_x + 1);
            int patch_coordinate_x = patch % (dim_x - patch_dim_x + 1);

            if ((y < patch_coordinate_y) || (y >= patch_coordinate_y + patch_dim_y) || (x < patch_coordinate_x) ||
                (x >= patch_coordinate_x + patch_dim_x)) {
                continue;
            }

            int p_y = y - patch_coordinate_y;
            int p_x = x - patch_coordinate_x;

            int patch_pos = class_features + (dim_y - patch_dim_y) + (dim_x - patch_dim_x) + p_y * patch_dim_x * dim_z +
                            p_x * dim_z + z;

            int chunk_nr = patch_pos / 32;
            int chunk_pos = patch_pos % 32;
            encoded_X[patch * number_of_ta_chunks + chunk_nr] &= ~(1U << chunk_pos);

            if (append_negated) {
                int chunk_nr = (patch_pos + number_of_features) / 32;
                int chunk_pos = (patch_pos + number_of_features) % 32;
                encoded_X[patch * number_of_ta_chunks + chunk_nr] |= (1U << chunk_pos);
            }
        }
    }
}

__global__ void encode_packed(unsigned int *X_indptr, unsigned int *X_indices, unsigned int *encoded_X, int e,
                              int dim_x, int dim_y, int dim_z, int patch_dim_x, int patch_dim_y, int append_negated,
                              int class_features) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    int number_of_features =
        class_features + patch_dim_x * patch_dim_y * dim_z + (dim_x - patch_dim_x) + (dim_y - patch_dim_y);
    int number_of_patches = (dim_x - patch_dim_x + 1) * (dim_y - patch_dim_y + 1);
    // int number_of_patch_chunks = (number_of_patches - 1) / 32 + 1;

    int number_of_literals;
    if (append_negated) {
        number_of_literals = number_of_features * 2;
    } else {
        number_of_literals = number_of_features;
    }

    unsigned int *indices = &X_indices[X_indptr[e]];
    int number_of_indices = X_indptr[e + 1] - X_indptr[e];

    // Looping over all pixels that are 1
    for (int k = 0; k < number_of_indices; ++k) {

        // Coordinate of the pixel
        int y = indices[k] / (dim_x * dim_z);
        int x = (indices[k] % (dim_x * dim_z)) / dim_z;
        int z = (indices[k] % (dim_x * dim_z)) % dim_z;

        //Looping over each patch
        for (int patch = index; patch < number_of_patches; patch += stride) {
            // Coordinate of the patch
            int patch_coordinate_y = patch / (dim_x - patch_dim_x + 1);
            int patch_coordinate_x = patch % (dim_x - patch_dim_x + 1);

            // Ignore patch if the pixel is not inside this patch
            if ((y < patch_coordinate_y) || (y >= patch_coordinate_y + patch_dim_y) || (x < patch_coordinate_x) ||
                (x >= patch_coordinate_x + patch_dim_x)) {
                continue;
            }

            int chunk = patch / 32;
            int pos = patch % 32;

            // Coordinate of this pixel relative to this patch, meaning location inside the patch
            int p_y = y - patch_coordinate_y;
            int p_x = x - patch_coordinate_x;

            // Location of this pixel, when all features are layed out in 1d format
            int patch_pos = class_features + (dim_y - patch_dim_y) + (dim_x - patch_dim_x) + p_y * patch_dim_x * dim_z +
                            p_x * dim_z + z;

            encoded_X[chunk * number_of_literals + patch_pos] |= (1U << pos);

            if (append_negated) {
                encoded_X[chunk * number_of_literals + patch_pos + number_of_features] &= ~(1U << pos);
            }
        }
    }
}

__global__ void restore_packed(unsigned int *X_indptr, unsigned int *X_indices, unsigned int *encoded_X, int e,
                               int dim_x, int dim_y, int dim_z, int patch_dim_x, int patch_dim_y, int append_negated,
                               int class_features) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    int number_of_features =
        class_features + patch_dim_x * patch_dim_y * dim_z + (dim_x - patch_dim_x) + (dim_y - patch_dim_y);
    int number_of_patches = (dim_x - patch_dim_x + 1) * (dim_y - patch_dim_y + 1);

    int number_of_literals;
    if (append_negated) {
        number_of_literals = number_of_features * 2;
    } else {
        number_of_literals = number_of_features;
    }

    unsigned int *indices = &X_indices[X_indptr[e]];
    int number_of_indices = X_indptr[e + 1] - X_indptr[e];

    for (int k = 0; k < number_of_indices; ++k) {
        int y = indices[k] / (dim_x * dim_z);
        int x = (indices[k] % (dim_x * dim_z)) / dim_z;
        int z = (indices[k] % (dim_x * dim_z)) % dim_z;

        for (int patch = index; patch < number_of_patches; patch += stride) {
            int patch_coordinate_y = patch / (dim_x - patch_dim_x + 1);
            int patch_coordinate_x = patch % (dim_x - patch_dim_x + 1);

            if ((y < patch_coordinate_y) || (y >= patch_coordinate_y + patch_dim_y) || (x < patch_coordinate_x) ||
                (x >= patch_coordinate_x + patch_dim_x)) {
                continue;
            }

            int chunk = patch / 32;
            int pos = patch % 32;

            int p_y = y - patch_coordinate_y;
            int p_x = x - patch_coordinate_x;

            int patch_pos = class_features + (dim_y - patch_dim_y) + (dim_x - patch_dim_x) + p_y * patch_dim_x * dim_z +
                            p_x * dim_z + z;

            encoded_X[chunk * number_of_literals + patch_pos] &= ~(1U << pos);

            if (append_negated) {
                encoded_X[chunk * number_of_literals + patch_pos + number_of_features] |= (1U << pos);
            }
        }
    }
}

__global__ void produce_autoencoder_example(curandState *state, unsigned int *active_output,
                                            int number_of_active_outputs, unsigned int *indptr_row,
                                            unsigned int *indices_row, int number_of_rows, unsigned int *indptr_col,
                                            unsigned int *indices_col, int number_of_cols, unsigned int *X,
                                            unsigned int *encoded_Y, int target, int accumulation, int T,
                                            int append_negated) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    // int stride = blockDim.x * gridDim.x;

    if (index != 0) {
        return;
    }

    /* Copy state to local memory for efficiency */
    curandState localState = state[index];

    int row;

    int number_of_features = number_of_cols;
    int number_of_literals = 2 * number_of_features;

    // unsigned int number_of_literal_chunks = (number_of_literals - 1) / 32 + 1;

    // Initialize example vector X

    for (int k = 0; k < number_of_features; ++k) {
        int chunk_nr = k / 32;
        int chunk_pos = k % 32;
        X[chunk_nr] &= ~(1U << chunk_pos);
    }

    if (append_negated) {
        for (int k = number_of_features; k < number_of_literals; ++k) {
            int chunk_nr = k / 32;
            int chunk_pos = k % 32;
            X[chunk_nr] |= (1U << chunk_pos);
        }
    }

    if ((indptr_col[active_output[target] + 1] - indptr_col[active_output[target]] == 0) ||
        (indptr_col[active_output[target] + 1] - indptr_col[active_output[target]] == number_of_rows)) {
        // If no positive/negative examples, produce a random example
        for (int a = 0; a < accumulation; ++a) {
            row = curand(&localState) % number_of_rows;
            for (int k = indptr_row[row]; k < indptr_row[row + 1]; ++k) {
                int chunk_nr = indices_row[k] / 32;
                int chunk_pos = indices_row[k] % 32;
                X[chunk_nr] |= (1U << chunk_pos);

                if (append_negated) {
                    chunk_nr = (indices_row[k] + number_of_features) / 32;
                    chunk_pos = (indices_row[k] + number_of_features) % 32;
                    X[chunk_nr] &= ~(1U << chunk_pos);
                }
            }
        }

        for (int i = 0; i < number_of_active_outputs; ++i) {
            if (i == target) {
                // int chunk_nr = active_output[i] / 32;
                // int chunk_pos = active_output[i] % 32;
                // X[chunk_nr] &= ~(1U << chunk_pos);

                encoded_Y[i] = T;
            } else {
                encoded_Y[i] = -T;
            }
        }

        state[index] = localState;

        return;
    }

    for (int a = 0; a < accumulation; ++a) {
        // Pick example randomly among positive examples
        int random_index =
            indptr_col[active_output[target]] +
            (curand(&localState) % (indptr_col[active_output[target] + 1] - indptr_col[active_output[target]]));
        row = indices_col[random_index];

        for (int k = indptr_row[row]; k < indptr_row[row + 1]; ++k) {
            int chunk_nr = indices_row[k] / 32;
            int chunk_pos = indices_row[k] % 32;
            X[chunk_nr] |= (1U << chunk_pos);

            if (append_negated) {
                chunk_nr = (indices_row[k] + number_of_features) / 32;
                chunk_pos = (indices_row[k] + number_of_features) % 32;
                X[chunk_nr] &= ~(1U << chunk_pos);
            }
        }
    }

    for (int i = 0; i < number_of_active_outputs; ++i) {
        if (i == target) {
            // int chunk_nr = active_output[i] / 32;
            // int chunk_pos = active_output[i] % 32;
            // X[chunk_nr] &= ~(1U << chunk_pos);

            encoded_Y[i] = T;
        } else {
            encoded_Y[i] = -T;
        }
    }

    state[index] = localState;
}
}
