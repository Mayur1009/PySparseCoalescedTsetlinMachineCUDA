#include <curand_kernel.h>
extern "C" {

__global__ void prepare(curandState *state, unsigned int *global_ta_state, unsigned int *batch_ta_state,
                        int *clause_weights, int *batch_clause_weights) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    curandState localState = state[index];

    for (int clause_chunk_bit = index; clause_chunk_bit < CLAUSES * LA_CHUNKS * STATE_BITS;
         clause_chunk_bit += stride) {
        int clause = clause_chunk_bit / (LA_CHUNKS * STATE_BITS);
        int chunk = (clause_chunk_bit % (LA_CHUNKS * STATE_BITS)) / STATE_BITS;
        int state_bit = (clause_chunk_bit % (LA_CHUNKS * STATE_BITS)) % STATE_BITS;
        if (state_bit == STATE_BITS - 1) {
            global_ta_state[clause_chunk_bit] = 0;
            batch_ta_state[clause_chunk_bit] = 0;
        } else {
            global_ta_state[clause_chunk_bit] = ~0;
            batch_ta_state[clause_chunk_bit] = ~0;
        }

        if (chunk == 0 && state_bit == 0) {
            for (int i = 0; i < CLASSES; i++) {
                if (NEGATIVE_CLAUSES) {
                    int val = 1 - 2 * (curand(&localState) % 2);
                    clause_weights[i * CLAUSES + clause] = val;
                    batch_clause_weights[i * CLAUSES + clause] = val;
                } else {
                    clause_weights[i * CLAUSES + clause] = 1;
                    batch_clause_weights[i * CLAUSES + clause] = 1;
                }
            }
        }
    }

    state[index] = localState;
}
__global__ void reset_clauses(unsigned int *global_ta_state, unsigned int *batch_ta_state) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int clause_chunk_bit = index; clause_chunk_bit < CLAUSES * LA_CHUNKS * STATE_BITS;
         clause_chunk_bit += stride) {
        int state_bit = clause_chunk_bit % STATE_BITS;
        if (state_bit == STATE_BITS - 1) {
            global_ta_state[clause_chunk_bit] = 0;
            batch_ta_state[clause_chunk_bit] = 0;
        } else {
            global_ta_state[clause_chunk_bit] = ~0;
            batch_ta_state[clause_chunk_bit] = ~0;
        }
    }
}

__global__ void reset_weights(curandState *state, int *clause_weights, int *batch_clause_weights) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    curandState localState = state[index];

    for (int class_clause = index; class_clause < CLASSES * CLAUSES; class_clause += stride) {
        if (NEGATIVE_CLAUSES) {
            int val = 1 - 2 * (curand(&localState) % 2);
            clause_weights[class_clause] = val;
            batch_clause_weights[class_clause] = val;
        } else {
            clause_weights[class_clause] = 1;
            batch_clause_weights[class_clause] = 1;
        }
    }

    state[index] = localState;
}

__global__ void prepare_packed(curandState *state, unsigned int *global_ta_state, unsigned int *included_literals,
                               unsigned int *included_literals_length, unsigned int *excluded_literals,
                               unsigned int *excluded_literals_length) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    curandState localState = state[index];

    for (int clause = index; clause < CLAUSES; clause += stride) {
        unsigned int *ta_state = &global_ta_state[clause * LA_CHUNKS * STATE_BITS];

        included_literals_length[clause] = 0;
        for (int literal = 0; literal < FEATURES; ++literal) {
            int chunk = literal / INT_SIZE;
            int pos = literal % INT_SIZE;

            if ((ta_state[chunk * STATE_BITS + STATE_BITS - 1] & (1U << pos)) > 0) {
                included_literals[clause * FEATURES * 2 + included_literals_length[clause] * 2] = literal;
                included_literals_length[clause]++;
            }
        }
    }
    state[index] = localState;
}
}
