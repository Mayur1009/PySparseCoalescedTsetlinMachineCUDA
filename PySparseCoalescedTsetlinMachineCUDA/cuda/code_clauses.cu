#include <curand_kernel.h>
extern "C" {

__global__ void get_literals(unsigned int *global_ta_state, unsigned int *out) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int clause_feature = index; clause_feature < CLAUSES * FEATURES; clause_feature += stride) {
        int clause = clause_feature / FEATURES;
        int chunk_nr = (clause_feature % FEATURES) / INT_SIZE;
        int chunk_pos = (clause_feature % FEATURES) % INT_SIZE;
        unsigned int *ta_state = &global_ta_state[clause * LA_CHUNKS * STATE_BITS];
        out[clause_feature] = (ta_state[chunk_nr * STATE_BITS + STATE_BITS - 1] & (1 << chunk_pos)) > 0;
    }
}

__global__ void get_ta_states(unsigned int *global_ta_state, unsigned int *out) {
    // :param: global_ta_state
    // Array of TAs for each literal in each clause.
    // Shape: (          CLAUSES,        LA_CHUNKS,           STATE_BITS)
    //        (number of clauses, number of chunks, number of state bits)
    //
    // :param: out
    // Output array to store the state values of each TA.
    // Shape: (CLAUSES, FEATURES)

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int clause_feature = index; clause_feature < CLAUSES * FEATURES; clause_feature += stride) {
        int clause = clause_feature / FEATURES;
        int chunk_nr = (clause_feature % FEATURES) / INT_SIZE;
        int chunk_pos = (clause_feature % FEATURES) % INT_SIZE;
        unsigned int state = 0;
        unsigned int *ta_state = &global_ta_state[clause * LA_CHUNKS * STATE_BITS];
        for (int b = 0; b < STATE_BITS; ++b)
            if (ta_state[chunk_nr * STATE_BITS + b] & (1 << chunk_pos)) state |= (1 << b);

        out[clause_feature] = state;
    }
}
}
