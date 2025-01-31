#include <curand_kernel.h>
extern "C" {

__global__ void get_literals(unsigned int *global_ta_state, unsigned int *out) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int combined_clause_id = index; combined_clause_id < GROUPS * CLAUSES; combined_clause_id += stride) {
        int group_id = combined_clause_id / CLAUSES;
        int clause = combined_clause_id % CLAUSES;
        unsigned int *ta_state =
            &global_ta_state[group_id * CLAUSES * LA_CHUNKS * STATE_BITS + clause * LA_CHUNKS * STATE_BITS];

        for (int feature = 0; feature < FEATURES; feature++) {
            int chunk_nr = feature / INT_SIZE;
            int chunk_pos = feature % INT_SIZE;
            out[group_id * CLAUSES * FEATURES + clause * FEATURES + feature] =
                (ta_state[(chunk_nr * STATE_BITS) + (STATE_BITS - 1)] & (1 << chunk_pos)) > 0;
        }
    }
}

__global__ void get_ta_states(unsigned int *global_ta_state, unsigned int *out) {
    // :param: global_ta_state
    // Array of TAs for each literal in each clause.
    // Shape: (          GROUPS,           CLAUSES,        LA_CHUNKS,           STATE_BITS)
    //        (number of groups, number of clauses, number of chunks, number of state bits)
    //
    // :param: out
    // Output array to store the state values of each TA.
    // Shape: (GROUPS, CLAUSES, FEATURES)

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int combined_clause_id = index; combined_clause_id < GROUPS * CLAUSES; combined_clause_id += stride) {
        int group_id = combined_clause_id / CLAUSES;
        int clause = combined_clause_id % CLAUSES;
        unsigned int *ta_state =
            &global_ta_state[group_id * CLAUSES * LA_CHUNKS * STATE_BITS + clause * LA_CHUNKS * STATE_BITS];

        for (int feature = 0; feature < FEATURES; feature++) {
            unsigned int state = 0;
            int chunk_nr = feature / INT_SIZE;
            int chunk_pos = feature % INT_SIZE;
            for (int b = 0; b < STATE_BITS; ++b)
                if (ta_state[chunk_nr * STATE_BITS + b] & (1 << chunk_pos)) state |= (1 << b);

            out[group_id * CLAUSES * FEATURES + clause * FEATURES + feature] = state;
        }
    }
}
}
