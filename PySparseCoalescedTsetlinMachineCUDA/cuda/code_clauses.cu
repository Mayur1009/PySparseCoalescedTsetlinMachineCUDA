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
            out[group_id * CLAUSES * FEATURES + clause * FEATURES + feature] =
                (ta_state[((feature / INT_SIZE) * STATE_BITS) + (STATE_BITS - 1)] & (1 << (feature % INT_SIZE))) > 0;
        }
    }
}
}
