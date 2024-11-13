#include <curand_kernel.h>
extern "C" {

__global__ void get_literals(unsigned int *global_ta_state, unsigned int *out) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (unsigned long long clause = index; clause < CLAUSES; clause += stride) {
        for (unsigned int group_id = 0; group_id < GROUPS; group_id++) {
            unsigned int *ta_state =
                &global_ta_state[group_id * CLAUSES * LA_CHUNKS * STATE_BITS + clause * LA_CHUNKS * STATE_BITS];

            for (int feature = 0; feature < FEATURES; feature++) {
                out[group_id * CLAUSES * FEATURES + clause * FEATURES + feature] =
                    (ta_state[((feature / INT_SIZE) * STATE_BITS) + (STATE_BITS - 1)] & (1 << (feature % INT_SIZE))) >
                    0;
            }
        }
    }
}
}
