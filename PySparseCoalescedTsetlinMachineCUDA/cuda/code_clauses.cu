#include <curand_kernel.h>
extern "C" {

__global__ void get_literals(unsigned int *ta_state, unsigned int *out) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (unsigned long long clause = index; clause < CLAUSES; clause += stride) {
        for (int feature = 0; feature < FEATURES; feature++) {
            out[clause * FEATURES + feature] =
                (ta_state[(clause * LA_CHUNKS * STATE_BITS) + ((feature / INT_SIZE) * STATE_BITS) + (STATE_BITS - 1)] &
                 (1 << (feature % INT_SIZE))) > 0;
        }
    }
}
}
