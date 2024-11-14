#include <curand_kernel.h>

extern "C" {
// Evaluate examples
__global__ void evaluate(unsigned int *global_ta_state, int *clause_weights, int *class_sum, int *X) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int combined_clause_id = index; combined_clause_id < GROUPS * CLAUSES; combined_clause_id += stride) {
        int group_id = combined_clause_id / CLAUSES;
        int clause = combined_clause_id % CLAUSES;
        unsigned int *ta_state =
            &global_ta_state[group_id * CLAUSES * LA_CHUNKS * STATE_BITS + clause * LA_CHUNKS * STATE_BITS];

        int all_exclude = 1;
        for (int la_chunk = 0; la_chunk < LA_CHUNKS - 1; ++la_chunk) {
            if (ta_state[la_chunk * STATE_BITS + STATE_BITS - 1] > 0) {
                all_exclude = 0;
                break;
            }
        }

        if ((ta_state[(LA_CHUNKS - 1) * STATE_BITS + STATE_BITS - 1] & FILTER) > 0) {
            all_exclude = 0;
        }

        if (all_exclude) {
            continue;
        }

        int clause_output;
        for (int patch = 0; patch < PATCHES; ++patch) {
            clause_output = 1;
            for (int la_chunk = 0; la_chunk < LA_CHUNKS - 1; ++la_chunk) {
                if ((ta_state[la_chunk * STATE_BITS + STATE_BITS - 1] & X[patch * LA_CHUNKS + la_chunk]) !=
                    ta_state[la_chunk * STATE_BITS + STATE_BITS - 1]) {
                    clause_output = 0;
                    break;
                }
            }

            if ((ta_state[(LA_CHUNKS - 1) * STATE_BITS + STATE_BITS - 1] & X[patch * LA_CHUNKS + LA_CHUNKS - 1] &
                 FILTER) != (ta_state[(LA_CHUNKS - 1) * STATE_BITS + STATE_BITS - 1] & FILTER)) {
                clause_output = 0;
            }

            if (clause_output) {
                break;
            }
        }

        if (clause_output) {
            for (int class_id = 0; class_id < CLASSES; ++class_id) {
                if (group_id == GROUP_ID[class_id]) {
                    int clause_weight = clause_weights[class_id * CLAUSES + clause];
                    atomicAdd(&class_sum[class_id], clause_weight);
                }
            }
        }
    }
}

// Evaluate examples
__global__ void evaluate_packed(unsigned int *included_literals, unsigned int *included_literals_length,
                                unsigned int *excluded_literals, unsigned int *excluded_literals_length,
                                int *clause_weights, int *class_sum, int *X) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int combined_clause_id = index; combined_clause_id < GROUPS * CLAUSES; combined_clause_id += stride) {
        int group_id = combined_clause_id / CLAUSES;
        int clause = combined_clause_id % CLAUSES;
        if (included_literals_length[group_id * CLAUSES + clause] == 0) {
            continue;
        }

        unsigned int clause_output = 0;
        for (int patch_chunk = 0; patch_chunk < PATCH_CHUNKS - 1; ++patch_chunk) {
            clause_output = (~(0U));
            for (int literal = 0; literal < included_literals_length[group_id * CLAUSES + clause]; ++literal) {
                clause_output &=
                    X[patch_chunk * FEATURES +
                      included_literals[group_id * CLAUSES * FEATURES * 2 + clause * FEATURES * 2 + literal * 2]];
            }

            if (clause_output) {
                break;
            }
        }

        if (!clause_output) {
            clause_output = PATCH_FILTER;
            for (int literal = 0; literal < included_literals_length[group_id * CLAUSES + clause]; ++literal) {
                clause_output &=
                    X[(PATCH_CHUNKS - 1) * FEATURES +
                      included_literals[group_id * CLAUSES * FEATURES * 2 + clause * FEATURES * 2 + literal * 2]];
            }
        }

        if (clause_output) {
            for (int class_id = 0; class_id < CLASSES; ++class_id) {
                if (group_id == GROUP_ID[class_id]) {
                    int clause_weight = clause_weights[class_id * CLAUSES + clause];
                    atomicAdd(&class_sum[class_id], clause_weight);
                }
            }
        }
    }
}
}
