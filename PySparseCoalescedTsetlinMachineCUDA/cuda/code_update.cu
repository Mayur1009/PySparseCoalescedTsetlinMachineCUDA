#include <curand_kernel.h>

#include <cstdio>
extern "C" {
// Counts number of include actions for a given clause
__device__ inline int number_of_include_actions(unsigned int *ta_state) {
    int number_of_include_actions = 0;
    for (int k = 0; k < LA_CHUNKS - 1; ++k) {
        unsigned int ta_pos = k * STATE_BITS + STATE_BITS - 1;
        number_of_include_actions += __popc(ta_state[ta_pos]);
    }
    unsigned int ta_pos = (LA_CHUNKS - 1) * STATE_BITS + STATE_BITS - 1;
    number_of_include_actions += __popc(ta_state[ta_pos] & FILTER);

    return (number_of_include_actions);
}

// Increment the states of each of those 32 Tsetlin Automata flagged in the
// active bit vector.
__device__ inline void inc(unsigned int *ta_state, int clause, int chunk, unsigned int active,
                           unsigned int factor = 1) {
    unsigned int carry, carry_next;
    int id = clause * LA_CHUNKS * STATE_BITS + chunk * STATE_BITS;
    carry = active;
    for (int f = 0; f < factor; f++) {
        for (int b = 0; b < STATE_BITS; ++b) {
            if (carry == 0) break;

            carry_next = ta_state[id + b] & carry;        // Sets carry bits (overflow)
                                                          // passing on to next bit
            ta_state[id + b] = ta_state[id + b] ^ carry;  // Performs increments with XOR
            carry = carry_next;
        }

        if (carry > 0) {
            for (int b = 0; b < STATE_BITS; ++b) {
                ta_state[id + b] |= carry;
            }
        }
    }
}

// Decrement the states of each of those 32 Tsetlin Automata flagged in the
// active bit vector.
__device__ inline void dec(unsigned int *ta_state, int clause, int chunk, unsigned int active) {
    unsigned int carry, carry_next;
    int id = clause * LA_CHUNKS * STATE_BITS + chunk * STATE_BITS;
    carry = active;
    for (int b = 0; b < STATE_BITS; ++b) {
        if (carry == 0) break;
        carry_next = (~ta_state[id + b]) & carry;     // Sets carry bits (overflow) passing on
                                                      // to next bit
        ta_state[id + b] = ta_state[id + b] ^ carry;  // Performs increments with XOR
        carry = carry_next;
    }

    if (carry > 0) {
        for (int b = 0; b < STATE_BITS; ++b) {
            ta_state[id + b] &= ~carry;
        }
    }
}

__device__ inline void calculate_clause_output(curandState *localState, unsigned int *ta_state,
                                               unsigned int *clause_output, int *clause_patch, int *X) {
    int output_one_patches[PATCHES];
    int output_one_patches_count;

    // Evaluate each patch (convolution)
    output_one_patches_count = 0;
    for (int patch = 0; patch < PATCHES; ++patch) {
        int patch_clause_output = 1;
        for (int la_chunk = 0; la_chunk < LA_CHUNKS - 1; ++la_chunk) {
            if ((ta_state[la_chunk * STATE_BITS + STATE_BITS - 1] & X[patch * LA_CHUNKS + la_chunk]) !=
                ta_state[la_chunk * STATE_BITS + STATE_BITS - 1]) {
                patch_clause_output = 0;
                break;
            }
        }

        if (((ta_state[(LA_CHUNKS - 1) * STATE_BITS + STATE_BITS - 1] & X[patch * LA_CHUNKS + LA_CHUNKS - 1] &
              FILTER) != (ta_state[(LA_CHUNKS - 1) * STATE_BITS + STATE_BITS - 1] & FILTER))) {
            patch_clause_output = 0;
        }

        if (patch_clause_output) {
            output_one_patches[output_one_patches_count] = patch;
            output_one_patches_count++;
        }
    }

    if (output_one_patches_count > 0) {
        *clause_output = 1;
        int patch_id = curand(localState) % output_one_patches_count;
        *clause_patch = output_one_patches[patch_id];
    } else {
        *clause_output = 0;
        *clause_patch = -1;
    }
}

__device__ inline void update_clause(curandState *localState, int *clause_weight, unsigned int *ta_state, int tp,
                                     int tn, float s, int clause_output, int clause_patch, int *X, int y, int class_sum,
                                     unsigned int weight_factor = 1, unsigned int state_inc_factor = 1) {
    int target = 1 - 2 * (class_sum > y);

    if (target == -1 && curand_uniform(localState) > 1.0 * Q / max(1, CLASSES - 1)) {
        return;
    }

    int sign = (*clause_weight >= 0) - (*clause_weight < 0);

    int absolute_prediction_error = abs(y - class_sum);
    if (curand_uniform(localState) <= 1.0 * absolute_prediction_error / (tp - tn)) {
        if (target * sign > 0) {
            int included_literals = number_of_include_actions(ta_state);

            if (clause_output && abs(*clause_weight) < INT_MAX) {
                (*clause_weight) += sign * weight_factor;
            }

            // Type I Feedback
            for (int la_chunk = 0; la_chunk < LA_CHUNKS; ++la_chunk) {
                // Generate random bit values
                unsigned int la_feedback = 0;
                for (int b = 0; b < INT_SIZE; ++b) {
                    if (curand_uniform(localState) <= 1.0 / s) {
                        la_feedback |= (1 << b);
                    }
                }

                if (clause_output && included_literals <= MAX_INCLUDED_LITERALS) {
#if BOOST_TRUE_POSITIVE_FEEDBACK == 1
                    inc(ta_state, 0, la_chunk, X[clause_patch * LA_CHUNKS + la_chunk]);
#else
                    inc(ta_state, 0, la_chunk, X[clause_patch * LA_CHUNKS + la_chunk] & (~la_feedback),
                        state_inc_factor);
#endif

                    dec(ta_state, 0, la_chunk, (~X[clause_patch * LA_CHUNKS + la_chunk]) & la_feedback);
                } else {
                    dec(ta_state, 0, la_chunk, la_feedback);
                }
            }
        } else if (target * sign < 0 && clause_output) {
            // Type II Feedback

            (*clause_weight) -= sign;
#if NEGATIVE_CLAUSES == 0
            if (*clause_weight < 1) {
                *clause_weight = 1;
            }
#endif

            for (int la_chunk = 0; la_chunk < LA_CHUNKS; ++la_chunk) {
                inc(ta_state, 0, la_chunk,
                    (~X[clause_patch * LA_CHUNKS + la_chunk]) & (~ta_state[la_chunk * STATE_BITS + STATE_BITS - 1]),
                    state_inc_factor);
            }
        }
    }
}

// Evaluate example
__global__ void evaluate(unsigned int *global_ta_state, int *clause_weights, int *class_sum, int *X) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int combined_clause_id = index; combined_clause_id < GROUPS * CLAUSES; combined_clause_id += stride) {
        int group_id = combined_clause_id / CLAUSES;
        int clause = combined_clause_id % CLAUSES;
        unsigned int *ta_state =
            &global_ta_state[group_id * CLAUSES * LA_CHUNKS * STATE_BITS + clause * LA_CHUNKS * STATE_BITS];

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

// Update state of Tsetlin Automata team
__global__ void update(curandState *state, unsigned int *global_ta_state, int *clause_weights, int *patch_weights,
                       int *class_sum, int *X, int *y, int example) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    /* Copy state to local memory for efficiency */
    curandState localState = state[index];

    // Calculate clause output first
    for (int combined_clause_id = index; combined_clause_id < GROUPS * CLAUSES; combined_clause_id += stride) {
        int group_id = combined_clause_id / CLAUSES;
        int clause = combined_clause_id % CLAUSES;
        unsigned int *ta_state =
            &global_ta_state[group_id * CLAUSES * LA_CHUNKS * STATE_BITS + clause * LA_CHUNKS * STATE_BITS];

        unsigned int clause_output;
        int clause_patch;
        calculate_clause_output(&localState, ta_state, &clause_output, &clause_patch, X);

        for (unsigned long long class_id = 0; class_id < CLASSES; ++class_id) {
            if (group_id == GROUP_ID[class_id]) {
                int local_class_sum = class_sum[class_id];
                if (local_class_sum > TP[class_id]) {
                    local_class_sum = TP[class_id];
                } else if (local_class_sum < TN[class_id]) {
                    local_class_sum = TN[class_id];
                }
                int enc_y = y[example * CLASSES + class_id];
                if (enc_y > 0)
                    enc_y = TP[class_id];
                else
                    enc_y = TN[class_id];

                if (clause_patch >= 0)
                    patch_weights[class_id * CLAUSES * PATCHES + clause * PATCHES + clause_patch] += 1;

                update_clause(&localState, &clause_weights[class_id * CLAUSES + clause], ta_state, TP[class_id],
                              TN[class_id], S[class_id], clause_output, clause_patch, X, enc_y, local_class_sum,
                              WEIGHT_UPDATE_FACTOR[class_id], STATE_INC_FACTOR[class_id]);
            }
        }
    }

    state[index] = localState;
}
}
