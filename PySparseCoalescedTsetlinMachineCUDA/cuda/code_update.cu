#include <curand_kernel.h>

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
__device__ inline void inc(unsigned int *ta_state, int clause, int chunk, unsigned int active) {
    unsigned int carry, carry_next;
    int id = clause * LA_CHUNKS * STATE_BITS + chunk * STATE_BITS;
    carry = active;
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

__device__ inline unsigned int get_state(unsigned int *ta_state, int clause, int chunk, int literal) {
    unsigned int state = 0;
    int id = clause * LA_CHUNKS * STATE_BITS + chunk * STATE_BITS;
    for (int b = 0; b < STATE_BITS; ++b) {
        if (ta_state[id + b] & (1 << literal)) {
            state |= (1 << b);
        }
    }
    return state;
}

__device__ inline void update_clause(curandState *localState, int *clause_weight, unsigned int *ta_state, int thresh,
                                     float s, float sr, float q, int clause_output, int clause_patch, int *X, int y,
                                     int class_sum, unsigned int clause_freeze_flag, unsigned int weights_freeze_flag) {
    int target = 1 - 2 * (class_sum > y);

    if (target == -1 && curand_uniform(localState) > 1.0 * q / max(1, CLASSES - 1)) {
        return;
    }

    int sign = (*clause_weight >= 0) - (*clause_weight < 0);

    int absolute_prediction_error = abs(y - class_sum);
    if (curand_uniform(localState) <= 1.0 * absolute_prediction_error / (2 * thresh)) {
        if (target * sign > 0) {
            int included_literals = number_of_include_actions(ta_state);

            if (weights_freeze_flag == 0 && clause_output && abs(*clause_weight) < INT_MAX) {
                (*clause_weight) += sign;
            }

            if (clause_freeze_flag == 0) {
                // Type I Feedback
                for (int la_chunk = 0; la_chunk < LA_CHUNKS; ++la_chunk) {
                    // Generate random bit values

                    if (clause_output && included_literals <= MAX_INCLUDED_LITERALS) {
                        unsigned int la_feedback = 0;
                        for (int b = 0; b < INT_SIZE; ++b) {
                            if (curand_uniform(localState) <= 1.0 / s) {
                                la_feedback |= (1 << b);
                            }
                        }

#if BOOST_TRUE_POSITIVE_FEEDBACK == 1
                        inc(ta_state, 0, la_chunk, X[clause_patch * LA_CHUNKS + la_chunk]);
#else
                        inc(ta_state, 0, la_chunk, X[clause_patch * LA_CHUNKS + la_chunk] & (~la_feedback));
#endif

                        dec(ta_state, 0, la_chunk, (~X[clause_patch * LA_CHUNKS + la_chunk]) & la_feedback);
                    } else {
                        // Modification with resistance
                        int la_feedback = 0;

                        for (int b = 0; b < INT_SIZE; ++b) {
                            if (sr != s && (ta_state[la_chunk * STATE_BITS + STATE_BITS - 1] & (1 << b))) {
                                unsigned int cur_state = get_state(ta_state, 0, la_chunk, b);
                                float t1 = (sr - s) / (sr);
                                float t2 = (2.0 * (float)cur_state - (float)MAX_STATE) / ((float)MAX_STATE + 2.0);
                                float mod = 1 - (t1 * pow(t2, 1 / RT));
                                if (curand_uniform(localState) <= mod / s) la_feedback |= (1 << b);
                            } else {
                                if (curand_uniform(localState) <= 1.0 / s) la_feedback |= (1 << b);
                            }
                        }
                        dec(ta_state, 0, la_chunk, la_feedback);
                    }
                }
            }
        } else if (target * sign < 0 && clause_output) {
            // Type II Feedback

            if (weights_freeze_flag == 0) {
                (*clause_weight) -= sign;
#if NEGATIVE_CLAUSES == 0
                if (*clause_weight < 1) {
                    *clause_weight = 1;
                }
#endif
            }

            if (clause_freeze_flag == 0) {
                for (int la_chunk = 0; la_chunk < LA_CHUNKS; ++la_chunk) {
                    inc(ta_state, 0, la_chunk,
                        (~X[clause_patch * LA_CHUNKS + la_chunk]) &
                            (~ta_state[la_chunk * STATE_BITS + STATE_BITS - 1]));
                }
            }
        }
    }
}

// Evaluate example
__global__ void evaluate(curandState *state, unsigned int *global_ta_state, int *clause_weights, int *class_sum,
                         unsigned int *clause_outputs, int *clause_patches, int *X) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    curandState localState = state[index];

    for (int combined_clause_id = index; combined_clause_id < GROUPS * CLAUSES; combined_clause_id += stride) {
        int group_id = combined_clause_id / CLAUSES;
        int clause = combined_clause_id % CLAUSES;
        unsigned int *ta_state =
            &global_ta_state[group_id * CLAUSES * LA_CHUNKS * STATE_BITS + clause * LA_CHUNKS * STATE_BITS];

        int output_one_patches[PATCHES];
        int output_one_patches_count = 0;
        for (int patch = 0; patch < PATCHES; ++patch) {
            int patch_clause_output = 1;
            for (int la_chunk = 0; la_chunk < LA_CHUNKS - 1; ++la_chunk) {
                if ((ta_state[la_chunk * STATE_BITS + STATE_BITS - 1] & X[patch * LA_CHUNKS + la_chunk]) !=
                    ta_state[la_chunk * STATE_BITS + STATE_BITS - 1]) {
                    patch_clause_output = 0;
                    break;
                }
            }

            if ((ta_state[(LA_CHUNKS - 1) * STATE_BITS + STATE_BITS - 1] & X[patch * LA_CHUNKS + LA_CHUNKS - 1] &
                 FILTER) != (ta_state[(LA_CHUNKS - 1) * STATE_BITS + STATE_BITS - 1] & FILTER)) {
                patch_clause_output = 0;
            }

            if (patch_clause_output) {
                output_one_patches[output_one_patches_count] = patch;
                output_one_patches_count++;
            }
        }

        if (output_one_patches_count > 0) {
            clause_outputs[combined_clause_id] = 1;
            int patch_id = curand(&localState) % output_one_patches_count;
            clause_patches[combined_clause_id] = output_one_patches[patch_id];
        } else {
            clause_outputs[combined_clause_id] = 0;
            clause_patches[combined_clause_id] = -1;
        }

        if (clause_outputs[combined_clause_id]) {
            for (int class_id = 0; class_id < CLASSES; ++class_id) {
                if (group_id == GROUP_ID[class_id]) {
                    int clause_weight = clause_weights[class_id * CLAUSES + clause];
                    atomicAdd(&class_sum[class_id], clause_weight);
                }
            }
        }
    }

    state[index] = localState;
}

// Update state of Tsetlin Automata team
__global__ void update(curandState *state, unsigned int *global_ta_state, int *clause_weights, int *patch_weights,
                       int *class_sum, unsigned int *clause_outputs, int *clause_patches, int *X, int *y, int example,
                       unsigned int *clause_freeze, unsigned int *weights_freeze) {
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

        unsigned int clause_output = clause_outputs[combined_clause_id];
        int clause_patch = clause_patches[combined_clause_id];
        // calculate_clause_output(&localState, ta_state, &clause_output, &clause_patch, X);

        unsigned int clause_freeze_flag = clause_freeze[combined_clause_id];
        unsigned int weights_freeze_flag = weights_freeze[combined_clause_id];

        for (unsigned long long class_id = 0; class_id < CLASSES; ++class_id) {
            if (group_id == GROUP_ID[class_id]) {
                int local_class_sum = class_sum[class_id];
                if (local_class_sum > T[class_id]) {
                    local_class_sum = T[class_id];
                } else if (local_class_sum < -T[class_id]) {
                    local_class_sum = -T[class_id];
                }
                int enc_y = y[example * CLASSES + class_id];
                if (enc_y > 0)
                    enc_y = T[class_id];
                else
                    enc_y = -T[class_id];

                if (clause_patch >= 0)
                    patch_weights[class_id * CLAUSES * PATCHES + clause * PATCHES + clause_patch] += 1;

                update_clause(&localState, &clause_weights[class_id * CLAUSES + clause], ta_state, T[class_id],
                              S[class_id], SR[class_id], Q[class_id], clause_output, clause_patch, X, enc_y,
                              local_class_sum, clause_freeze_flag, weights_freeze_flag);
            }
        }
    }

    state[index] = localState;
}
}
