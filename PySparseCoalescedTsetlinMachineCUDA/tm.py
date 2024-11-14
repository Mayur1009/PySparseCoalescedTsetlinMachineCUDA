# Copyright (c) 2023 Ole-Christoffer Granmo

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# This code implements the Convolutional Tsetlin Machine from paper arXiv:1905.09688
# https://arxiv.org/abs/1905.09688

import sys

import numpy as np
import pycuda.autoinit  # noqa: F401
import pycuda.curandom as curandom
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
from scipy.sparse import csr_matrix
from tqdm import tqdm

import PySparseCoalescedTsetlinMachineCUDA.kernels as kernels

g = curandom.XORWOWRandomNumberGenerator()


class CommonTsetlinMachine:
    def __init__(
        self,
        number_of_clauses: int,
        T: int | list[int] | list[tuple[int, int]],
        s: float | list[float],
        q: float = 1.0,
        max_included_literals: int | None = None,
        boost_true_positive_feedback: int = 1,
        number_of_state_bits: int = 8,
        append_negated: bool = True,
        group_ids: list = [],  # list of length number_of_classes, giving a group_id to each class, starting from 0.
        weight_update_factor: list = [],
        state_inc_factor: list = [],
        grid=(16 * 13, 1, 1),
        block=(128, 1, 1),
    ):
        # Initialize Hyperparams
        self.number_of_clauses = number_of_clauses
        self.number_of_clause_chunks = (number_of_clauses - 1) / 32 + 1
        self.number_of_state_bits = number_of_state_bits
        self.T = T
        self.s = s
        self.q = float(q)
        self.max_included_literals = max_included_literals
        self.boost_true_positive_feedback = boost_true_positive_feedback
        self.append_negated = append_negated
        self.grid = grid
        self.block = block

        self.X_train = np.array([])
        self.X_test = np.array([])
        self.encoded_Y = np.array([])
        self.ta_state = np.array([])
        self.clause_weights = np.array([])
        self.patch_weights = np.array([])
        self.group_ids = np.array(group_ids, dtype=np.uint32)
        self.weight_update_factor = np.array(weight_update_factor, dtype=np.uint32)
        self.state_inc_factor = np.array(state_inc_factor, dtype=np.uint32)

        self.negative_clauses = 1  # Default is 1, set to 0 in RegressionTsetlinMachine
        self.initialized = False

    def allocate_gpu_memory(self):
        self.ta_state_gpu = cuda.mem_alloc(
            self.number_of_groups * self.number_of_clauses * self.number_of_ta_chunks * self.number_of_state_bits * 4
        )
        self.clause_weights_gpu = cuda.mem_alloc(self.number_of_outputs * self.number_of_clauses * 4)
        self.class_sum_gpu = cuda.mem_alloc(self.number_of_outputs * 4)
        self.patch_weights_gpu = cuda.mem_alloc(
            self.number_of_outputs * self.number_of_clauses * self.number_of_patches * 4
        )

        self.included_literals_gpu = cuda.mem_alloc(
            self.number_of_groups * self.number_of_clauses * self.number_of_features * 2 * 4
        )  # Contains index and state of included literals per clause, none at start
        self.included_literals_length_gpu = cuda.mem_alloc(
            self.number_of_groups * self.number_of_clauses * 4
        )  # Number of included literals per clause

        self.excluded_literals_gpu = cuda.mem_alloc(
            self.number_of_groups * self.number_of_clauses * self.number_of_features * 2 * 4
        )  # Contains index and state of excluded literals per clause
        self.excluded_literals_length_gpu = cuda.mem_alloc(
            self.number_of_groups * self.number_of_clauses * 4
        )  # Number of excluded literals per clause

    def ta_action(self, clause, ta):
        if np.array_equal(self.ta_state, np.array([])):
            self.ta_state = np.empty(
                self.number_of_clauses * self.number_of_ta_chunks * self.number_of_state_bits, dtype=np.uint32
            )
            cuda.memcpy_dtoh(self.ta_state, self.ta_state_gpu)
        ta_state = self.ta_state.reshape((self.number_of_clauses, self.number_of_ta_chunks, self.number_of_state_bits))
        return (ta_state[clause, ta // 32, self.number_of_state_bits - 1] & (1 << (ta % 32))) > 0

    def get_literals(self):
        literals_gpu = cuda.mem_alloc(self.number_of_groups * self.number_of_clauses * self.number_of_features * 4)
        self.get_literals_gpu(
            self.ta_state_gpu,
            literals_gpu,
            grid=self.grid,
            block=self.block,
        )
        cuda.Context.synchronize()

        literals = np.empty((self.number_of_groups * self.number_of_clauses * self.number_of_features), dtype=np.uint32)
        cuda.memcpy_dtoh(literals, literals_gpu)
        return literals.reshape((self.number_of_groups, self.number_of_clauses, self.number_of_features)).astype(
            np.uint8
        )

    def get_weights(self):
        self.clause_weights = np.empty(self.number_of_outputs * self.number_of_clauses, dtype=np.int32)
        cuda.memcpy_dtoh(self.clause_weights, self.clause_weights_gpu)

        return self.clause_weights.reshape((self.number_of_outputs, self.number_of_clauses))

    def get_patch_weights(self):
        self.patch_weights = np.empty(
            self.number_of_outputs * self.number_of_clauses * self.number_of_patches, dtype=np.int32
        )
        cuda.memcpy_dtoh(self.patch_weights, self.patch_weights_gpu)

        return self.patch_weights.reshape(
            (
                self.number_of_outputs,
                self.number_of_clauses,
                self.dim[0] - self.patch_dim[0] + 1,
                self.dim[1] - self.patch_dim[1] + 1,
            )
        )

    def get_state(self):
        self.ta_state = np.empty(
            self.number_of_groups * self.number_of_clauses * self.number_of_ta_chunks * self.number_of_state_bits,
            dtype=np.uint32,
        )
        self.clause_weights = np.empty(self.number_of_outputs * self.number_of_clauses, dtype=np.int32)
        self.patch_weights = np.empty(
            self.number_of_outputs * self.number_of_clauses * self.number_of_patches, dtype=np.int32
        )
        cuda.memcpy_dtoh(self.ta_state, self.ta_state_gpu)
        cuda.memcpy_dtoh(self.clause_weights, self.clause_weights_gpu)
        cuda.memcpy_dtoh(self.patch_weights, self.patch_weights_gpu)

        return (
            self.ta_state,
            self.clause_weights,
            self.number_of_outputs,
            self.number_of_clauses,
            self.number_of_features,
            self.dim,
            self.patch_dim,
            self.number_of_patches,
            self.number_of_state_bits,
            self.number_of_ta_chunks,
            self.append_negated,
            self.min_y,
            self.max_y,
            self.patch_weights,
            self.number_of_groups,
            self.group_ids,
        )

    def set_state(self, state):
        self.number_of_outputs = state[2]
        self.number_of_clauses = state[3]
        self.number_of_features = state[4]
        self.dim = state[5]
        self.patch_dim = state[6]
        self.number_of_patches = state[7]
        self.number_of_state_bits = state[8]
        self.number_of_ta_chunks = state[9]
        self.append_negated = state[10]
        self.min_y = state[11]
        self.max_y = state[12]
        self.number_of_groups = state[14]
        self.group_ids = state[15]

        self._init_fit()
        self.init_gpu()
        cuda.memcpy_htod(self.ta_state_gpu, state[0])
        cuda.memcpy_htod(self.clause_weights_gpu, state[1])
        cuda.memcpy_htod(self.patch_weights_gpu, state[13])
        self._init_encoded_X()
        self.initialized = True

        self.X_train = np.array([])
        self.X_test = np.array([])

        self.encoded_Y = np.array([])

        self.ta_state = np.array([])
        self.clause_weights = np.array([])
        self.patch_weights = np.array([])

    # Transform input data for processing at next layer
    def transform(self, X) -> csr_matrix:
        """Returns csr_matix of clause outputs. Array shape: (num_groups, num_samples, num_clauses)"""
        if not self.initialized:
            print("Error: Model not trained.")
            sys.exit(-1)

        X = csr_matrix(X)
        number_of_examples = X.shape[0]

        # Copy data to GPU
        X_indptr_gpu = cuda.mem_alloc(X.indptr.nbytes)
        X_indices_gpu = cuda.mem_alloc(X.indices.nbytes)
        cuda.memcpy_htod(X_indptr_gpu, X.indptr)
        cuda.memcpy_htod(X_indices_gpu, X.indices)

        X_transformed = np.empty((number_of_examples, self.number_of_groups * self.number_of_clauses), dtype=np.uint32)
        X_transformed_gpu = cuda.mem_alloc(self.number_of_groups * self.number_of_clauses * 4)

        self.prepare_packed(
            g.state,
            self.ta_state_gpu,
            self.included_literals_gpu,
            self.included_literals_length_gpu,
            self.excluded_literals_gpu,
            self.excluded_literals_length_gpu,
            grid=self.grid,
            block=self.block,
        )
        cuda.Context.synchronize()

        for e in range(number_of_examples):
            self.encode_packed.prepared_call(
                self.grid,
                self.block,
                X_indptr_gpu,
                X_indices_gpu,
                self.encoded_X_packed_gpu,
                np.int32(e),
                np.int32(self.dim[0]),
                np.int32(self.dim[1]),
                np.int32(self.dim[2]),
                np.int32(self.patch_dim[0]),
                np.int32(self.patch_dim[1]),
                np.int32(self.append_negated),
                np.int32(0),
            )
            cuda.Context.synchronize()

            self.transform_gpu(
                self.included_literals_gpu,
                self.included_literals_length_gpu,
                self.encoded_X_packed_gpu,
                X_transformed_gpu,
                grid=self.grid,
                block=self.block,
            )
            cuda.Context.synchronize()

            cuda.memcpy_dtoh(X_transformed[e, :], X_transformed_gpu)

            self.restore_packed.prepared_call(
                self.grid,
                self.block,
                X_indptr_gpu,
                X_indices_gpu,
                self.encoded_X_packed_gpu,
                np.int32(e),
                np.int32(self.dim[0]),
                np.int32(self.dim[1]),
                np.int32(self.dim[2]),
                np.int32(self.patch_dim[0]),
                np.int32(self.patch_dim[1]),
                np.int32(self.append_negated),
                np.int32(0),
            )

        X_transformed = (X_transformed > 0).astype(np.uint8)
        return csr_matrix(X_transformed)

    def transform_patchwise(self, X) -> csr_matrix:
        """Returns SPARSE CSR MATRIX of patch outputs for each clause. Array shape: (num_samples, num_clauses * num_patches)"""
        if not self.initialized:
            print("Error: Model not trained.")
            sys.exit(-1)

        X = csr_matrix(X)
        number_of_examples = X.shape[0]

        # Copy data to GPU
        X_indptr_gpu = cuda.mem_alloc(X.indptr.nbytes)
        X_indices_gpu = cuda.mem_alloc(X.indices.nbytes)
        cuda.memcpy_htod(X_indptr_gpu, X.indptr)
        cuda.memcpy_htod(X_indices_gpu, X.indices)

        # Array to capture output from gpu
        X_transformed = np.empty(
            (number_of_examples, self.number_of_groups * self.number_of_clauses * self.number_of_patches),
            dtype=np.uint32,
        )
        X_transformed_gpu = cuda.mem_alloc(self.number_of_groups * self.number_of_clauses * self.number_of_patches * 4)

        self.prepare_packed(
            g.state,
            self.ta_state_gpu,
            self.included_literals_gpu,
            self.included_literals_length_gpu,
            self.excluded_literals_gpu,
            self.excluded_literals_length_gpu,
            grid=self.grid,
            block=self.block,
        )
        cuda.Context.synchronize()

        for e in range(number_of_examples):
            self.encode_packed.prepared_call(
                self.grid,
                self.block,
                X_indptr_gpu,
                X_indices_gpu,
                self.encoded_X_packed_gpu,
                np.int32(e),
                np.int32(self.dim[0]),
                np.int32(self.dim[1]),
                np.int32(self.dim[2]),
                np.int32(self.patch_dim[0]),
                np.int32(self.patch_dim[1]),
                np.int32(self.append_negated),
                np.int32(0),
            )
            cuda.Context.synchronize()

            self.transform_patchwise_gpu(
                self.included_literals_gpu,
                self.included_literals_length_gpu,
                self.encoded_X_packed_gpu,
                X_transformed_gpu,
                grid=self.grid,
                block=self.block,
            )
            cuda.Context.synchronize()

            cuda.memcpy_dtoh(X_transformed[e, :], X_transformed_gpu)

            self.restore_packed.prepared_call(
                self.grid,
                self.block,
                X_indptr_gpu,
                X_indices_gpu,
                self.encoded_X_packed_gpu,
                np.int32(e),
                np.int32(self.dim[0]),
                np.int32(self.dim[1]),
                np.int32(self.dim[2]),
                np.int32(self.patch_dim[0]),
                np.int32(self.patch_dim[1]),
                np.int32(self.append_negated),
                np.int32(0),
            )

        # NOTE: RETURNS CSR_MATRIX
        return csr_matrix(
            X_transformed.reshape(
                (number_of_examples, self.number_of_groups * self.number_of_clauses * self.number_of_patches)
            )
        )

    def init_gpu(self):
        self._init_gpu_code()
        self.allocate_gpu_memory()

    def _init_gpu_code(self):
        # Encode and pack input
        mod_encode = SourceModule(kernels.code_encode, no_extern_c=True)
        self.encode = mod_encode.get_function("encode")
        self.encode.prepare("PPPiiiiiiii")

        self.restore = mod_encode.get_function("restore")
        self.restore.prepare("PPPiiiiiiii")

        self.encode_packed = mod_encode.get_function("encode_packed")
        self.encode_packed.prepare("PPPiiiiiiii")

        self.restore_packed = mod_encode.get_function("restore_packed")
        self.restore_packed.prepare("PPPiiiiiiii")

        self.produce_autoencoder_examples = mod_encode.get_function("produce_autoencoder_example")
        self.produce_autoencoder_examples.prepare("PPiPPiPPiPPiiii")

        parameters = """
            #define CLASSES %d
            #define CLAUSES %d
            #define FEATURES %d
            #define STATE_BITS %d
            #define BOOST_TRUE_POSITIVE_FEEDBACK %d
            #define Q %f
            #define MAX_INCLUDED_LITERALS %d
            #define NEGATIVE_CLAUSES %d
            #define PATCHES %d
            #define GROUPS %d
        """ % (
            self.number_of_outputs,
            self.number_of_clauses,
            self.number_of_features,
            self.number_of_state_bits,
            self.boost_true_positive_feedback,
            self.q,
            self.max_included_literals,
            self.negative_clauses,
            self.number_of_patches,
            self.number_of_groups,
        )

        parameters = f"""
            {parameters}
            __device__ unsigned int GROUP_ID[CLASSES] = {{{','.join(self.group_ids.astype(str))}}};
            __device__ float S[GROUPS] = {{{','.join(self.s.astype(str))}}};
            __device__ int TP[GROUPS] = {{{','.join(self.Tp.astype(str))}}};
            __device__ int TN[GROUPS] = {{{','.join(self.Tn.astype(str))}}};
            __device__ int WEIGHT_UPDATE_FACTOR[CLASSES] = {{{','.join(self.weight_update_factor.astype(str))}}};
            __device__ int STATE_INC_FACTOR[CLASSES] = {{{','.join(self.state_inc_factor.astype(str))}}};
        """

        # Prepare
        mod_prepare = SourceModule(parameters + kernels.code_header + kernels.code_prepare, no_extern_c=True)
        self.prepare = mod_prepare.get_function("prepare")
        self.prepare_packed = mod_prepare.get_function("prepare_packed")

        # Update
        mod_update = SourceModule(parameters + kernels.code_header + kernels.code_update, no_extern_c=True)
        self.update = mod_update.get_function("update")
        self.update.prepare("PPPPPPPi")

        self.evaluate_update = mod_update.get_function("evaluate")
        self.evaluate_update.prepare("PPPP")

        # Evaluate
        mod_evaluate = SourceModule(parameters + kernels.code_header + kernels.code_evaluate, no_extern_c=True)
        self.evaluate = mod_evaluate.get_function("evaluate")
        self.evaluate.prepare("PPPP")

        self.evaluate_packed = mod_evaluate.get_function("evaluate_packed")
        self.evaluate_packed.prepare("PPPPPPP")

        # Transform
        mod_transform = SourceModule(parameters + kernels.code_header + kernels.code_transform, no_extern_c=True)
        self.transform_gpu = mod_transform.get_function("transform")
        self.transform_gpu.prepare("PPPP")

        self.transform_patchwise_gpu = mod_transform.get_function("transform_patchwise")
        self.transform_patchwise_gpu.prepare("PPPP")

        # Misc Clause operations
        mod_clauses = SourceModule(parameters + kernels.code_header + kernels.code_clauses, no_extern_c=True)
        self.get_literals_gpu = mod_clauses.get_function("get_literals")
        self.get_literals_gpu.prepare("PP")

    def _validate_args(self):
        if len(self.weight_update_factor) == 0:
            self.weight_update_factor = np.array([1] * self.number_of_outputs, dtype=np.uint32)
        assert (
            len(self.weight_update_factor) == self.number_of_outputs
        ), "len(weight_update_factor) should be equal to number of classes"

        if len(self.state_inc_factor) == 0:
            self.state_inc_factor = np.array([1] * self.number_of_outputs, dtype=np.uint32)
        assert (
            len(self.state_inc_factor) == self.number_of_outputs
        ), "len(state_inc_factor) should be equal to number of classes"

        if len(self.group_ids) == 0:
            self.group_ids = np.array([0] * self.number_of_outputs, dtype=np.uint32)
        assert len(self.group_ids) == self.number_of_outputs, "len(group_ids) should be equal to number of classes"
        self.number_of_groups = int(np.max(self.group_ids) + 1)  # int() is important, dont know why

        if isinstance(self.s, float) or isinstance(self.s, int):
            self.s = np.array([self.s] * self.number_of_groups, dtype=float)
        else:
            self.s = np.array(self.s, dtype=float)

        if isinstance(self.T, int):
            self.Tp = np.array([self.T] * self.number_of_groups, dtype=int)
            self.Tn = np.array([-self.T] * self.number_of_groups, dtype=int)
        elif isinstance(self.T, list):
            Tp, Tn = [], []
            for v in list(self.T):
                if isinstance(v, tuple):
                    Tn.append(v[0])
                    Tp.append(v[1])
                else:
                    Tp.append(v)
                    Tn.append(-v)

            self.Tp = np.array(Tp, dtype=int)
            self.Tn = np.array(Tn, dtype=int)

        assert (
            len(self.s) == self.number_of_groups
        ), "s should be float or list of floats with length equal to number of groups."
        assert (
            len(self.Tp) == self.number_of_groups
        ), "Something wrong with T, Tp should be float or list of floats with length equal to number of groups."
        assert (
            len(self.Tn) == self.number_of_groups
        ), "Something wrong with T,Tn should be float or list of floats with length equal to number of groups."

    def _init_fit(self):
        self._validate_args()

        if self.append_negated:
            self.number_of_features = (
                int(
                    self.patch_dim[0] * self.patch_dim[1] * self.dim[2]
                    + (self.dim[0] - self.patch_dim[0])
                    + (self.dim[1] - self.patch_dim[1])
                )
                * 2
            )
        else:
            self.number_of_features = int(
                self.patch_dim[0] * self.patch_dim[1] * self.dim[2]
                + (self.dim[0] - self.patch_dim[0])
                + (self.dim[1] - self.patch_dim[1])
            )

        if self.max_included_literals is None:
            self.max_included_literals = self.number_of_features

        self.number_of_patches = int((self.dim[0] - self.patch_dim[0] + 1) * (self.dim[1] - self.patch_dim[1] + 1))
        self.number_of_ta_chunks = int((self.number_of_features - 1) / 32 + 1)

    def _init_encoded_X(self):
        encoded_X = np.zeros((self.number_of_patches, self.number_of_ta_chunks), dtype=np.uint32)
        for patch_coordinate_y in range(self.dim[1] - self.patch_dim[1] + 1):
            for patch_coordinate_x in range(self.dim[0] - self.patch_dim[0] + 1):
                p = patch_coordinate_y * (self.dim[0] - self.patch_dim[0] + 1) + patch_coordinate_x

                if self.append_negated:
                    for k in range(self.number_of_features // 2, self.number_of_features):
                        chunk = k // 32
                        pos = k % 32
                        encoded_X[p, chunk] |= 1 << pos

                for y_threshold in range(self.dim[1] - self.patch_dim[1]):
                    patch_pos = y_threshold
                    if patch_coordinate_y > y_threshold:
                        chunk = patch_pos // 32
                        pos = patch_pos % 32
                        encoded_X[p, chunk] |= 1 << pos

                        if self.append_negated:
                            chunk = (patch_pos + self.number_of_features // 2) // 32
                            pos = (patch_pos + self.number_of_features // 2) % 32
                            encoded_X[p, chunk] &= ~(1 << pos)

                for x_threshold in range(self.dim[0] - self.patch_dim[0]):
                    patch_pos = (self.dim[1] - self.patch_dim[1]) + x_threshold
                    if patch_coordinate_x > x_threshold:
                        chunk = patch_pos // 32
                        pos = patch_pos % 32
                        encoded_X[p, chunk] |= 1 << pos

                        if self.append_negated:
                            chunk = (patch_pos + self.number_of_features // 2) // 32
                            pos = (patch_pos + self.number_of_features // 2) % 32
                            encoded_X[p, chunk] &= ~(1 << pos)

        encoded_X = encoded_X.reshape(-1)
        self.encoded_X_gpu = cuda.mem_alloc(encoded_X.nbytes)
        cuda.memcpy_htod(self.encoded_X_gpu, encoded_X)

        # Encoded X packed
        encoded_X_packed = np.zeros(((self.number_of_patches - 1) // 32 + 1, self.number_of_features), dtype=np.uint32)
        if self.append_negated:
            for p_chunk in range((self.number_of_patches - 1) // 32 + 1):
                for k in range(self.number_of_features // 2, self.number_of_features):
                    encoded_X_packed[p_chunk, k] = ~0

        for patch_coordinate_y in range(self.dim[1] - self.patch_dim[1] + 1):
            for patch_coordinate_x in range(self.dim[0] - self.patch_dim[0] + 1):
                p = patch_coordinate_y * (self.dim[0] - self.patch_dim[0] + 1) + patch_coordinate_x
                p_chunk = p // 32
                p_pos = p % 32

                for y_threshold in range(self.dim[1] - self.patch_dim[1]):
                    patch_pos = y_threshold
                    if patch_coordinate_y > y_threshold:
                        encoded_X_packed[p_chunk, patch_pos] |= 1 << p_pos

                        if self.append_negated:
                            encoded_X_packed[p_chunk, patch_pos + self.number_of_features // 2] &= ~(1 << p_pos)

                for x_threshold in range(self.dim[0] - self.patch_dim[0]):
                    patch_pos = (self.dim[1] - self.patch_dim[1]) + x_threshold
                    if patch_coordinate_x > x_threshold:
                        encoded_X_packed[p_chunk, patch_pos] |= 1 << p_pos

                        if self.append_negated:
                            encoded_X_packed[p_chunk, patch_pos + self.number_of_features // 2] &= ~(1 << p_pos)

        encoded_X_packed = encoded_X_packed.reshape(-1)
        self.encoded_X_packed_gpu = cuda.mem_alloc(encoded_X_packed.nbytes)
        cuda.memcpy_htod(self.encoded_X_packed_gpu, encoded_X_packed)

    def _fit(self, X, encoded_Y, epochs=100, incremental=False):
        # Initialize fit
        if not self.initialized:
            self._init_fit()
            self.init_gpu()
            self._init_encoded_X()
            self.prepare(
                g.state,
                self.ta_state_gpu,
                self.clause_weights_gpu,
                self.class_sum_gpu,
                grid=self.grid,
                block=self.block,
            )
            self.initialized = True
            cuda.Context.synchronize()

        # If not incremental, clear ta-state and clause_weghts
        elif not incremental:
            self.prepare(
                g.state,
                self.ta_state_gpu,
                self.clause_weights_gpu,
                self.class_sum_gpu,
                grid=self.grid,
                block=self.block,
            )
            cuda.Context.synchronize()

        # Copy data to Gpu
        if not np.array_equal(self.X_train, np.concatenate((X.indptr, X.indices))):
            self.X_train = np.concatenate((X.indptr, X.indices))
            self.X_train_indptr_gpu = cuda.mem_alloc(X.indptr.nbytes)
            cuda.memcpy_htod(self.X_train_indptr_gpu, X.indptr)

            self.X_train_indices_gpu = cuda.mem_alloc(X.indices.nbytes)
            cuda.memcpy_htod(self.X_train_indices_gpu, X.indices)

        if not np.array_equal(self.encoded_Y, encoded_Y):
            self.encoded_Y = encoded_Y
            self.encoded_Y_gpu = cuda.mem_alloc(encoded_Y.nbytes)
            cuda.memcpy_htod(self.encoded_Y_gpu, encoded_Y)

        for epoch in range(epochs):
            for e in tqdm(range(X.shape[0]), leave=False, desc="Fit"):
                class_sum = np.zeros(self.number_of_outputs).astype(np.int32)
                cuda.memcpy_htod(self.class_sum_gpu, class_sum)

                self.encode.prepared_call(
                    self.grid,
                    self.block,
                    self.X_train_indptr_gpu,
                    self.X_train_indices_gpu,
                    self.encoded_X_gpu,
                    np.int32(e),
                    np.int32(self.dim[0]),
                    np.int32(self.dim[1]),
                    np.int32(self.dim[2]),
                    np.int32(self.patch_dim[0]),
                    np.int32(self.patch_dim[1]),
                    np.int32(self.append_negated),
                    np.int32(0),
                )
                cuda.Context.synchronize()

                self.evaluate_update.prepared_call(
                    self.grid,
                    self.block,
                    self.ta_state_gpu,
                    self.clause_weights_gpu,
                    self.class_sum_gpu,
                    self.encoded_X_gpu,
                )
                cuda.Context.synchronize()

                self.update.prepared_call(
                    self.grid,
                    self.block,
                    g.state,
                    self.ta_state_gpu,
                    self.clause_weights_gpu,
                    self.patch_weights_gpu,
                    self.class_sum_gpu,
                    self.encoded_X_gpu,
                    self.encoded_Y_gpu,
                    np.int32(e),
                )
                cuda.Context.synchronize()

                self.restore.prepared_call(
                    self.grid,
                    self.block,
                    self.X_train_indptr_gpu,
                    self.X_train_indices_gpu,
                    self.encoded_X_gpu,
                    np.int32(e),
                    np.int32(self.dim[0]),
                    np.int32(self.dim[1]),
                    np.int32(self.dim[2]),
                    np.int32(self.patch_dim[0]),
                    np.int32(self.patch_dim[1]),
                    np.int32(self.append_negated),
                    np.int32(0),
                )
                cuda.Context.synchronize()

        self.ta_state = np.array([])
        self.clause_weights = np.array([])

        return

    def predict(self, X, return_class_sums=False):
        raise NotImplementedError

    def _score(self, X):
        if not self.initialized:
            print("Error: Model not trained.")
            sys.exit(-1)

        if not np.array_equal(self.X_test, np.concatenate((X.indptr, X.indices))):
            self.X_test = np.concatenate((X.indptr, X.indices))

            self.X_test_indptr_gpu = cuda.mem_alloc(X.indptr.nbytes)
            cuda.memcpy_htod(self.X_test_indptr_gpu, X.indptr)

            self.X_test_indices_gpu = cuda.mem_alloc(X.indices.nbytes)
            cuda.memcpy_htod(self.X_test_indices_gpu, X.indices)

        self.prepare_packed(
            g.state,
            self.ta_state_gpu,
            self.included_literals_gpu,
            self.included_literals_length_gpu,
            self.excluded_literals_gpu,
            self.excluded_literals_length_gpu,
            grid=self.grid,
            block=self.block,
        )
        cuda.Context.synchronize()

        class_sum = np.zeros((X.shape[0], self.number_of_outputs), dtype=np.int32)
        for e in tqdm(range(X.shape[0]), leave=False, desc="Predict"):
            cuda.memcpy_htod(self.class_sum_gpu, class_sum[e, :])

            self.encode_packed.prepared_call(
                self.grid,
                self.block,
                self.X_test_indptr_gpu,
                self.X_test_indices_gpu,
                self.encoded_X_packed_gpu,
                np.int32(e),
                np.int32(self.dim[0]),
                np.int32(self.dim[1]),
                np.int32(self.dim[2]),
                np.int32(self.patch_dim[0]),
                np.int32(self.patch_dim[1]),
                np.int32(self.append_negated),
                np.int32(0),
            )
            cuda.Context.synchronize()

            self.evaluate_packed.prepared_call(
                self.grid,
                self.block,
                self.included_literals_gpu,
                self.included_literals_length_gpu,
                self.excluded_literals_gpu,
                self.excluded_literals_length_gpu,
                self.clause_weights_gpu,
                self.class_sum_gpu,
                self.encoded_X_packed_gpu,
            )
            cuda.Context.synchronize()

            self.restore_packed.prepared_call(
                self.grid,
                self.block,
                self.X_test_indptr_gpu,
                self.X_test_indices_gpu,
                self.encoded_X_packed_gpu,
                np.int32(e),
                np.int32(self.dim[0]),
                np.int32(self.dim[1]),
                np.int32(self.dim[2]),
                np.int32(self.patch_dim[0]),
                np.int32(self.patch_dim[1]),
                np.int32(self.append_negated),
                np.int32(0),
            )
            cuda.Context.synchronize()

            cuda.memcpy_dtoh(class_sum[e, :], self.class_sum_gpu)

        return class_sum


class MultiClassConvolutionalTsetlinMachine2D(CommonTsetlinMachine):
    """
    This class ...
    """

    def __init__(
        self,
        number_of_clauses,
        T,
        s,
        dim,
        patch_dim,
        q=1.0,
        max_included_literals=None,
        boost_true_positive_feedback=1,
        number_of_state_bits=8,
        append_negated=True,
        group_ids=[],  # list of length number_of_classes, giving a group_id to each class, starting from 0.
        weight_update_factor: list = [],
        state_inc_factor: list = [],
        grid=(16 * 13, 1, 1),
        block=(128, 1, 1),
    ):
        super().__init__(
            number_of_clauses,
            T,
            s,
            q=q,
            max_included_literals=max_included_literals,
            boost_true_positive_feedback=boost_true_positive_feedback,
            number_of_state_bits=number_of_state_bits,
            append_negated=append_negated,
            group_ids=group_ids,
            weight_update_factor=weight_update_factor,
            state_inc_factor=state_inc_factor,
            grid=grid,
            block=block,
        )
        self.dim = dim
        self.patch_dim = patch_dim
        self.negative_clauses = 1

    def fit(self, X, Y, epochs=100, incremental=False):
        if len(X.shape) == 3:
            print(f"Expecting X with 2D shape, got {X.shape}. Flattening the array...")
            X = X.reshape((X.shape[0], -1))
            print(f"New X.shape => {X.shape}")
        X = csr_matrix(X)

        self.number_of_outputs = int(np.max(Y) + 1)

        self.max_y = None
        self.min_y = None

        encoded_Y = np.empty((Y.shape[0], self.number_of_outputs), dtype=np.int32)
        for i in range(self.number_of_outputs):
            encoded_Y[:, i] = np.where(Y == i, 1, 0)

        self._fit(X, encoded_Y, epochs=epochs, incremental=incremental)

    def score(self, X):
        X = csr_matrix(X)
        return self._score(X)

    def predict(self, X, return_class_sums=False):
        class_sums = self.score(X)
        preds = np.argmax(class_sums, axis=1)
        if return_class_sums:
            return preds, class_sums
        else:
            return preds


class MultiOutputConvolutionalTsetlinMachine2D(CommonTsetlinMachine):
    """
    This class ...
    """

    def __init__(
        self,
        number_of_clauses,
        T,
        s,
        dim,
        patch_dim,
        q=1.0,
        max_included_literals=None,
        boost_true_positive_feedback=1,
        number_of_state_bits=8,
        append_negated=True,
        group_ids=[],  # list of length number_of_classes, giving a group_id to each class, starting from 0.
        weight_update_factor: list = [],
        state_inc_factor: list = [],
        grid=(16 * 13, 1, 1),
        block=(128, 1, 1),
    ):
        super().__init__(
            number_of_clauses,
            T,
            s,
            q=q,
            max_included_literals=max_included_literals,
            boost_true_positive_feedback=boost_true_positive_feedback,
            number_of_state_bits=number_of_state_bits,
            append_negated=append_negated,
            group_ids=group_ids,
            weight_update_factor=weight_update_factor,
            state_inc_factor=state_inc_factor,
            grid=grid,
            block=block,
        )
        self.dim = dim
        self.patch_dim = patch_dim
        self.negative_clauses = 1

    def fit(self, X, Y, epochs=100, incremental=False):
        if len(X.shape) == 3:
            print(f"Expecting X with 2D shape, got {X.shape}. Flattening the array...")
            X = X.reshape((X.shape[0], -1))
            print(f"New X.shape => {X.shape}")
        X = csr_matrix(X)

        self.number_of_outputs = Y.shape[1]

        self.max_y = None
        self.min_y = None

        encoded_Y = np.where(Y == 1, 1, 0).astype(np.int32)

        self._fit(X, encoded_Y, epochs=epochs, incremental=incremental)

    def score(self, X):
        X = csr_matrix(X)

        return self._score(X)

    def predict(self, X, return_class_sums=False):
        if len(X.shape) == 3:
            print(f"Expecting X with 2D shape, got {X.shape}. Flattening samples...")
            X = X.reshape((X.shape[0], -1))
            print(f"New X.shape => {X.shape}")
        class_sums = self.score(X)
        preds = (class_sums >= 0).astype(np.uint32)
        if return_class_sums:
            return preds, class_sums
        else:
            return preds


class MultiOutputTsetlinMachine(CommonTsetlinMachine):
    def __init__(
        self,
        number_of_clauses,
        T,
        s,
        q=1.0,
        max_included_literals=None,
        boost_true_positive_feedback=1,
        number_of_state_bits=8,
        append_negated=True,
        group_ids=[],  # list of length number_of_classes, giving a group_id to each class, starting from 0.
        weight_update_factor: list = [],
        state_inc_factor: list = [],
        grid=(16 * 13, 1, 1),
        block=(128, 1, 1),
    ):
        super().__init__(
            number_of_clauses,
            T,
            s,
            q=q,
            max_included_literals=max_included_literals,
            boost_true_positive_feedback=boost_true_positive_feedback,
            number_of_state_bits=number_of_state_bits,
            append_negated=append_negated,
            group_ids=group_ids,
            weight_update_factor=weight_update_factor,
            state_inc_factor=state_inc_factor,
            grid=grid,
            block=block,
        )
        self.negative_clauses = 1

    def fit(self, X, Y, epochs=100, incremental=False):
        X = csr_matrix(X)

        self.number_of_outputs = Y.shape[1]

        self.dim = (X.shape[1], 1, 1)
        self.patch_dim = (X.shape[1], 1)

        self.max_y = None
        self.min_y = None

        encoded_Y = np.where(Y == 1, 1, 0).astype(np.int32)
        self._fit(X, encoded_Y, epochs=epochs, incremental=incremental)

        return

    def score(self, X):
        X = csr_matrix(X)
        return self._score(X)

    def predict(self, X, return_class_sums=True):
        if len(X.shape) == 3:
            print(f"Expecting X with 2D shape, got {X.shape}. Flattening samples...")
            X = X.reshape((X.shape[0], -1))
            print(f"New X.shape => {X.shape}")
        class_sums = self.score(X)
        preds = (class_sums >= 0).astype(np.uint32)
        if return_class_sums:
            return preds, class_sums
        else:
            return preds


class MultiClassTsetlinMachine(CommonTsetlinMachine):
    def __init__(
        self,
        number_of_clauses,
        T,
        s,
        q=1.0,
        max_included_literals=None,
        boost_true_positive_feedback=1,
        number_of_state_bits=8,
        append_negated=True,
        group_ids=[],  # list of length number_of_classes, giving a group_id to each class, starting from 0.
        weight_update_factor: list = [],
        state_inc_factor: list = [],
        grid=(16 * 13, 1, 1),
        block=(128, 1, 1),
    ):
        super().__init__(
            number_of_clauses,
            T,
            s,
            q=q,
            max_included_literals=max_included_literals,
            boost_true_positive_feedback=boost_true_positive_feedback,
            number_of_state_bits=number_of_state_bits,
            append_negated=append_negated,
            group_ids=group_ids,
            weight_update_factor=weight_update_factor,
            state_inc_factor=state_inc_factor,
            grid=grid,
            block=block,
        )
        self.negative_clauses = 1

    def fit(self, X, Y, epochs=100, incremental=False):
        X = csr_matrix(X)

        self.number_of_outputs = int(np.max(Y) + 1)

        self.dim = (X.shape[1], 1, 1)
        self.patch_dim = (X.shape[1], 1)

        self.max_y = None
        self.min_y = None

        encoded_Y = np.empty((Y.shape[0], self.number_of_outputs), dtype=np.int32)
        for i in range(self.number_of_outputs):
            encoded_Y[:, i] = np.where(Y == i, 1, 0)

        self._fit(X, encoded_Y, epochs=epochs, incremental=incremental)

        return

    def score(self, X):
        X = csr_matrix(X)
        return self._score(X)

    def predict(self, X, return_class_sums=False):
        class_sums = self.score(X)
        preds = np.argmax(class_sums, axis=1)
        if return_class_sums:
            return preds, class_sums
        else:
            return preds


class TsetlinMachine(CommonTsetlinMachine):
    def __init__(
        self,
        number_of_clauses,
        T,
        s,
        q=1.0,
        max_included_literals=None,
        boost_true_positive_feedback=1,
        number_of_state_bits=8,
        append_negated=True,
        group_ids=[],  # list of length number_of_classes, giving a group_id to each class, starting from 0.
        weight_update_factor: list = [],
        state_inc_factor: list = [],
        grid=(16 * 13, 1, 1),
        block=(128, 1, 1),
    ):
        super().__init__(
            number_of_clauses,
            T,
            s,
            q=q,
            max_included_literals=max_included_literals,
            boost_true_positive_feedback=boost_true_positive_feedback,
            number_of_state_bits=number_of_state_bits,
            append_negated=append_negated,
            group_ids=group_ids,
            weight_update_factor=weight_update_factor,
            state_inc_factor=state_inc_factor,
            grid=grid,
            block=block,
        )
        self.negative_clauses = 1

    def fit(self, X, Y, epochs=100, incremental=False):
        X = X.reshape(X.shape[0], X.shape[1], 1)

        self.number_of_outputs = 1
        self.patch_dim = (X.shape[1], 1, 1)

        self.max_y = None
        self.min_y = None

        encoded_Y = np.where(Y == 1, 1, 0).astype(np.int32)

        self._fit(X, encoded_Y, epochs=epochs, incremental=incremental)

        return

    def score(self, X):
        X = X.reshape(X.shape[0], X.shape[1], 1)
        return self._score(X)[0, :]

    def predict(self, X, return_class_sums=False):
        class_sums = self.score(X)
        preds = int(class_sums >= 0)

        if return_class_sums:
            return preds, class_sums
        else:
            return preds


# FIXME: Broken because of Multiple T values.
class RegressionTsetlinMachine(CommonTsetlinMachine):
    def __init__(
        self,
        number_of_clauses,
        T,
        s,
        max_included_literals=None,
        boost_true_positive_feedback=1,
        number_of_state_bits=8,
        append_negated=True,
        group_ids=[],  # list of length number_of_classes, giving a group_id to each class, starting from 0.
        weight_update_factor: list = [],
        state_inc_factor: list = [],
        grid=(16 * 13, 1, 1),
        block=(128, 1, 1),
    ):
        super().__init__(
            number_of_clauses,
            T,
            s,
            max_included_literals=max_included_literals,
            boost_true_positive_feedback=boost_true_positive_feedback,
            number_of_state_bits=number_of_state_bits,
            append_negated=append_negated,
            group_ids=group_ids,
            grid=grid,
            block=block,
        )
        self.negative_clauses = 0

    def fit(self, X, Y, epochs=100, incremental=False):
        X = X.reshape(X.shape[0], X.shape[1], 1)

        self.number_of_outputs = 1
        self.patch_dim = (X.shape[1], 1, 1)

        self.max_y = np.max(Y)
        self.min_y = np.min(Y)

        encoded_Y = ((Y - self.min_y) / (self.max_y - self.min_y) * self.T).astype(np.int32)

        self._fit(X, encoded_Y, epochs=epochs, incremental=incremental)

        return

    def predict(self, X, return_class_sums=False):
        X = X.reshape(X.shape[0], X.shape[1], 1)
        class_sums = self._score(X)
        preds = 1.0 * (class_sums[0, :]) * (self.max_y - self.min_y) / (self.T) + self.min_y

        if return_class_sums:
            return preds, class_sums
        else:
            return preds


# FIXME: Broken because of patch_weights and other things.
class AutoEncoderTsetlinMachine(CommonTsetlinMachine):
    def __init__(
        self,
        number_of_clauses,
        T,
        s,
        active_output,
        q=1.0,
        max_included_literals=None,
        accumulation=1,
        boost_true_positive_feedback=1,
        number_of_state_bits=8,
        append_negated=True,
        grid=(16 * 13, 1, 1),
        block=(128, 1, 1),
    ):
        super().__init__(
            number_of_clauses,
            T,
            s,
            q=q,
            max_included_literals=max_included_literals,
            boost_true_positive_feedback=boost_true_positive_feedback,
            number_of_state_bits=number_of_state_bits,
            append_negated=append_negated,
            grid=grid,
            block=block,
        )
        self.negative_clauses = 1

        self.active_output = np.array(active_output).astype(np.uint32)
        self.accumulation = accumulation

    # FIX
    def _init_fit(self, X_csr, encoded_Y, incremental):
        if not self.initialized:
            self._init(X_csr)
            self.prepare(
                g.state,
                self.ta_state_gpu,
                self.clause_weights_gpu,
                self.class_sum_gpu,
                grid=self.grid,
                block=self.block,
            )
            cuda.Context.synchronize()

        elif not incremental:
            self.prepare(
                g.state,
                self.ta_state_gpu,
                self.clause_weights_gpu,
                self.class_sum_gpu,
                grid=self.grid,
                block=self.block,
            )
            cuda.Context.synchronize()

        if not np.array_equal(self.X_train, np.concatenate((X_csr.indptr, X_csr.indices))):
            self.train_X = np.concatenate((X_csr.indptr, X_csr.indices))

            X_csc = X_csr.tocsc()

            self.X_train_csr_indptr_gpu = cuda.mem_alloc(X_csr.indptr.nbytes)
            cuda.memcpy_htod(self.X_train_csr_indptr_gpu, X_csr.indptr)

            self.X_train_csr_indices_gpu = cuda.mem_alloc(X_csr.indices.nbytes)
            cuda.memcpy_htod(self.X_train_csr_indices_gpu, X_csr.indices)

            self.X_train_csc_indptr_gpu = cuda.mem_alloc(X_csc.indptr.nbytes)
            cuda.memcpy_htod(self.X_train_csc_indptr_gpu, X_csc.indptr)

            self.X_train_csc_indices_gpu = cuda.mem_alloc(X_csc.indices.nbytes)
            cuda.memcpy_htod(self.X_train_csc_indices_gpu, X_csc.indices)

            self.encoded_Y_gpu = cuda.mem_alloc(encoded_Y.nbytes)
            cuda.memcpy_htod(self.encoded_Y_gpu, encoded_Y)

            self.active_output_gpu = cuda.mem_alloc(self.active_output.nbytes)
            cuda.memcpy_htod(self.active_output_gpu, self.active_output)

    def _fit(self, X_csr, encoded_Y, number_of_examples, epochs, incremental=False):
        self._init_fit(X_csr, encoded_Y, incremental=incremental)

        for epoch in range(epochs):
            for e in range(number_of_examples):
                class_sum = np.zeros(self.number_of_outputs).astype(np.int32)
                cuda.memcpy_htod(self.class_sum_gpu, class_sum)

                target = np.random.choice(self.number_of_outputs)
                self.produce_autoencoder_examples.prepared_call(
                    self.grid,
                    self.block,
                    g.state,
                    self.active_output_gpu,
                    self.active_output.shape[0],
                    self.X_train_csr_indptr_gpu,
                    self.X_train_csr_indices_gpu,
                    X_csr.shape[0],
                    self.X_train_csc_indptr_gpu,
                    self.X_train_csc_indices_gpu,
                    X_csr.shape[1],
                    self.encoded_X_gpu,
                    self.encoded_Y_gpu,
                    target,
                    int(self.accumulation),
                    int(self.T),
                    int(self.append_negated),
                )
                cuda.Context.synchronize()

                self.evaluate_update.prepared_call(
                    self.grid,
                    self.block,
                    self.ta_state_gpu,
                    self.clause_weights_gpu,
                    self.class_sum_gpu,
                    self.encoded_X_gpu,
                )
                cuda.Context.synchronize()

                self.update.prepared_call(
                    self.grid,
                    self.block,
                    g.state,
                    self.ta_state_gpu,
                    self.clause_weights_gpu,
                    self.class_sum_gpu,
                    self.encoded_X_gpu,
                    self.encoded_Y_gpu,
                    np.int32(0),
                )
                cuda.Context.synchronize()

        self.ta_state = np.array([])
        self.clause_weights = np.array([])

        return

    def fit(self, X, number_of_examples=2000, epochs=100, incremental=False):
        X_csr = csr_matrix(X)

        self.number_of_outputs = self.active_output.shape[0]

        self.dim = (X_csr.shape[1], 1, 1)
        self.patch_dim = (X_csr.shape[1], 1)

        self.max_y = None
        self.min_y = None

        encoded_Y = np.zeros(self.number_of_outputs, dtype=np.int32)

        self._fit(X_csr, encoded_Y, number_of_examples, epochs, incremental=incremental)

        return

    def score(self, X):
        X = csr_matrix(X)
        return self._score(X)

    def predict(self, X, return_class_sums=False):
        class_sums = self.score(X)
        preds = np.argmax(class_sums, axis=1)
        if return_class_sums:
            return preds, class_sums
        else:
            return preds
