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

from typing import Literal

import numpy as np
from scipy.sparse import csr_matrix

from PySparseCoalescedTsetlinMachineCUDA.base import CommonTsetlinMachine


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
		q: float = 1.0,
		max_included_literals=None,
		boost_true_positive_feedback=1,
		number_of_state_bits=8,
		append_negated=True,
		r: float = 1.0,
		sr: float | None = None,
		encode_loc: bool = True,
		max_weight: int | None = None,
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
			r=r,
			sr=sr,
			encode_loc=encode_loc,
			max_weight=max_weight,
			grid=grid,
			block=block,
		)
		self.dim = dim
		self.patch_dim = patch_dim
		# self.negative_clauses = 1

	def fit(self, X, Y, epochs=100, incremental=False):
		if len(X.shape) == 3:
			print(f"Expecting X with 2D shape, got {X.shape}. Flattening the array...")
			X = X.reshape((X.shape[0], -1))
			print(f"New X.shape => {X.shape}")
		X = csr_matrix(X)

		self.number_of_outputs = int(np.max(Y) + 1)
		self.negative_clauses = np.ones(self.number_of_outputs, dtype=np.uint32)

		self.max_y = None
		self.min_y = None

		encoded_Y = np.empty((Y.shape[0], self.number_of_outputs), dtype=np.int32)
		for i in range(self.number_of_outputs):
			encoded_Y[:, i] = np.where(Y == i, self.T, -self.T)

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
		q: float = 1.0,
		max_included_literals=None,
		boost_true_positive_feedback=1,
		number_of_state_bits=8,
		append_negated=True,
		r: float = 1.0,
		sr: float | None = None,
		encode_loc: bool = True,
		max_weight: int | None = None,
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
			r=r,
			sr=sr,
			encode_loc=encode_loc,
			max_weight=max_weight,
			grid=grid,
			block=block,
		)
		self.dim = dim
		self.patch_dim = patch_dim
		# self.negative_clauses = 1

	def fit(self, X, Y, epochs=100, incremental=False):
		if len(X.shape) == 3:
			print(f"Expecting X with 2D shape, got {X.shape}. Flattening the array...")
			X = X.reshape((X.shape[0], -1))
			print(f"New X.shape => {X.shape}")
		X = csr_matrix(X)

		self.number_of_outputs = Y.shape[1]
		self.negative_clauses = np.ones(self.number_of_outputs, dtype=np.uint32)

		self.max_y = None
		self.min_y = None

		encoded_Y = np.where(Y == 1, self.T, -self.T).astype(np.int32)

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
		q: float = 1.0,
		max_included_literals=None,
		boost_true_positive_feedback=1,
		number_of_state_bits=8,
		append_negated=True,
		r: float = 1.0,
		sr: float | None = None,
		max_weight: int | None = None,
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
			r=r,
			sr=sr,
			max_weight=max_weight,
			grid=grid,
			block=block,
		)
		# self.negative_clauses = 1

	def fit(self, X, Y, epochs=100, incremental=False):
		X = csr_matrix(X)

		self.number_of_outputs = Y.shape[1]
		self.negative_clauses = np.ones(self.number_of_outputs, dtype=np.uint32)

		self.dim = (X.shape[1], 1, 1)
		self.patch_dim = (X.shape[1], 1)

		self.max_y = None
		self.min_y = None

		encoded_Y = np.where(Y == 1, self.T, -self.T).astype(np.int32)
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
		q: float = 1.0,
		max_included_literals=None,
		boost_true_positive_feedback=1,
		number_of_state_bits=8,
		append_negated=True,
		r: float = 1.0,
		sr: float | None = None,
		max_weight: int | None = None,
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
			r=r,
			sr=sr,
			max_weight=max_weight,
			grid=grid,
			block=block,
		)
		# self.negative_clauses = 1

	def fit(self, X, Y, epochs=100, incremental=False):
		X = csr_matrix(X)

		self.number_of_outputs = int(np.max(Y) + 1)
		self.negative_clauses = np.ones(self.number_of_outputs, dtype=np.uint32)

		self.dim = (X.shape[1], 1, 1)
		self.patch_dim = (X.shape[1], 1)

		self.max_y = None
		self.min_y = None

		encoded_Y = np.empty((Y.shape[0], self.number_of_outputs), dtype=np.int32)
		for i in range(self.number_of_outputs):
			encoded_Y[:, i] = np.where(Y == i, self.T, -self.T)

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
		q: float = 1.0,
		max_included_literals=None,
		boost_true_positive_feedback=1,
		number_of_state_bits=8,
		append_negated=True,
		r: float = 1.0,
		sr: float | None = None,
		max_weight: int | None = None,
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
			r=r,
			sr=sr,
			max_weight=max_weight,
			grid=grid,
			block=block,
		)
		# self.negative_clauses = 1

	def fit(self, X, Y, epochs=100, incremental=False):
		X = X.reshape(X.shape[0], X.shape[1], 1)

		self.number_of_outputs = 1
		self.negative_clauses = np.ones(self.number_of_outputs, dtype=np.uint32)
		self.patch_dim = (X.shape[1], 1, 1)

		self.max_y = None
		self.min_y = None

		encoded_Y = np.where(Y == 1, self.T, -self.T).astype(np.int32)

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
		r: float = 1.0,
		sr: float | None = None,
		max_weight: int | None = None,
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
			r=r,
			sr=sr,
			max_weight=max_weight,
			grid=grid,
			block=block,
		)
		# self.negative_clauses = 0

	def fit(self, X, Y, epochs=100, incremental=False):
		X = X.reshape(X.shape[0], X.shape[1], 1)

		self.number_of_outputs = 1
		self.negative_clauses = np.zeros(self.number_of_outputs, dtype=np.uint32)
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


class HybridConvolutionalTsetlinMachine(CommonTsetlinMachine):
	"""
	A hybrid TM, combining the Convolutional TM with a Regression TM.
	The class_types parameter is used to specify the type of each output, if it is a classification or regression task.

	IMP: THE CLASSIFICATION LABELS MUST BE ONE-HOT ENCODED.
	"""

	def __init__(
		self,
		number_of_clauses,
		T,
		s,
		dim,
		patch_dim,
		class_types: list[Literal["C", "R", "L"]],
		q: float = 1.0,
		max_included_literals=None,
		boost_true_positive_feedback=1,
		number_of_state_bits=8,
		append_negated=True,
		r: float = 1.0,
		sr: float | None = None,
		encode_loc: bool = True,
		max_weight: int | None = None,
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
			r=r,
			sr=sr,
			max_weight=max_weight,
			encode_loc=encode_loc,
			grid=grid,
			block=block,
		)
		self.dim = dim
		self.patch_dim = patch_dim
		self.class_types = class_types

	def fit(self, X, Y, epochs=100, incremental=False):
		X = csr_matrix(X)
		if len(Y.shape) == 1:
			Y = Y.reshape((Y.shape[0], 1))

		assert len(self.class_types) == Y.shape[1], f"Number of class types ({len(self.class_types)}) does not match {Y.shape[1]=}"

		self.y_layout = []
		self.number_of_outputs = 0
		for i, ct in enumerate(self.class_types):
			if ct == "R" or ct == "L":
				# Regression or Multi-Label
				self.number_of_outputs += 1
				self.y_layout.append(f"{ct}1")
			elif ct == "C":
				# Classification
				self.number_of_outputs += int(np.max(Y[:, i])) + 1
				self.y_layout.append(f"{ct}{int(np.max(Y[:, i])) + 1}")

		self.negative_clauses = np.zeros(self.number_of_outputs, dtype=np.uint32)
		encoded_Y = np.empty((Y.shape[0], self.number_of_outputs), dtype=np.int32)

		for i, ct in enumerate(self.class_types):
			if ct == "R":
				self.max_y = np.max(Y[:, i])
				self.min_y = np.min(Y[:, i])
				encoded_Y[:, i] = ((Y[:, i] - self.min_y) / (self.max_y - self.min_y) * self.T).astype(np.int32)
				self.negative_clauses[i] = 0
			elif ct == "L":
				encoded_Y[:, i] = np.where(Y[:, i] == 1, self.T, -self.T)
				self.negative_clauses[i] = 1
			elif ct == "C":
				# Classification
				for j in range(int(np.max(Y[:, i]) + 1)):
					encoded_Y[:, i + j] = np.where(Y[:, i] == j, self.T, -self.T)
					self.negative_clauses[i + j] = 1

		self._fit(X, encoded_Y, epochs=epochs, incremental=incremental)

		return

	def score(self, X):
		X = csr_matrix(X)
		return self._score(X)

	def predict(self, X, return_class_sums=False):
		if len(X.shape) == 3:
			print(f"Expecting X with 2D shape, got {X.shape}. Flattening samples...")
			X = X.reshape((X.shape[0], -1))
			print(f"New X.shape => {X.shape}")

		class_sums = self.score(X)

		preds = np.zeros((X.shape[0], len(self.class_types)), dtype=np.float32)
		for i, ct in enumerate(self.class_types):
			if ct == "R":
				preds[:, i] = 1.0 * (class_sums[:, i]) * (self.max_y - self.min_y) / (self.T) + self.min_y
			elif ct == "L":
				preds[:, i] = (class_sums[:, i] >= 0).astype(np.uint32)
			elif ct == "C":
				n = int(self.y_layout[i][1:])
				preds[:, i] = np.argmax(class_sums[:, i : i + n], axis=1)

		preds = preds.squeeze()
		if return_class_sums:
			# NOTE: preds.shape[1] != class_sums.shape[1]
			return preds, class_sums
		else:
			return preds
