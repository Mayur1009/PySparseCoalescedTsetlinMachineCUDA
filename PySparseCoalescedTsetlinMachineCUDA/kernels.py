# Copyright (c) 2021 Ole-Christoffer Granmo

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

import pathlib

current_dir = pathlib.Path(__file__).parent

def get_kernel(file):
    path = current_dir.joinpath(file)
    with path.open("r") as f:
        ker = f.read()
    return ker


code_header = """
	#include <curand_kernel.h>
	
	#define INT_SIZE 32

	#define LA_CHUNKS (((FEATURES-1)/INT_SIZE + 1))
	#define CLAUSE_CHUNKS ((CLAUSES-1)/INT_SIZE + 1)

	#if (FEATURES % 32 != 0)
	#define FILTER (~(0xffffffff << (FEATURES % INT_SIZE)))
	#else
	#define FILTER 0xffffffff
	#endif

	#define PATCH_CHUNKS (((PATCHES-1)/INT_SIZE + 1))

	#if (PATCH_CHUNKS % 32 != 0)
	#define PATCH_FILTER (~(0xffffffff << (PATCHES % INT_SIZE)))
	#else
	#define PATCH_FILTER 0xffffffff
	#endif
"""

code_update = get_kernel("cuda/code_update.cu")
code_evaluate = get_kernel("cuda/code_evaluate.cu")
code_prepare = get_kernel("cuda/code_prepare.cu")
code_encode = get_kernel("cuda/code_encode.cu")
code_transform = get_kernel("cuda/code_transform.cu")
