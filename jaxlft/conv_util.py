# Adapted from:
# https://github.com/deepmind/dm-haiku/blob/main/haiku/_src/conv.py
# and
# https://github.com/deepmind/dm-haiku/blob/main/haiku/_src/utils.py
#
# Copyright 2020 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations
import re
from jax import lax
from collections.abc import Sequence as ABCSeq


_SPATIAL_CHANNELS_FIRST = re.compile("^NC[^C]*$")
_SPATIAL_CHANNELS_LAST = re.compile("^N[^C]*C$")
_SEQUENTIAL = re.compile("^((BT)|(TB))[^D]*D$")


def replicate(element, num_times: int, name: str) -> tuple:
    """Replicates entry in `element` `num_times` if needed."""
    if isinstance(element, (str, bytes)) or not isinstance(element, ABCSeq):
        return (element,) * num_times
    elif len(element) == 1:
        return tuple(element * num_times)
    elif len(element) == num_times:
        return tuple(element)
    raise TypeError(
        "{} must be a scalar or sequence of length 1 or sequence of length {}."
            .format(name, num_times))


def to_dimension_numbers(
        num_spatial_dims: int,
        channels_last: bool,
        transpose: bool) -> lax.ConvDimensionNumbers:
    """Create a `lax.ConvDimensionNumbers` for the given inputs."""
    num_dims = num_spatial_dims + 2

    if channels_last:
        spatial_dims = tuple(range(1, num_dims - 1))
        image_dn = (0, num_dims - 1) + spatial_dims
    else:
        spatial_dims = tuple(range(2, num_dims))
        image_dn = (0, 1) + spatial_dims

    if transpose:
        kernel_dn = (num_dims - 2, num_dims - 1) + tuple(range(num_dims - 2))
    else:
        kernel_dn = (num_dims - 1, num_dims - 2) + tuple(range(num_dims - 2))

    return lax.ConvDimensionNumbers(lhs_spec=image_dn, rhs_spec=kernel_dn,
                                    out_spec=image_dn)
