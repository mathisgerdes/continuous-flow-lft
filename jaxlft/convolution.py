# Copyright (c) 2022 Mathis Gerdes
# This code is licensed under MIT license (see LICENSE.txt for details).

from __future__ import annotations

import haiku as hk
import jax.numpy as jnp
import numpy as np

from typing import Optional, Union, Callable
from jax import lax
from functools import partial
from collections import namedtuple
from . import conv_util


def conv_indices(shape: tuple[int, ...], return_flat=True, center=True):
    """Generate index matrix for translation invariant layer (conv).

    This can be understood as giving the 'orbit index' for each pair of
    points x, y.

    Returns:
        An array of shape ``[len(shape), *shape, *shape]`` potentially
        flattened to ``[len(shape), prod(shape), prod(shape)]`` where
       ``out[i, x, y] = (y-x)[i]``.
    """
    xs_grid = np.indices(shape)
    flat = np.reshape(xs_grid, (len(shape), np.prod(shape)))
    mods = np.array(shape).reshape((-1, 1, 1))
    if center:
        shift = np.array(shape).reshape((-1, 1, 1)) // 2
    else:
        shift = 0
    added = np.mod(flat[:, None, :] - flat[:, :, None] + shift, mods)
    full_shape = (len(shape), *shape, *shape)
    return added if return_flat else added.reshape(full_shape)


def _get_dist(shape: tuple[int, ...]):
    """Distance squared to origin of lattice sites.

    Args:
        shape: Shape of lattice.
    Returns:
        An integer array where each entry is the site's
        distance to the origin squared. The origin is
        at L//2 in each dimension of length L.
    """
    coords = [np.arange(-(s - 1) // 2, (s + 1) // 2) for s in shape]
    coords = np.meshgrid(*coords, indexing='ij')
    dist = sum(c ** 2 for c in coords)
    return dist


def _gather_orbit_indices(orbits: jnp.ndarray):
    """Translate non-contiguous orbit indices to sequential ones."""
    unique = np.unique(orbits)
    index_map = np.empty(unique[-1] + 1, dtype=int)
    index_map[unique] = np.arange(len(unique))
    orbits = index_map[orbits]
    return len(unique), orbits


def unique_index_kernel(shape: tuple[int, ...]):
    """Unique lattice indices odered by distance to origin.

    Args:
        shape: Shape of lattice.
    Returns:
        An integer array of given `shape` with unique entries ordered
        by the site's distance to the origin at L//2 (in each dimension).
    """
    dist = _get_dist(shape)
    index = np.empty(np.prod(shape), dtype=int)
    index[np.argsort(dist.flatten())] = np.arange(np.prod(shape))
    index = index.reshape(shape)
    return index


def flip_lattice(lattice: jnp.ndarray, axis: int) -> jnp.ndarray:
    """Flip array along one axis."""
    return np.roll(np.flip(lattice, axis), 1 - lattice.shape[axis] % 2, axis)


def rot_lattice_90(lattice: jnp.ndarray, ax1: int, ax2: int) -> jnp.ndarray:
    """Rotate two axes into each other by 90 degrees."""
    lattice = np.swapaxes(lattice, ax1, ax2)
    lattice = flip_lattice(lattice, ax1)
    return lattice


def gather_orbits(
        shape: tuple[int, ...],
        transformations: list[Callable[[jnp.ndarray], jnp.ndarray]]) \
        -> tuple[int, jnp.ndarray]:
    """Compute orbit indices for lattice given shape and transformations.

    One orbit is defined by a set of indices which can be transformed
    into each other by any composition of operations given by the
    list of transformations.

    Args:
        shape: Lattice shape.
        transformations: List of transformations.
            Each transformation takes an array as input and returns
            a correspondingly transformed array.
    Returns:
        Tuple (number of orbits, integer array of orbit for each site).
    """
    # start by assigning a unique 'orbit' to each index of the lattice
    lattice = unique_index_kernel(shape)
    unique_indices = lattice.flatten()

    # generate the stack of transformed lattices
    partial_orbits = np.empty(
        (len(unique_indices), len(transformations) + 1), dtype=int)
    partial_orbits[:, 0] = unique_indices
    for i, op in enumerate(transformations):
        partial_orbits[:, i + 1] = op(lattice).flatten()

    # maps orbit -> set(indices)
    orbit_members = dict()
    # maps index -> orbit
    orbit_index = np.full_like(unique_indices, -1)

    # consider all indices that appear at a given site given the
    # 'generator' transformations applied above
    for parts in partial_orbits:
        orbits = set()  # track what orbits need to be merged
        update_indices = set()  # track for which indices orbit needs updating
        for i in parts:
            if orbit_index[i] != -1:  # index was already assigned to an orbit
                orbits.add(orbit_index[i])
            else:
                update_indices.add(i)

        # let the orbit index be the minimum of the site indices it contains
        new_orbit = np.min(parts)
        if len(orbits) != 0:
            new_orbit = min(new_orbit, min(orbits))

        if new_orbit not in orbits:
            members = update_indices.copy()
            orbit_members[new_orbit] = members
        else:
            members = orbit_members[new_orbit]
            members.update(update_indices)

        # merge orbits
        for orbit in orbits:
            if orbit == new_orbit:
                continue
            new_members = orbit_members[orbit]
            members.update(new_members)
            update_indices.update(new_members)
            del orbit_members[orbit]

        for i in update_indices:
            orbit_index[i] = new_orbit

    new_orbits = orbit_index[unique_indices].reshape(lattice.shape)
    return _gather_orbit_indices(new_orbits)


def kernel_d4(shape: tuple[int, ...]) -> tuple[int, jnp.ndarray]:
    """Orbit number and indices per site for D4 symmetry.

    Args:
        shape: Lattice shape (tuple).
    Returns:
        Tuple (number of orbits, integer array of orbit for each site).
    """
    assert all(shape[0] == li for li in shape[1:]), \
        'Rotation requires all side lengths to be equal.'
    transformations = [partial(flip_lattice, axis=i)
                       for i in range(len(shape))]
    for i in range(len(shape)):
        # probably redundantly many
        for j in range(i, len(shape)):
            transformations.append(partial(rot_lattice_90, ax1=i, ax2=j))

    return gather_orbits(shape, transformations)


def kernel_equidist(shape: tuple[int, ...]) -> tuple[int, jnp.ndarray]:
    """Orbit number and indices per site identifying equidistant sites.

    Args:
        shape: Lattice shape (tuple).
    Returns:
        Tuple (number of orbits, integer array of orbit for each site).
    """
    dist = _get_dist(shape)
    return _gather_orbit_indices(dist)


def unfold_kernel(
        kernel_params: jnp.ndarray, orbits: jnp.ndarray) -> jnp.ndarray:
    """Expand the parameters of a symmetric kernel into full conv kernel.

    This function is the inverse of ``fold_kernel``.

    Args:
        kernel_params: Parameters of kernel with shape
            (orbit_count, in_channels, out_channels).
        orbits: Integer array giving the orbit index for each lattice site.

    Returns:
        The full convolutional kernel.
    """
    return kernel_params[orbits]


def pad_kernel_weights(
        kernel: jnp.ndarray,
        new_shape: Union[int, tuple[int,...]]) -> jnp.ndarray:
    """Increase the size of conv kernel by padding with zeros.

    The non-trivial part comes from dimensions with even length.
    Using circular padding, the edge value is copied when going to a
    larger length and divided by the number of copies.

    Args:
        kernel: Kernel weight array (already expanded, not the
            reduced parameters). Shape of array is
            ``(L_1, ..., L_dim, in_channels, out_channels)``.
        new_shape: Either an integer or a tuple.
            New side length of lattice. Must be larger than the
            side length L of the original kernel.

    Returns:
        Kernel with new shape.
    """
    space_shape = kernel.shape[:-2]
    if isinstance(new_shape, tuple):
        assert len(new_shape) == len(space_shape), \
            'The dimension of the new shape does not match existing kernel.'
    else:
        new_shape = (new_shape,) * len(space_shape)

    # in even dimensions, copy the 'wrap-around' indices (at the edge)
    wraps = [(0, 1) if length % 2 == 0 else (0, 0) for length in space_shape]
    kernel = np.pad(kernel, [*wraps, (0, 0), (0, 0)], 'wrap')
    _slice = np.index_exp[:]
    for dim, length in enumerate(space_shape):
        if length % 2 == 0:
            kernel[_slice * dim + ([0, -1],)] /= 2

    # add zeros to reach desired shape
    padding = [((new - old + 1) // 2, (new - old) // 2)
               for new, old in zip(new_shape, kernel.shape[:-2])]
    w = np.pad(kernel, padding + [(0, 0)] * 2, 'constant', constant_values=0.)
    return w


def fold_kernel(
        kernel_weights: jnp.ndarray,
        orbits: jnp.ndarray,
        orbit_count: int) -> jnp.ndarray:
    """Convert symmetric conv kernel into the unique independent parameters.

    This function is the inverse of ``unfold_kernel``.

    Args:
        kernel_weights: Kernel weight array of a convolution.
        orbits: Integer array giving the orbit index for each lattice site.
        orbit_count: Total number of orbits.

    Returns:
        Reduced array giving the independent parameters of the conv kernel.
    """
    in_channels, out_channels = kernel_weights.shape[-2:]
    assert orbits.shape == kernel_weights.shape[:-2]

    w_raw = np.zeros((orbit_count, in_channels, out_channels))
    count = np.zeros((orbit_count, 1, 1))

    flat_kernel = kernel_weights.reshape(-1, in_channels, out_channels)
    for index, ws in zip(orbits.flatten(), flat_kernel):
        w_raw[index] += ws
        count[index] += 1

    return w_raw / count


_equiv_conv_config = namedtuple(
    'EquivConvConfig', [
        'orbits', 'kernel_shape', 'num_spatial_dims',
        'in_space_shape', 'channel_index', 'unbatched_rank',
        'dimension_numbers'])


def init_equiv_conv(
        inputs: jnp.ndarray, *,
        num_spatial_dims: int,
        output_channels: int,
        kernel_shape: Union[int, tuple[int, ...]] = None,
        orbit_function=kernel_d4,
        w_init: Optional[hk.initializers.Initializer] = None) \
        -> tuple[jnp.ndarray, _equiv_conv_config]:
    """Initialize weights and auxiliary data of convolution.

    Args:
        inputs: Input data (batch, space, ..., space, features).
        num_spatial_dims: Number of spacial dimensions.
        output_channels: Number of output feature dimensions.
        kernel_shape: Shape of the convolution kernel (integer or tuple).
        orbit_function: Function returning the number of orbits and
            the orbit index for each kernel entry given the shape of the
            kernel. The orbit index array must match the spacial dimensions
            of the kernel. Can be None if there are no symmetries
            (besides translations).
        w_init: Initializer for the kernel weights.

    Returns:
        The initialized weight and a named tuple of auxiliary data.
    """
    channel_index = -1
    unbatched_rank = num_spatial_dims + 1
    dimension_numbers = conv_util.to_dimension_numbers(
        num_spatial_dims,
        channels_last=True,
        transpose=False)

    unbatched = inputs.ndim == unbatched_rank
    if unbatched:
        inputs = jnp.expand_dims(inputs, axis=0)

    space_idc = dimension_numbers.lhs_spec[2:]
    in_space_shape = tuple(inputs.shape[i] for i in space_idc)
    if kernel_shape is None:
        kernel_shape = in_space_shape
    else:
        kernel_shape = conv_util.replicate(
            kernel_shape, num_spatial_dims, 'kernel_shape')
    input_channels = inputs.shape[channel_index]

    if w_init is None:
        fan_in_shape = np.prod(kernel_shape) * input_channels
        stddev = 1. / np.sqrt(fan_in_shape)
        w_init = hk.initializers.TruncatedNormal(stddev=stddev)

    if orbit_function is not None:
        orbit_count, orbits = orbit_function(kernel_shape)
        w_shape = (orbit_count, input_channels, output_channels)
        w = hk.get_parameter('w', w_shape, inputs.dtype, init=w_init)
    else:
        w_shape = kernel_shape + (input_channels, output_channels)
        w = hk.get_parameter('w', w_shape, inputs.dtype, init=w_init)
        orbits = None

    config = dict(
        orbits=orbits,
        kernel_shape=kernel_shape,
        num_spatial_dims=num_spatial_dims,
        in_space_shape=in_space_shape,
        channel_index=channel_index,
        unbatched_rank=unbatched_rank,
        dimension_numbers=dimension_numbers,
    )

    return w, _equiv_conv_config(**config)


def apply_equiv_conv(
        inputs: jnp.ndarray,
        w: jnp.ndarray,
        config: _equiv_conv_config, *,
        use_conv: bool = True,
        with_bias: bool = False,
        b_init: hk.initializers.Initializer = jnp.zeros) -> jnp.ndarray:
    """Apply equivariant convolution.

    Args:
        inputs: Array of inputs.
        w: Kernel weights.
        config: Meta-data about convolution, generated by ``init_equiv_conv``.
        use_conv: Whether to use built-in convolution or expand kernel to full
            dense layer and do matrix multiplication.
        with_bias: Whether the conv-layer adds a bias to the output.
        b_init: Initializer for bias.

    Returns:
        The output of the convolution.
    """
    output_channels = w.shape[-1]
    input_channels = w.shape[-2]
    stride = conv_util.replicate([1], config.num_spatial_dims, 'strides')
    unbatched = inputs.ndim == config.unbatched_rank
    if unbatched:
        inputs = jnp.expand_dims(inputs, axis=0)

    if config.orbits is not None:
        w = w[config.orbits]
    else:
        w = w.reshape(*config.kernel_shape, *w.shape[1:])

    if use_conv:
        pads = [(0, 0)]  # no padding in batch dimension
        space_padding = [(k // 2, (k - 1) // 2) for k in config.kernel_shape]
        if config.channel_index == -1:
            pads += space_padding + [(0, 0)]
        else:
            pads += [(0, 0)] + space_padding

        inputs = jnp.pad(inputs, pads, mode='wrap')

        out = lax.conv_general_dilated(
            inputs,
            w,
            window_strides=stride,
            padding='VALID',
            dimension_numbers=config.dimension_numbers)

    elif config.in_space_shape == config.kernel_shape:
        conv_idc = conv_indices(
            config.kernel_shape, return_flat=True, center=True)
        w = w[tuple(conv_idc)]
        inputs_flat = inputs.reshape(
            (inputs.shape[0], -1, input_channels))
        out = jnp.einsum('ijco,bjc->bio', w, inputs_flat)
        out = out.reshape(inputs.shape[:-1] + (output_channels,))

    else:
        raise RuntimeError('Kernel shapes != full input shape is not '
                           'implemented for use_conv = False.')

    if with_bias:
        if config.channel_index == -1:
            bias_shape = (output_channels,)
        else:
            bias_shape = (output_channels,) + (1,) * config.num_spatial_dims
        b = hk.get_parameter('b', bias_shape, inputs.dtype, init=b_init)
        b = jnp.broadcast_to(b, out.shape)
        out = out + b

    if unbatched:
        out = jnp.squeeze(out, axis=0)

    return out


class EquivConvND(hk.Module):
    """Equivariant N-dimensional convolution."""
    def __init__(
            self,
            num_spatial_dims: int,
            output_channels: int,
            with_bias: bool = True,
            orbit_function=None,
            kernel_shape: tuple = None,
            w_init: Optional[hk.initializers.Initializer] = None,
            b_init: Optional[hk.initializers.Initializer] = None,
            use_conv: bool = True,
            name: Optional[str] = None):
        """Initializes the module.

        The kernel shape is always the full size of the input.
        Note that the channel index here is fixed to be -1
        and the stride is fixed to 1.

        Args:
            num_spatial_dims: The number of spatial dimensions of the input.
            output_channels: Number of output channels.
            orbit_function: Function (x, shape) -> yield g(x) which applies all
                transformations g to ths position array x.
            with_bias: Whether to add a bias. By default, true.
            w_init: Optional weight initialization.
                By default, truncated normal.
            b_init: Optional bias initialization. By default, zeros.
            use_conv: Whether to use lax.conv for convolution or manual
                indexing and a linear contraction. The latter may be
                faster for small lattices.
            name: The name of the module.
        """
        super().__init__(name=name)
        self.with_bias = with_bias

        self.use_conv = use_conv
        self.with_bias = with_bias
        self.b_init = b_init or jnp.zeros

        self.init = partial(
            init_equiv_conv,
            num_spatial_dims=num_spatial_dims,
            output_channels=output_channels,
            kernel_shape=kernel_shape,
            orbit_function=orbit_function,
            w_init=w_init)

    def __call__(self, inputs):
        """Connects ``EquivConvND`` layer.

        Args:
            inputs: An array of shape ``(spatial_dims, C)`` and rank-N+1 if
                unbatched, or an array of shape ``(N, spatial_dims, C)``
                and rank-N+2 if batched.

        Returns:
            An array of shape ``(spatial_dims, output_channels)`` and
            rank-N+1 if unbatched, or an array of shape
            ``(N, spatial_dims, output_channels)`` and rank-N+2 if batched.
        """
        w, config = self.init(inputs)
        return apply_equiv_conv(
            inputs, w, config,
            use_conv=self.use_conv,
            with_bias=self.with_bias,
            b_init=self.b_init)
