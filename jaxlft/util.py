# Copyright (c) 2022 Mathis Gerdes
# Licensed under the MIT license (see LICENSE for details).

from __future__ import annotations

import jax
import jax.numpy as jnp
import chex
import numpy as np
import haiku as hk

from functools import partial
from typing import Callable, Optional


@jax.jit
def cyclic_corr(arr1: jnp.ndarray, arr2: jnp.ndarray) -> jnp.ndarray:
    """Compute ``out[x] = 1/N sum_y arr1[y] arr2[y+x]``.

    x and y are d-dimensional (lattice) indices. The shapes of arr1
    and arr2 must match.
    The sum is executed with periodic boundary conditions.

    Args:
        arr1: d-dimensional array.
        arr2: d-dimensional array.

    Returns:
        d-dimensional array.
    """
    chex.assert_equal_shape((arr1, arr2))
    dim = arr1.ndim
    shape = arr1.shape

    def _compute_shift(shifted, _, axis, child):
        # first compute out value then shift to next shifted configuration
        _, sub_matrix = child(shifted)
        shifted = jnp.roll(shifted, -1, axis)
        return shifted, sub_matrix

    def _scan_component(axis, child, size):
        body = partial(_compute_shift, axis=axis, child=child)
        return lambda init: jax.lax.scan(body, init, None, size)

    def _base(shifted):
        return None, jnp.mean(arr1 * shifted)

    fn = _base
    for axis in range(dim - 1, -1, -1):
        fn = _scan_component(axis, fn, shape[axis])

    _, c = fn(arr2)
    return c


@jax.jit
def cyclic_tensor(arr1: jnp.ndarray, arr2: jnp.ndarray) -> jnp.ndarray:
    """Compute ``out[x, y] = arr1[y] arr2[y+x]``.

    x and y are d-dimensional (lattice) indices. The shapes of arr1
    and arr2 must match.
    The sum is executed with periodic boundary conditions.

    Args:
        arr1: d-dimensional array.
        arr2: d-dimensional array.

    Returns:
        2*d-dimensional array."""
    chex.assert_equal_shape((arr1, arr2))
    dim = arr1.ndim
    shape = arr1.shape

    def _compute_shift(shifted, _, axis, child):
        # first compute out value then shift to next shifted configuration
        _, sub_matrix = child(shifted)
        shifted = jnp.roll(shifted, -1, axis)
        return shifted, sub_matrix

    def _scan_component(axis, child, size):
        body = partial(_compute_shift, axis=axis, child=child)
        return lambda init: jax.lax.scan(body, init, None, size)

    def _base(shifted):
        return None, arr1 * shifted

    fn = _base
    for axis in range(dim - 1, -1, -1):
        fn = _scan_component(axis, fn, shape[axis])

    _, c = fn(arr2)
    return c


@partial(jax.jit)
def cyclic_corr_mat(arr: jnp.ndarray) -> jnp.ndarray:
    """Compute ``out[x] = 1/N sum_y arr[x,x+y]``.

    x and y are d-dimensional (lattice) indices.
    `arr` is a 2*d dimensional array.
    The sum is executed with periodic boundary conditions.

    This function is related to `cyclic_tensor` and `cyclic_corr`:
        >>> a, b = jnp.ones((2, 12, 12))
        >>> c1 = cyclic_corr(a, b)
        >>> c2 = jnp.mean(cyclic_tensor(a, b), 0)
        >>> jnp.all(c1 == c2).item()
        True
        >>> outer_product = jnp.einsum('ij,kl->ijkl', a, b)
        >>> c3 = cyclic_corr_mat(outer_product)
        >>> jnp.all(c2 == c3).item()
        True

    Args:
        arr: 2*d-dimensional array. x is the index of the first d
            dimensions, y is the index of the last d dimensions.

    Returns:
        d-dimensional array.
    """
    dim = arr.ndim // 2
    shape = arr.shape[:dim]
    assert shape == arr.shape[dim:], 'Invalid outer_product shape.'
    lattice_size = np.prod(shape)
    arr = arr.reshape((lattice_size,) * 2)

    def _compute_shift(shifted, _, axis, child):
        # first compute out value then shift to next shifted configuration
        _, sub_matrix = child(shifted)
        shifted = jnp.roll(shifted, -1, axis)
        return shifted, sub_matrix

    def _scan_component(axis, child, size):
        body = partial(_compute_shift, axis=axis, child=child)
        return lambda init: jax.lax.scan(body, init, None, size)

    def _base(shifted):
        return None, jnp.trace(arr[:, shifted.flatten()])

    fn = _base
    for axis in range(dim - 1, -1, -1):
        fn = _scan_component(axis, fn, shape[axis])

    idx = jnp.arange(lattice_size).reshape(shape)
    _, c = fn(idx)
    return c.reshape(shape) / lattice_size


@jax.jit
def reverse_dkl(logp: jnp.ndarray, logq: jnp.ndarray) -> jnp.ndarray:
    """Reverse KL divergence.

    The two likelihood arrays must be evaluated for the same set of samples.
    This function then approximates ``int_x q(x) log(q(x)/p(x)) dx``.

    If the samples ``x`` are distributed according to ``p(x)``, this
    is the reverse KL divergence.
    If the samples were taken from p(x), the returned value is the negative
    forward KL divergence.

    Args:
        logp: The log likelihood of p (up to a constant shift).
        logq: The log likelihood of q (up to a constant shift).

    Returns:
        Scalar representing the estimated reverse KL divergence.
    """
    return jnp.mean(logq - logp)


@jax.jit
def effective_sample_size(logp: jnp.ndarray, logq: jnp.ndarray) -> jnp.ndarray:
    """Compute the ESS given log likelihoods.

    The two likelihood arrays must be evaluated for the same set of samples.
    The samples are assumed to be sampled from ``p``, such that ``logp``
    is are the corresponding log-likelihoods.

    Args:
        logp: The log likelihood of p (up to a constant shift).
        logq: The log likelihood of q (up to a constant shift).

    Returns:
        The effective sample size per sample (between 0 and 1).
    """
    logw = logp - logq
    log_ess = 2*jax.nn.logsumexp(logw, axis=0) - jax.nn.logsumexp(2*logw, axis=0)
    ess_per_sample = jnp.exp(log_ess) / len(logw)
    return ess_per_sample


def moving_average(x: jnp.ndarray, window: int = 10):
    """Moving average over 1d array."""
    if len(x) < window:
        return jnp.mean(x, keepdims=True)
    else:
        return jnp.convolve(x, jnp.ones(window), 'valid') / window


class IndependentUnitNormal:
    def __init__(self, shape: tuple[int, ...]):
        """Element-wise independent unit normal distribution.

        This class is meant to be compatible with the ones
        in ``tensorflow_probability.distributions``.

        Args:
            shape: Shape of base space.
        """
        self.shape = shape

    def sample(self,
               sample_shape: tuple[int, ...] = (),
               seed: chex.PRNGKey = None) -> jnp.ndarray:
        """Generate a random independent unit normal sample.

        Args:
            sample_shape: Batch shape of the sample.
            seed: Random seed.

        Returns:
            An array of shape ``(*sample_shape, *self.shape)``.
        """
        if isinstance(sample_shape, int):
            sample_shape = (sample_shape,)
        return jax.random.normal(seed, sample_shape + self.shape)

    def log_prob(self, value: jnp.ndarray) -> jnp.ndarray:
        """Evaluate the log likelihood.

        Args:
            value: A sample from the present distribution.
                The last dimensions must match the shape of the base space.

        Returns:
            The log likelihood of the samples.
        """
        logp = jax.scipy.stats.norm.logpdf(value)
        logp = jnp.sum(logp, axis=tuple(range(-1, -len(self.shape)-1, -1)))
        return logp


def mcmc_chain(
        key: chex.PRNGKey,
        params: hk.Params,
        sample: Callable[[hk.Params, chex.PRNGKey, int], jnp.ndarray],
        action: Callable[[jnp.ndarray], jnp.ndarray],
        batch_size: int,
        sample_count: int,
        initial: Optional[tuple[jnp.ndarray, jnp.ndarray]] = None) \
        -> tuple[jnp.ndarray, float, jnp.ndarray]:
    """Run MCMC chain with optional initial sample._

    Args:
        key: Random key.
        params: Parameters of the sample.
        sample: Function (params, key, batch_size) -> sample,
            where sample is an array of field samples with shape
            (batch_size, L_1, ..., L_d).
        action: A function (field samples) -> action giving the action
            for each of the field samples such that log(p) = -action + const.
        batch_size: Number of field configurations to sample as a batch
            each time new samples are needed. These generated samples are
            then used sequentially as proposals.
        sample_count: Total number of accepted samples to generate.
        initial: Optional initial sample. Tuple of a (single) sample and
            corresponding action.
    Returns:
        Tuple of samples, acceptance rate, last sample.
        The samples are an array of shape (sample_count, L_1, ..., L2).
        The last sample is of the same kind as the `initial` parameter.
    """
    def index_batch(i_batch_key):
        # index into batch & increment index
        i, batch, key = i_batch_key
        new = (batch[0][i], batch[1][i], batch[2][i])
        return i + 1, new, batch, key

    def new_batch(i_batch_key):
        # generate new batch & reset index
        i, batch, key = i_batch_key
        k0, key = jax.random.split(key)
        x, logq = sample(params, k0, batch_size)
        logp = -action(x)
        return index_batch((0, (x, logq, logp), key))

    def mcmc(state, _):
        i, accepted, last, batch, key = state
        i, new, batch, key = jax.lax.cond(
            i < batch_size,
            index_batch,
            new_batch,
            (i, batch, key))

        last_x, last_logq, last_logp = last
        new_x, new_logq, new_logp = new
        k, key = jax.random.split(key)
        rand = jax.random.uniform(k, ())
        p_accept = jnp.exp((new_logp - new_logq) - (last_logp - last_logq))
        new, accepted = jax.lax.cond(
            rand < p_accept,
            # accept
            lambda inp: (inp[1], inp[2] + 1),
            # reject
            lambda inp: (inp[0], inp[2]),
            (last, new, accepted))

        return (i, accepted, new, batch, key), new[0]

    # initial state
    k0, key = jax.random.split(key)
    x, logq = sample(params, k0, batch_size)
    if initial is not None:
        x = x.at[0].set(initial[0])
        logq = logq.at[0].set(initial[1])
    logp = -action(x)

    # state: next batch index, accepted, last, batch, random key
    init = (1, 0, (x[0], logq[0], logp[0]), (x, logq, logp), key)

    state, chain = jax.lax.scan(mcmc, init, None, length=sample_count)
    _, accepted, last, *_ = state
    return chain, accepted / sample_count, last[:-1]
