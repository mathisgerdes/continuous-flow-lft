# Copyright (c) 2022 Mathis Gerdes
# Licensed under the MIT license (see LICENSE for details).

from __future__ import annotations
import jax
import jax.numpy as jnp
import haiku as hk

from collections import namedtuple
from typing import Mapping
from .util import IndependentUnitNormal


def _without_apply_rng(fn):
    def wrapped(par: Mapping[str, Mapping[str, jnp.ndarray]], *args, **kwargs):
        return fn(par, None, *args, **kwargs)
    return wrapped


def transform_flow(
        flow_generator, prior, *, apply_jit=False, init_forward_only=True):
    """Generate pure normalizing flow functions from generator function.

    See https://dm-haiku.readthedocs.io/en/latest/api.html#haiku.multi_transform

    Args:
        flow_generator: Parameterless function returning a model
            with the methods 'forward' and 'reverse'. The model must
            either be a tuple (forward, reverse) or have these two
            functions as attributes.
        prior: Prior distribution.
        init_forward_only: Only flow in forward direction during
            initialization. Admissible as long as reverse direction
            has no additional parameters that need to be initialized.
        apply_jit: If true, apply jit to all transformed pure functions
            where ``batch_size`` is a static argument.

    Returns:
        Flow object Flow{init, sample, forward, reverse} containing
            the transformed functions.
    """
    # use jax.random-style function signature again here (not tfp)
    def sample_log_prob(key, batch_size):
        x = prior.sample(sample_shape=batch_size, seed=key)
        log_prob = prior.log_prob(x)
        return x, log_prob

    def generator():
        flow = flow_generator()
        if isinstance(flow, tuple):
            forward, reverse = flow
        else:
            forward, reverse = flow.forward, flow.reverse

        def sample(batch_size, **kwargs):
            x, logr = sample_log_prob(hk.next_rng_key(), batch_size)
            x, logprob = forward(x, logprob=logr, **kwargs)
            return x, logprob

        def init(batch_size=1, **kwargs):
            x, _ = sample(batch_size, **kwargs)
            if not init_forward_only:
                x, _ = reverse(x, **kwargs)
            return x

        return init, (sample, forward, reverse)

    init, (sample, forward, reverse) = hk.multi_transform(generator)
    forward, reverse = map(_without_apply_rng, (forward, reverse))

    if apply_jit:
        forward = jax.jit(forward)
        reverse = jax.jit(reverse)
        sample_log_prob = jax.jit(
            sample_log_prob, static_argnums=(1,),
            static_argnames=('batch_size',))
        sample = jax.jit(
            sample, static_argnums=(2,),
            static_argnames=('batch_size',))

    flow_type = namedtuple('Flow',
                           ['init', 'prior', 'sample', 'forward', 'reverse'])
    return flow_type(init, sample_log_prob, sample, forward, reverse)


class AbstractFlow:
    """Base class for normalizing flows."""
    def __call__(self):
        """Generator of forward and reverse functions to be transformed.

        This function is later passed to ``hk.multi_transform`` via
        the method ``flow.transform_flow``.

        Returns:
            Two functions ``forward(sample, logprob=None, **kwargs)`` and
            ``reverse(sample, logprob=None, **kwargs)``.
        """
        raise NotImplementedError

    def transform(self, prior=None, lattice_shape=None,
                  apply_jit=False, init_forward_only=True):
        """Transform the flow into pure functions.

        Signatures of generated functions:
            - prior: ``(key, batch_size) -> (samples, log prob)``
            - forward: ``(params, samples, logprob=None, **kwargs) -> (samples, log prob)``
            - reverse: ``(params, samples, logprob=None, **kwargs) -> (samples, log prob)``
            - sample: ``(params, key, batch_size, **kwargs) -> (samples, log prob)``

        Args:
            prior: Prior distribution. Default is ``IndependentUnitNormal``.
            lattice_shape: Shape of lattice, defines shape of samples.
                Only used if prior is not given to initialize the default.
            init_forward_only: Only flow in forward direction during
                initialization. Admissible as long as reverse direction
                has no additional parameters that need to be initialized.
            apply_jit: If true, apply jit to all transformed pure functions
                where ``batch_size`` is a static argument.

        Returns:
            A named tuple ``Flow`` with attributes init, prior, sample,
            forward, and reverse.
        """
        if prior is None:
            prior = IndependentUnitNormal(lattice_shape)

        return transform_flow(self, prior,
                              apply_jit=apply_jit,
                              init_forward_only=init_forward_only)
