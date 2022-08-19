# Copyright (c) 2022 Mathis Gerdes
# Licensed under the MIT license (see LICENSE for details).

from __future__ import annotations

import chex
import jax
import jax.numpy as jnp
import numpy as np
from .util import cyclic_corr
from functools import partial


@partial(jax.jit, static_argnames=('average',))
def two_point(phis: jnp.ndarray, average: bool = True) -> jnp.ndarray:
    """Estimate ``G(x) = <phi(0) phi(x)>``.

    Translational invariance is assumed, so to improve the estimate we compute
        ``mean_y <phi(y) phi(x+y)>``
    using periodic boundary conditions.

    Args:
        phis: Samples of field configurations of shape
            ``(batch size, L_1, ..., L_d)``.
        average: If false, average over samples is not executed.

    Returns:
        Array of shape ``(L_1, ..., L_d)`` if ``average`` is true, otherwise
        of shape ``(batch size, L_1, ..., L_d)``.
    """
    corr = jax.vmap(cyclic_corr)(phis, phis)
    return jnp.mean(corr, axis=0) if average else corr


@jax.jit
def two_point_central(phis: jnp.ndarray) -> jnp.ndarray:
    """Estimate ``G_c(x) = <phi(0) phi(x)> - <phi(0)> <phi(x)>``.

    Translational invariance is assumed, so to improve the estimate we compute
        ``mean_y <phi(y) phi(x+y)> - <phi(x)> mean_y <phi(x+y)>``
    using periodic boundary conditions.

    Args:
        phis: Samples of field configurations of shape
            ``(batch size, L_1, ..., L_d)``.

    Returns:
        Array of shape ``(L_1, ..., L_d)``.
    """
    phis_mean = jnp.mean(phis, axis=0)
    outer = phis_mean * jnp.mean(phis_mean)

    return two_point(phis, True) - outer


@jax.jit
def correlation_length(G):
    """Estimator for the correlation length.

    Args:
        G: Centered two-point function.

    Returns:
        Scalar. Estimate of correlation length.
    """
    Gs = jnp.mean(G, axis=0)
    arg = (jnp.roll(Gs, 1) + jnp.roll(Gs, -1)) / (2 * Gs)
    mp = jnp.arccosh(arg[1:])
    return 1 / jnp.nanmean(mp)


@jax.jit
def phi4_action(phi: jnp.ndarray,
                m2: chex.Scalar = 1,
                lam: chex.Scalar = None) -> jnp.ndarray:
    """Compute the Euclidean action for the scalar phi^4 theory.

    The Lagrangian density is kin(phi) + m2 * phi + l * phi^4

    Args:
        phi: Single field configuration of shape L^d.
        m2: Mass squared term (can be negative).
        lam: Coupling constant for phi^4 term.

    Returns:
        Scalar, the action of the field configuration..
    """
    phis2 = phi ** 2

    a = m2 * phis2
    if lam is not None:
        a += lam * phis2 ** 2

    # Kinetic term
    a += sum((jnp.roll(phi, 1, d) - phi) ** 2 for d in range(phi.ndim))

    return jnp.sum(a)


@chex.dataclass
class Phi4Theory:
    """Scalar phi^4 theory."""
    shape: tuple[int, ...]
    m2: chex.Scalar
    lam: chex.Scalar = None

    @property
    def lattice_size(self):
        return np.prod(self.shape)

    @property
    def dim(self):
        return len(self.shape)

    def action(self, phis: jnp.ndarray, *,
               m2: chex.Scalar = None,
               lam: chex.Scalar = None,
               half=False) -> jnp.ndarray:
        """Compute the phi^4 action.

        Args:
            phis: Either a single field configuration of shape L^d or
                a batch of those field configurations.
            m2: Mass squared (can be negative).
            lam: Coupling constant for phi^4 term.
            half: Whether to include a 1/2 factor in the (Euclidean)
                lagrangian.

        Returns:
            Either a scalar value or a 1d array of actions for the
            field configuration(s).
        """
        lam = self.lam if lam is None else lam
        m2 = self.m2 if m2 is None else m2

        # check whether phis are a batch or a single sample
        if phis.ndim == self.dim:
            chex.assert_shape(phis, self.shape)
            action = phi4_action(phis, m2, lam)
            return action / 2 if half else action
        else:
            chex.assert_shape(phis[0], self.shape)
            act = partial(phi4_action, m2=m2, lam=lam)
            action = jax.vmap(act)(phis)
            return action / 2 if half else action
