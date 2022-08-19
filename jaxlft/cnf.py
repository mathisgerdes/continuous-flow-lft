# Copyright (c) 2022 Mathis Gerdes
# Licensed under the MIT license (see LICENSE for details).
"""Continuous normalizing flow architecture for phi^4."""

from __future__ import annotations
import numpy as np
import jax
import jax.numpy as jnp
import haiku as hk

from functools import partial
from typing import Callable, Type, Optional, Union
from jax.experimental.ode import odeint
import chex
from .ode import odeint_rk4
from .flow import AbstractFlow
from . import convolution


class KernelGauss(hk.Module):
    """Smooth interpolation based on Gaussians."""
    def __init__(
            self,
            n_kernel: int,
            minmax: Optional[tuple[chex.Scalar, chex.Scalar]] = None,
            width_factor: chex.Numeric = np.log(np.exp(1)-1),
            adaptive_width: bool = True,
            norm: bool = True,
            one_width: bool = True,
            name: str = None):
        """Initialize a Gaussian-based kernel function.

        Given a value x, the output of the kernel function is an array
        roughly like ``[exp(-(x-s1)^2), exp(-(x-s2)^2), ...]``.
        This can be understood as a smooth approximation to linear
        interpolation based on Gaussians located and positions s1, s2, etc.
        These positions are evenly spaced and fixed, here.

        Args:
            n_kernel: Number of positions/Gaussians in kernel.
            minmax: Range of input values to normalize to.
            width_factor: Initial width factor of the Gaussians.
                The smaller the factor, the wider the Gaussians.
            adaptive_width: Whether to make the width trainable.
            norm: Whether to keep the sum of the kernel values fixed to 1
                for each input value.
            one_width: Whether the widths, if trainable, can be different
                for each kernel position.
            name: Name of module.
        """
        super().__init__(name=name)
        self.minmax = minmax
        self.norm = norm

        width_shape = () if one_width else (n_kernel,)

        if adaptive_width:
            factor = hk.get_parameter(
                'width_factor', width_shape,
                init=hk.initializers.Constant(width_factor))
        else:
            factor = width_factor

        factor = jax.nn.softplus(factor)
        self.inverse_width = factor * (n_kernel - 1)
        # could also make this adaptive
        self.pos = jnp.linspace(0, 1, n_kernel)

    def __call__(self, val):
        minmax = self.minmax
        if minmax is not None:
            val = (val - minmax[0]) / (minmax[1] - minmax[0])
        val = - (val - self.pos)**2 * self.inverse_width
        out = jnp.exp(val)
        return out / jnp.sum(out) if self.norm else out


class KernelLin(hk.Module):
    """Linear interpolation kernel."""
    def __init__(
            self,
            n_kernel: int,
            minmax: Optional[tuple[chex.Scalar, chex.Scalar]] = None,
            name: str = None):
        """Initialize a linear interpolation kernel.

        The output of the model is an array like ``[a1, a2, ...]``
        where either one or two neighboring entries are non-zero.
        The position of the non-zero entry is given by the input value
        with linear interpolation (and thus two entries non-zero)
        if the input value falls between two array indices.
        The value the first and last indices correspond to are either
        0 and 1 or given by the ``minmax`` argument.

        Args:
            n_kernel: Number of elements in linear interpolation.
            minmax: Range of input values.
            name: Module name.
        """
        super().__init__(name=name)

        self.width = 1 / (n_kernel - 1)
        self.pos = np.linspace(0, 1, n_kernel)
        self.minmax = minmax

    def __call__(self, val):
        minmax = self.minmax
        if minmax is not None:
            val = (val - minmax[0]) / (minmax[1] - minmax[0])
        val = 1 - jnp.abs(val - self.pos) / self.width
        return jnp.maximum(val, 0)


class KernelFourier:
    """Truncated fourier expansion on given interval."""
    def __init__(
            self,
            n_kernel: int,
            minmax: Optional[tuple[chex.Scalar, chex.Scalar]] = None):
        """Initialize fourier basis kernel.

        Given an input x to the model, the output is an array like
            ``[1, sin(2 pi x), cos(2 pi x), sin(4 pi x), ...]``
        (except in a different order).

        Args:
            n_kernel: The number of Fourier-terms.
                (This is true if the number is odd. Otherwise, in effect,
                the next smallest odd number is used).
            minmax: The range of input values such that the largest input
                is normalized to 1.
        """
        self.freq = jnp.arange(1, (n_kernel - 1) // 2 + 1)
        self.minmax = minmax

    def __call__(self, val):
        minmax = self.minmax
        if minmax is not None:
            val = (val - minmax[0]) / (minmax[1] - minmax[0])

        sin = jnp.sin(2 * jnp.pi * self.freq * val)
        cos = jnp.cos(2 * jnp.pi * self.freq * val)
        return jnp.concatenate((sin, cos, jnp.array([1.])))


class NODEFlowBase(AbstractFlow):
    def __init__(self, *, int_steps: Optional[int] = 50):
        self.int_steps = int_steps

    def vector_field(self, state, t, **kwargs):
        """Given state and time, compute the gradient of the state.

        Args:
            state: Tuple of field values and log probabilities.
                The log probabilities are a one dimensional array with
                length matching the first (batch) dimension of the field
                configurations array.
            t: Scalar giving the current integration time.
            **kwargs: Additional arguments of the ODE.

        Returns:
            A tuple of the field value gradients and the gradients
            of the log probabilities.
        """
        # phis, logprob = state
        # return grad_phi, grad_logprob
        raise NotImplementedError

    def vector_field_reverse(self, phis_logprob, t, **kwargs):
        grad_phi, grad_logprob = self.vector_field(
            phis_logprob, 1 - t, **kwargs)
        return -grad_phi, -grad_logprob

    def __call__(self):
        def forward(phis, logprob=None, **kwargs):
            """Flow field configurations ``phis`` forward."""
            field = partial(self.vector_field, **kwargs)
            if logprob is None:
                logprob = jnp.zeros(len(phis))
            start = (phis, logprob)

            if hk.running_init():
                # must not call odeint within hk.transform
                return field(start, 0.)
            elif self.int_steps is None:
                out = odeint(field, start, jnp.array([0., 1.]))
                (_, phis), (_, logprob) = out
            else:
                phis, logprob = odeint_rk4(
                    field, start, 1.0, step_size=1/self.int_steps)
            return phis, logprob

        def reverse(phis, logprob=None, **kwargs):
            """Flow field configurations ``phis`` backward."""
            field = partial(self.vector_field_reverse, **kwargs)
            if logprob is None:
                logprob = jnp.zeros(len(phis))
            start = (phis, logprob)
            if hk.running_init():
                # must not call odeint within hk.transform
                return field(start, 0.)
            elif self.int_steps is None:
                out = odeint(field, start, jnp.array([0., 1.]))
                (_, phis), (_, logprob) = out
            else:
                phis, logprob = odeint_rk4(
                    field, start, 1.0, step_size=1 / self.int_steps)
            return phis, logprob

        return forward, reverse


class Phi4CNF(NODEFlowBase):
    def __init__(
            self, *,
            # time kernel
            n_time_kernel: int = 21,
            n_time_kernel_bond: Optional[int] = 20,
            time_kernel_base: Type[hk.Module] = KernelFourier,
            # field-features
            n_phi_freq: int = 50,
            n_phi_freq_bond: Optional[int] = 20,
            add_phi_linear: bool = True,
            phi_freq_init: hk.initializers.Initializer = hk.initializers.RandomUniform(0, 5),
            # convolution
            use_conv: bool = True,
            kernel_shape: Optional[tuple[int, ...]] = None,
            symmetry: Optional[Callable[[tuple[int, ...]], tuple[int, np.ndarray]]] = convolution.kernel_d4,
            # integrator
            int_steps: Optional[int] = 50):
        """Continuous normalizing flow with ODE based on feature kernels.

        The ODE is constructed by tensor contraction & convolution of feature
        vectors/matrices generated from the time and field values.

        Args:
            n_time_kernel: Number of time kernels.
            n_time_kernel_bond: If not ``None``, contract the time kernel
                features with a dense matrix of shape
                (n_time_kernel, n_time_kernel_bond). This may improve
                training dynamics and, if the number of features is reduced,
                may decrease computational cost.
            time_kernel_base: Type of kernel functions to use for the time
                features. See ``KernelLin`` for an example.
            n_phi_freq: Number of phi frequencies used to compute the site-wise
                field features (sine of field values times frequency
                for each site). This including the linear term,
                if ``add_phi_linear`` is true.
            n_phi_freq_bond: If not ``None``, contract the site-wise kernel
                features with a dense matrix of shape
                (n_phi_freq, n_phi_freq_bond). This may improve
                training dynamics and, if the number of features is reduced,
                may decrease computational cost.
            add_phi_linear: If true, add a linear term to the site-wise
                field value features.
            phi_freq_init: Initializer used for the field-feature frequencies.
            use_conv: Whether to use built-in convolution or a manual
                expansion to a dense operation for computing the conv step.
            kernel_shape: Spacial shape of the convolutional kernel. If None,
                the most general case is used where the kernel shape is exactly
                the lattice shape.
            symmetry: Function that generates the number of orbits and
                the orbit index array. If None, no symmetries (besides
                translation) are enforced.
            int_steps: Number of integration steps. If None, use a dynamic
                step size integrator.
        """
        super().__init__(int_steps=int_steps)

        self.n_time_kernel = n_time_kernel
        self.n_time_kernel_bond = n_time_kernel_bond
        self.time_kernel_base = time_kernel_base
        self.decompose_time_kernel = n_time_kernel_bond is not None

        self.n_phi_freq = n_phi_freq
        self.n_phi_freq_bond = n_phi_freq_bond
        self.add_phi_linear = add_phi_linear
        self.phi_freq_init = phi_freq_init
        self.decompose_phi_freq = n_phi_freq_bond is not None

        self.use_conv = use_conv
        self.kernel_shape = kernel_shape
        self.symmetry = symmetry

    def vector_field(self, state, t):
        phis, _ = state

        time_kernel = self.time_kernel_base(self.n_time_kernel)
        interp_time = time_kernel(t)
        if self.decompose_time_kernel:
            time_superpos = hk.get_parameter(
                'time_superpos',
                (self.n_time_kernel_bond, self.n_time_kernel),
                float, hk.initializers.Orthogonal()) / self.n_time_kernel
            interp_time = time_superpos @ interp_time

        phi_freq = hk.get_parameter(
            'phi_freq', (self.n_phi_freq - self.add_phi_linear,),
            float,  self.phi_freq_init)
        phi_lin = jnp.expand_dims(phis, -1)
        phi_wf = phi_lin * phi_freq
        inputs = jnp.sin(phi_wf)
        if self.add_phi_linear:
            inputs = jnp.concatenate((inputs, phi_lin), axis=-1)
        if self.decompose_phi_freq:
            freq_superpos = hk.get_parameter(
                'freq_superpos',
                (self.n_phi_freq_bond, self.n_phi_freq),
                float, hk.initializers.Orthogonal()) / self.n_phi_freq
            inputs = jnp.einsum('fw,...w->...f', freq_superpos, inputs)

        dim = len(phis.shape) - 1  # assume one batch dimension here
        # kernel_params = orbit, in channels, out channels
        w_params, conv_config = convolution.init_equiv_conv(
            inputs,
            kernel_shape=self.kernel_shape,
            num_spatial_dims=dim,
            output_channels=(self.n_time_kernel_bond
                             if self.decompose_time_kernel
                             else self.n_time_kernel),
            orbit_function=self.symmetry,
            w_init=hk.initializers.Constant(0.))

        # contract with time kernels before convolution
        w_params = w_params @ interp_time
        w00 = w_params[0]  # shape = (n_phi_freq,)
        w_params = jnp.expand_dims(w_params, -1)

        activ = convolution.apply_equiv_conv(
            inputs, w_params, conv_config,
            use_conv=self.use_conv, with_bias=False)
        grad_phi = jnp.squeeze(activ, -1)  # now: shape = lattice shape

        # Compute the gradient of log(p) given by the divergence of grad_phi.
        # Need to distinguish different architecture cases...
        inputs_div = jnp.cos(phi_wf) * phi_freq
        if self.decompose_phi_freq:
            if self.add_phi_linear:
                divergence = jnp.einsum(
                    'w,wf,b...f->b', w00, freq_superpos[:, :-1], inputs_div)
                divergence += \
                    jnp.einsum('w,w', w00, freq_superpos[:, -1]) \
                    * np.prod(inputs_div.shape[1:-1])
            else:
                divergence = jnp.einsum(
                    'w,wf,b...f->b', w00, freq_superpos, inputs_div)
        elif self.add_phi_linear:
            divergence = jnp.einsum('w,b...w->b', w00[:-1], inputs_div)
            divergence += w00[-1] * np.prod(inputs_div.shape[1:-1])
        else:
            divergence = jnp.einsum('w,b...w->b', w00, inputs_div)

        return grad_phi, -divergence


def _batch_orth_init(size, dtype):
    init = hk.initializers.Orthogonal()

    if len(size) != 3:
        return init(size, dtype)
    vis, hidden, batch = size
    return jnp.stack([init((vis, hidden), dtype) for _ in range(batch)], axis=-1)


class Phi4CNFConditional(NODEFlowBase):
    def __init__(
            self, *,
            # lambda interpolation
            n_lam_kernel: int = 50,
            lam_kernel_base: Type[hk.Module],
            lam_interp_time: bool = True,
            lam_interp_phi: bool = True,
            # time kernel
            n_time_kernel: int = 21,
            n_time_kernel_bond: int = 20,
            time_kernel_base: Type[hk.Module] = KernelFourier,
            # field-features
            n_phi_freq: int = 300,
            n_phi_freq_bond: int = 20,
            add_phi_linear: int = True,
            phi_freq_init: hk.initializers.Initializer = hk.initializers.RandomUniform(0, 5),
            # convolution
            use_conv: bool = True,
            kernel_shape: Optional[tuple[int, ...]] = None,
            symmetry: Optional[Callable[[tuple[int, ...]], tuple[int, np.ndarray]]] = convolution.kernel_d4,
            # integrator
            int_steps: Optional[int] = 50):
        """Theory-conditional CNF with ODE based on feature kernels.

        The ODE is constructed by tensor contraction & convolution of feature
        vectors/matrices generated from the time and field values.
        The lambda dependency is achieved by interpolating based on the
        ``lam_kernel_base``, where the matrices that are interpolated
        can be controlled.

        If the lambda values passed to forward, reverse, or sample is
        an array of length N, the generated phi samples may have a general
        shape (B*N, *lattice_shape). This should be interpreted as B samples
        for each of the N values of lambda (note that lambda may also be a
        single scalar value). To recover the correct batch structure,
        the samples must be reshaped to (B, N, *lattice_shape).


        Args:
            n_lam_kernel: Number of lambda interpolation kernels, i.e.
                number of values to interpolate between based on lambda.
            lam_kernel_base: Type of (interpolation) kernel to use to introduce
                lambda dependency.
            lam_interp_time: Whether to add a lambda-interpolation index to the
                time interpolation matrix.
            lam_interp_phi: Whether to add a lambda-interpolation index to the
                field-feature interpolation matrix.
            n_time_kernel: Number of time kernels.
            n_time_kernel_bond: Contract the time kernel
                features with a dense matrix of shape
                (n_time_kernel, n_time_kernel_bond).
            time_kernel_base: Type of kernel functions to use for the time
                features. See ``KernelLin`` for an example.
            n_phi_freq: Number of phi frequencies used to compute the site-wise
                field features (sine of field values times frequency
                for each site). This including the linear term,
                if ``add_phi_linear`` is true.
            n_phi_freq_bond: Contract the site-wise kernel
                features with a dense matrix of shape
                (n_phi_freq, n_phi_freq_bond).
            add_phi_linear: If true, add a linear term to the site-wise
                field value features.
            phi_freq_init: Initializer used for the field-feature frequencies.
            use_conv: Whether to use built-in convolution or a manual
                expansion to a dense operation for computing the conv step.
            kernel_shape: Spacial shape of the convolutional kernel. If None,
                the most general case is used where the kernel shape is exactly
                the lattice shape.
            symmetry: Function that generates the number of orbits and
                the orbit index array. If None, no symmetries (besides
                translation) are enforced.
            int_steps: Number of integration steps. If None, use a dynamic
                step size integrator.
        """
        super().__init__(int_steps=int_steps)

        self.n_lam_kernel = n_lam_kernel
        self.lam_kernel_base = lam_kernel_base
        self.lam_interp_time = lam_interp_time
        self.lam_interp_phi = lam_interp_phi

        self.n_time_kernel = n_time_kernel
        self.n_time_kernel_bond = n_time_kernel_bond
        self.time_kernel_base = time_kernel_base
        self.decompose_time_kernel = n_time_kernel_bond is not None

        self.n_phi_freq = n_phi_freq
        self.n_phi_freq_bond = n_phi_freq_bond
        self.add_phi_linear = add_phi_linear
        self.phi_freq_init = phi_freq_init
        self.decompose_phi_freq = n_phi_freq_bond is not None

        self.use_conv = use_conv
        self.kernel_shape = kernel_shape
        self.symmetry = symmetry

        self.vector_field_batched = hk.vmap(
            self.vector_field_single, (0, None, 0), split_rng=False)

    def vector_field_single(self, phis, t, lam):
        """Vector field where lam is a scalar."""
        n_phi_freq = self.n_phi_freq - self.add_phi_linear

        lam_kernel = self.lam_kernel_base(self.n_lam_kernel)
        interp_lam = lam_kernel(lam)

        # get time kernel features
        time_shape = (self.n_time_kernel_bond, self.n_time_kernel)
        if self.lam_interp_time:
            time_shape += (self.n_lam_kernel,)
        time_superpos = hk.get_parameter(
            'time_superpos', time_shape,
            float, _batch_orth_init) / self.n_time_kernel
        if self.lam_interp_time:
            time_superpos = time_superpos @ interp_lam

        time_kernel = self.time_kernel_base(self.n_time_kernel)
        interp_time = time_superpos @ time_kernel(t)

        # get field value-feature superposition matrix
        freq_shape = (self.n_phi_freq_bond, self.n_phi_freq)
        if self.lam_interp_phi:
            freq_shape += (self.n_lam_kernel,)
        freq_superpos = hk.get_parameter(
            'freq_superpos', freq_shape,
            float, _batch_orth_init) / self.n_phi_freq
        if self.lam_interp_phi:
            freq_superpos = freq_superpos @ interp_lam

        phi_freq = hk.get_parameter(
            'phi_freq', (n_phi_freq,), float, self.phi_freq_init)

        phi_lin = jnp.expand_dims(phis, -1)
        phi_wf = phi_lin * phi_freq
        inputs = jnp.sin(phi_wf)
        if self.add_phi_linear:
            inputs = jnp.concatenate((inputs, phi_lin), axis=-1)
        inputs = jnp.einsum('fw,...w->...f', freq_superpos, inputs)

        dim = len(phis.shape) - 1  # assume one batch dimension here
        # kernel_params = orbit, in channels, out channels
        w_params, conv_config = convolution.init_equiv_conv(
            inputs,
            kernel_shape=self.kernel_shape,
            num_spatial_dims=dim,
            output_channels=self.n_time_kernel_bond * self.n_lam_kernel,
            orbit_function=self.symmetry,
            w_init=hk.initializers.Constant(0.))

        w_params = w_params.reshape(
            *w_params.shape[:-1], self.n_time_kernel_bond, self.n_lam_kernel)

        # contract with time kernels before convolution
        w_params = jnp.einsum('...tl,t,l->...', w_params, interp_time, interp_lam)
        w00 = w_params[0]  # shape = (n_phi_freq,)
        w_params = jnp.expand_dims(w_params, -1)

        activ = convolution.apply_equiv_conv(
            inputs, w_params, conv_config,
            use_conv=self.use_conv, with_bias=False)
        grad_phi = jnp.squeeze(activ, -1)  # now: shape = lattice shape

        inputs_div = jnp.cos(phi_wf) * phi_freq
        if self.add_phi_linear:
            divergence = jnp.einsum(
                'w,wf,b...f->b', w00, freq_superpos[:, :-1], inputs_div)
            divergence += \
                jnp.einsum('w,w', w00, freq_superpos[:, -1]) \
                * np.prod(inputs_div.shape[1:-1])
        else:
            divergence = jnp.einsum(
                'w,wf,b...f->b', w00, freq_superpos, inputs_div)

        return grad_phi, -divergence

    def vector_field(self, state, t, lam):
        lam = jnp.asarray(lam)
        phis, _ = state

        if lam.ndim == 0:
            return self.vector_field_single(phis, t, lam)
        elif lam.ndim > 1:
            raise RuntimeError('The lam-array must be rank 0 or 1')

        phis = phis.reshape(len(lam), -1, *phis.shape[1:])

        grad_phi, grad_logprob = self.vector_field_batched(phis, t, lam)
        return grad_phi.reshape(phis.shape), grad_logprob.flatten()


def scale_params(
        params: hk.Params,
        new_size: Union[int, tuple[int, ...]],
        orbits: jnp.ndarray) -> hk.Params:
    """Increase the kernel size of ``Phi4CNFConditional`` or ``Phi4CNF``."""
    params = hk.data_structures.to_mutable_dict(params)
    w_full = convolution.unfold_kernel(params['~']['w'], orbits)
    w_full = convolution.pad_kernel_weights(w_full, new_size)
    w = convolution.fold_kernel(w_full, orbits, len(params['~']['w']))
    params['~']['w'] = w
    return hk.data_structures.to_haiku_dict(params)
