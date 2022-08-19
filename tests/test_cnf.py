import haiku as hk
import jax.random
import numpy as np
from jaxlft.cnf import Phi4CNF, Phi4CNFConditional, KernelGauss
from unittest import TestCase
from functools import partial


class TestPhi4CNF(TestCase):
    def run_integration(self, rns, model_def, lattice_shape):
        model = model_def.transform(lattice_shape=lattice_shape, apply_jit=True)
        params = model.init(next(rns))
        batch = 10

        sample_prior, logp_prior = model.prior(next(rns), batch)
        self.assertEqual(sample_prior.shape, (batch,) + lattice_shape)
        self.assertEqual(logp_prior.shape, (batch,))

        sample_fwd, logp_fwd = model.forward(params, sample_prior, logp_prior)
        self.assertEqual(sample_fwd.shape, (batch,) + lattice_shape)
        self.assertEqual(logp_fwd.shape, (batch,))

        self.assertTrue(np.allclose(sample_prior, sample_fwd))
        self.assertTrue(np.allclose(logp_prior, logp_fwd))

        sample_spl, logp_spl = model.sample(params, next(rns), batch)
        self.assertEqual(sample_spl.shape, (batch,) + lattice_shape)
        self.assertEqual(logp_spl.shape, (batch,))

        # modify kernel weights to make behaviour non-trivial
        params = hk.data_structures.to_mutable_dict(params)
        params['~']['w'] = jax.random.normal(next(rns), params['~']['w'].shape)
        params = hk.data_structures.to_haiku_dict(params)

        # check that reverse is the inverse of forward
        sample_prior, logp_prior = model.prior(next(rns), batch)
        sample_fwd, logp_fwd = model.forward(params, sample_prior, logp_prior)
        sample_rev, logp_rev = model.reverse(params, sample_fwd, logp_fwd)
        self.assertTrue(np.allclose(sample_rev, sample_prior, atol=1e-6))
        self.assertTrue(np.allclose(logp_rev, logp_prior))
        self.assertFalse(np.allclose(sample_fwd, sample_prior, atol=1e-3))
        self.assertFalse(np.allclose(logp_fwd, logp_prior))

    def test_fixed_step_size(self):
        rns = hk.PRNGSequence(0)
        lattice_shapes = [(6, 6), (7, 7)]
        model_def = Phi4CNF(int_steps=50)
        for lattice_shape in lattice_shapes:
            self.run_integration(rns, model_def, lattice_shape)

    def test_dynamic_step_size(self):
        rns = hk.PRNGSequence(0)
        lattice_shapes = [(6, 6), (7, 7)]
        model_def = Phi4CNF(int_steps=None)
        for lattice_shape in lattice_shapes:
            self.run_integration(rns, model_def, lattice_shape)


class TestPhi4CNFConditional(TestCase):
    def run_integration(self, rns, model_def, lattice_shape, lam):
        model = model_def.transform(lattice_shape=lattice_shape, apply_jit=True)
        params = model.init(next(rns), lam=lam)
        batch = 10

        sample_prior, logp_prior = model.prior(next(rns), batch)
        self.assertEqual(sample_prior.shape, (batch,) + lattice_shape)
        self.assertEqual(logp_prior.shape, (batch,))

        sample_fwd, logp_fwd = model.forward(params, sample_prior, logp_prior, lam=lam)
        self.assertEqual(sample_fwd.shape, (batch,) + lattice_shape)
        self.assertEqual(logp_fwd.shape, (batch,))

        self.assertTrue(np.allclose(sample_prior, sample_fwd))
        self.assertTrue(np.allclose(logp_prior, logp_fwd))

        sample_spl, logp_spl = model.sample(params, next(rns), batch, lam=lam)
        self.assertEqual(sample_spl.shape, (batch,) + lattice_shape)
        self.assertEqual(logp_spl.shape, (batch,))

        # modify kernel weights to make behaviour non-trivial
        params = hk.data_structures.to_mutable_dict(params)
        params['~']['w'] = jax.random.normal(next(rns), params['~']['w'].shape)
        params = hk.data_structures.to_haiku_dict(params)

        # check that reverse is the inverse of forward
        sample_prior, logp_prior = model.prior(next(rns), batch)
        sample_fwd, logp_fwd = model.forward(params, sample_prior, logp_prior, lam=lam)
        sample_rev, logp_rev = model.reverse(params, sample_fwd, logp_fwd, lam=lam)
        self.assertTrue(np.allclose(sample_rev, sample_prior, atol=1e-6))
        self.assertTrue(np.allclose(logp_rev, logp_prior))
        self.assertFalse(np.allclose(sample_fwd, sample_prior, atol=1e-4))
        self.assertFalse(np.allclose(logp_fwd, logp_prior))

    def test_fixed_step_size(self):
        rns = hk.PRNGSequence(0)
        lattice_shapes = [(6, 6), (7, 7)]
        lam_range = (5., 6.)
        model_def = Phi4CNFConditional(
            int_steps=50,
            n_lam_kernel=5,
            lam_kernel_base=partial(KernelGauss, minmax=lam_range))
        for lattice_shape in lattice_shapes:
            self.run_integration(rns, model_def, lattice_shape, 5.5)

    def test_dynamic_step_size(self):
        rns = hk.PRNGSequence(0)
        lattice_shapes = [(6, 6), (7, 7)]
        lam_range = (5., 6.)
        model_def = Phi4CNFConditional(
            int_steps=None,
            n_lam_kernel=5,
            lam_kernel_base=partial(KernelGauss, minmax=lam_range))
        for lattice_shape in lattice_shapes:
            self.run_integration(rns, model_def, lattice_shape, 5.5)
