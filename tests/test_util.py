import chex
import jax
import jax.numpy as jnp

from jaxlft.util import cyclic_corr, cyclic_tensor, cyclic_corr_mat


class Test(chex.TestCase):
    def setUp(self) -> None:
        super().setUp()

        shape = (2, 3, 2, 1, 7)
        self.shape = shape
        self.a = jax.random.normal(jax.random.PRNGKey(100), shape)
        self.b = jax.random.normal(jax.random.PRNGKey(200), shape)

    def test_compatible(self):
        """Test different ways of calculation give the same result.

        Specifically, the methods ``cyclic_corr``, ``cyclic_corr_mat``
        and ``cyclic_tensor`` are related.
        """
        x = cyclic_corr(self.a, self.b)
        last_idc = tuple(range(-1, -len(self.shape)-1, -1))
        y = jnp.mean(cyclic_tensor(self.a, self.b), last_idc)
        outer = (self.a.flatten()[:, None] * self.b.flatten()[None, :])
        z = cyclic_corr_mat(outer.reshape(self.shape*2))
        self.assertTrue(jnp.allclose(x, y))
        self.assertTrue(jnp.allclose(y, z))
