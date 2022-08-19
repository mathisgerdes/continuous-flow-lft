from setuptools import setup


setup(
    name='jaxlft',
    version='0.1',
    description='Lattice Field Theory with JAX.',
    author='Mathis Gerdes',
    author_email='MathisGerdes@gmail.com',
    packages=['jaxlft'],
    python_requires='>=3.6',
    install_requires=[
        'chex',
        'dm-haiku',
        'jax',
        'jaxlib',
        'numpy',
        'optax',
    ],
)
