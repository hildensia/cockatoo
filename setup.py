#!/usr/bin/env python2

from setuptools import setup
from Cython.Build import cythonize
import cockatoo
import numpy

setup(
    name='cockatoo',
    version="0.0.1",
    description='The cockatoo experiment',
    include_dirs=[numpy.get_include()],
    author='Johannes Kulick',
    author_email='johannes.kulick@ipvs.uni-stuttgart.de',
    url='',
    packages=['cockatoo'],
    requires=['scipy', 'numpy', 'bayesian_changepoint_detection', 'pandas',
              'scikit.mcts', 'joint_dependency',
              'progressbar', 'enum34', 'blessings']
)
