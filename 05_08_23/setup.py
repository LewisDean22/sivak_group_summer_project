# -*- coding: utf-8 -*-
"""
Created on Sun Jul 23 13:21:01 2023

@author: lewis
"""
import numpy

from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize("*.pyx"),
    include_dirs=[numpy.get_include()],
    zip_safe=False
)