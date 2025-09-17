import pybind11
from pybind11.setup_helpers import Pybind11Extension, build_ext
from pybind11 import get_cmake_dir
import pybind11.setup_helpers
from setuptools import setup, Extension
import numpy as np

ext_modules = [
    Pybind11Extension(
        "backtesting_core", 
        [
            "core/cpp/core.cpp",
            "core/cpp/bindings.cpp",
        ],
        include_dirs=[
            pybind11.get_include(),
            np.get_include(),
            "/usr/include/eigen3",  
            "core/cpp/",
            "core/third_party/eigen-3.4.0", 
            "core/third_party/eigen-3.4.0/Eigen",
        ],
        language='c++',
        cxx_std=17,   
        define_macros=[
            ('VERSION_INFO', '"dev"'),
            ('EIGEN_USE_THREADS', None),  
            ('EIGEN_DONT_PARALLELIZE', None),  
        ],
        extra_compile_args=[
            '-O3', 
            '-march=native',  
            '-fopenmp',  
            '-ffast-math',  
            '-DNDEBUG', 
        ],
        extra_link_args=[
            '-fopenmp', 
        ],
    ),
]

setup(
    name="quantitative backtesting engine",
    version="1",
    author="Yash Gupta",
    author_email="yas0901gupta@gmail.com",
    description="This is a high performance institutional grade backtesting engine with Python & C++",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)