from setuptools import setup, Extension
from Cython.Build import cythonize

extensions = [
    Extension(
        "mcts",  # Name of the compiled module
        ["MCTS_cpp.py", "MCTS.cpp"],  # Cython and C++ source files
        language="c++",  # Compile as C++
        libraries=["stdc++"],  # Link with the standard C++ library
        extra_compile_args=["-fopenmp", "-std=c++23"],  # Use C++11 standard
        extra_link_args=["-fopenmp", "-std=c++23"],  # Ensure linking also uses C++11
    )
]

setup(
    ext_modules=cythonize(extensions, annotate=True),
)
