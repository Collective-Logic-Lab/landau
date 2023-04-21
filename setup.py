
import os
from setuptools import setup

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = "landau",
    version = "0.0.0",
    author = "Bryan Daniels",
    author_email = "bryan.daniels.1@asu.edu",
    description = ("Use analogy with Landau theory of phase transitions to look for critical transitions in data."),
    license = "MIT license",
    keywords = "",
    url = "https://github.com/Collective-Logic-Lab/landau",
    packages=['landau',],
    install_requires=[
          'numpy',
          'scipy',
          'scikit-learn',
    ],
    extras_require = {
        'simulation':  ['pandas','numba']
    },
    long_description=read('README.md'),
    classifiers=[
        "",
    ],
)
