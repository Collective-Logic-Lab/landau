
import os
from setuptools import setup

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = "landau",
    version = "0.0.0",
    author = "Bryan Daniels",
    author_email = "bryan.daniels.1@asu.edu",
    description = (""),
    license = "",
    keywords = "",
    url = "https://github.com/bcdaniels/landau",
    packages=['landau',],
    long_description=read('README.md'),
    classifiers=[
        "",
    ],
)
