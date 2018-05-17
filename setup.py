# Austin Griffith
# Setup File

from setuptools import setup, find_packages
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_d = f.read()


setup(
    # name of package
    # setup using
    # $ pip install 'name'
    name = 'qcfoptions',

    version = '0.1',
    description = 'Option calculator',

    # uses readme file to give long description
    long_description = long_d,
    long_description_content_type = 'text/markdown',

    url = 'https://github.com/austingriffith94/qcfoptions',
    author = 'Austin Griffith',

    classifiers = [
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: QCF Students',
        # began with python 3.6.5
        'Programming Language :: Python :: 3.6.5',
    ],

    packages = ['bsoptions','barriers'],
    install_requires = ['numpy','scipy'],
    )
