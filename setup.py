import os
from setuptools import setup, find_packages
import sys

if sys.version_info[:2] < (3, 5):
    raise RuntimeError('Python version >= 3.5 required.')

CLASSIFIERS = [
    'Development Status :: 4 - Beta',
    'License :: OSI Approved :: BSD License',
    'Operating System :: OS Independent',
    'Intended Audience :: Science/Research',
    'Programming Language :: Python',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Topic :: Scientific/Engineering',
]


setup(name='h5netcdf',
      description='netCDF4 via h5py',
      long_description=(open('README.rst').read()
                        if os.path.exists('README.rst')
                        else ''),
      version='0.7.4',
      license='BSD',
      classifiers=CLASSIFIERS,
      author='Stephan Hoyer',
      author_email='shoyer@gmail.com',
      url='https://github.com/shoyer/h5netcdf',
      python_requires='>=3.5',
      install_requires=['h5py'],
      tests_require=['netCDF4', 'pytest'],
      packages=find_packages())
