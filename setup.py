import os
from setuptools import setup, find_packages


CLASSIFIERS = [
    'Development Status :: 4 - Beta',
    'License :: OSI Approved :: BSD License',
    'Operating System :: OS Independent',
    'Intended Audience :: Science/Research',
    'Programming Language :: Python',
    'Programming Language :: Python :: 2',
    'Programming Language :: Python :: 2.7',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
    'Topic :: Scientific/Engineering',
]


setup(name='h5netcdf',
      description='netCDF4 via h5py',
      long_description=(open('README.rst').read()
                        if os.path.exists('README.rst')
                        else ''),
      version='0.4.1',
      license='BSD',
      classifiers=CLASSIFIERS,
      author='Stephan Hoyer',
      author_email='shoyer@gmail.com',
      url='https://github.com/shoyer/h5netcdf',
      install_requires=['h5py'],
      tests_require=['netCDF4', 'pytest'],
      packages=find_packages())
