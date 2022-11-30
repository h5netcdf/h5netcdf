import os
import sys

from setuptools import find_packages, setup

if sys.version_info[:2] < (3, 8):
    raise RuntimeError("Python version >= 3.8 required.")

CLASSIFIERS = [
    "Development Status :: 5 - Production/Stable",
    "License :: OSI Approved :: BSD License",
    "Operating System :: OS Independent",
    "Intended Audience :: Science/Research",
    "Programming Language :: Python",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Topic :: Scientific/Engineering",
]


setup(
    name="h5netcdf",
    description="netCDF4 via h5py",
    long_description=(
        open("README.rst").read() if os.path.exists("README.rst") else ""
    ),
    license="BSD",
    classifiers=CLASSIFIERS,
    author="h5netcdf developers",
    author_email="devteam@h5netcdf.org",
    url="https://h5netcdf.org",
    python_requires=">=3.8",
    setup_requires=["setuptools_scm"],
    install_requires=["h5py", "packaging"],
    tests_require=["netCDF4", "pytest"],
    packages=find_packages(),
)
