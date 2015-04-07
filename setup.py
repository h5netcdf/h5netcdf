from setuptools import setup, find_packages


setup(name='h5netcdf',
      version='0.1.dev0',
      license='MIT',
      author='Stephan Hoyer',
      author_email='shoyer@gmail.com',
      install_requires=['h5py'],
      packages=find_packages())
