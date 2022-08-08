from setuptools import setup, find_packages

setup(
    name='helmholtz_x',
    version = '2.1',
    author='Ekrem Ekici',
    author_email='ee331@cam.ac.uk',
    packages=['helmholtz_x'],
    install_requires=[
        'h5py',
        'meshio',
        'numpy',
        'matplotlib',
        'scipy'
    ]
)
