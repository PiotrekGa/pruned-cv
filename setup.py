from distutils.core import setup

setup(
    name='prunedcv',
    author='Piotr Gabryś',
    author_email='piotrek.gabrys@gmail.com',
    version='0.1',
    packages=['prunedcv',],
    install_requires=['scikit-learn>=0.20.2'],
    license='Creative Commons Attribution-Noncommercial-Share Alike license',
    long_description=open('README.md').read(),
    url="https://github.com/PiotrekGa/pruned-cv"
)