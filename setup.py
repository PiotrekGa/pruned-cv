from distutils.core import setup

setup(
    name='prunedcv',
    author='Piotr Gabryś',
    author_email='piotrek.gabrys@gmail.com',
    version='0.0.2',
    packages=['prunedcv',],
    install_requires=[
        'scikit-learn>=0.20.2',
        'pandas>=0.24.1',
        'numpy>=1.16.1'],
    license='MIT',
    long_description=open('README.md').read(),
    url="https://github.com/PiotrekGa/pruned-cv"
)