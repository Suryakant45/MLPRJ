from setuptools import find_packages,setup


setup(
    name='DimondPricePrediction',
    version='0.0.1',
    author='suryakant sahoo',
    author_email='ssuryakanta696@gmail.com',
    packages=find_packages(where="src"),
    package_dir={"": "src"},
)