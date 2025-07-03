from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="discoclip",
    version="0.1.0",
    packages=find_packages(),
    description="DiscoClip package for ARO dataset processing",
    author="Kin Ian Lo",
    author_email="kin.lo.20@ucl.ac.uk",
    install_requires=requirements,
    python_requires=">=3.9",
)
