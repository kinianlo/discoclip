from setuptools import setup, find_packages

setup(
    name="discoclip",
    version="0.1.0",
    packages=find_packages(),
    description="DiscoClip package for ARO dataset processing",
    author="Kinian Lo",
    author_email="",  # Add your email if desired
    install_requires=[
        "torch",
        "pandas",
        "lambeq",
        "scikit-learn",
    ],
    python_requires=">=3.9",
)
