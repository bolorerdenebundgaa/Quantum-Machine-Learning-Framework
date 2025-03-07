from setuptools import setup, find_packages

setup(
    name="quantum_ml",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "torch",
        "torchvision",
        "qiskit",
        "scikit-learn",
        "matplotlib",
        "pylatexenc"
    ],
    author="Bolorerdene Bundgaa",
    description="A framework for integrating quantum computing with machine learning",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/bolorerdenebundgaa/quantum_ml",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
)
