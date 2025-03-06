from setuptools import setup, find_packages

setup(
    name="SpS-SpecDec",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "transformers",
        "torch",
        "tqdm",
        "numpy"
    ],
    author="Jayesh Thakare",
    author_email="thakarej91@gmail.com",
    description="A fast and optimized speculative decoding library for autoregressive models.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/jayeshthk/SpS-SpecDec.git",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)
