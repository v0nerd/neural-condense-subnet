from setuptools import setup, find_packages

setup(
    name="neural_condense_subnet",  # Your package name
    version="0.0.1",  # Initial version
    author="CondenseAI",  # Your name
    author_email="",  # Your email
    description="Neural Condense Subnet - Bittensor",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/condenses/neural-condense-subnet",  # GitHub repo URL
    packages=find_packages(),  # Automatically find packages in the directory
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
