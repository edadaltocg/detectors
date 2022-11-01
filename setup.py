import re

from setuptools import find_packages, setup

_deps = [
    "Pillow",
    "accelerate>=0.10.0",
    "black==22.3",  # after updating to black 2023, also update Python version in pyproject.toml to 3.7
    "flake8>=3.8.3",
    "isort>=5.5.4",
    "numpy>=1.17",
    "optuna",
    "pyyaml>=5.1",
    "pytest",
    "pytest-timeout",
    "pytest-xdist",
    "python>=3.8.0",
    "ray[tune]",
    "regex!=2019.12.17",
    "requests",
    "scikit-learn",
    "timm",
    "torch>=1.7,!=1.12.0,<1.13.0",
    "tqdm>=4.27",
]
deps = {b: a for a, b in (re.findall(r"^(([^!=<>~ ]+)(?:[!=<>~ ].*)?$)", x)[0] for x in _deps)}

install_requires = [
    deps["numpy"],
    deps["pyyaml"],  # used for the model cards metadata
    deps["requests"],  # for downloading models over HTTPS
]

setup(
    name="detectors",
    version="0.1.0",
    author="Eduardo Dadalto with the help of all our contributors.",
    author_email="edadaltocg@gmail.com",
    description="Out of distribution detection benchmark.",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    keywords="NLP vision speech deep learning pytorch OOD",
    license="MIT",
    url="https://github.com/edadaltocg/detectors",
    package_dir={"": "src"},
    packages=find_packages("src"),
    package_data={"detectors": ["py.typed", "*.cu", "*.cpp", "*.cuh", "*.h"]},
    zip_safe=False,
    python_requires=">=3.8.0",
    install_requires=install_requires,
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
