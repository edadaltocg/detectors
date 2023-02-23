from setuptools import find_packages, setup

install_requires = [
    "Pillow",
    "numpy",
    "optuna",
    "scikit-learn",
    "timm>=0.8.11.dev0",
    "torch>=1.7",
    "tqdm",
    "torchmetrics",
    "torchvision",
    "scipy",
    "transformers",
    "accelerate",
    "psutil",
    "jinja2",
    "matplotlib",
]


setup(
    name="detectors",
    author="Eduardo Dadalto with the help of all our contributors.",
    author_email="edadaltocg@gmail.com",
    description="Out of distribution detection benchmark.",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    keywords="vision deep learning pytorch OOD",
    license="MIT",
    url="https://github.com/edadaltocg/detectors",
    package_dir={"": "src"},
    packages=find_packages("src"),
    zip_safe=False,
    python_requires=">=3.8.0",
    install_requires=install_requires,
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
