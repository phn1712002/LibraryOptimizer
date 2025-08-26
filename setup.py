from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="library_optimizer",
    version="1.0.0",
    author="HoangggNam",
    author_email="phn1712002@gmail.com",
    description="A comprehensive Python library for metaheuristic optimization algorithms",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/HoangggNam/LibraryOptimizer",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Education",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.19.0",
        "matplotlib>=3.3.0",
        "tqdm>=4.60.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.12.0",
            "flake8>=3.9.0",
            "black>=21.0",
            "isort>=5.9.0",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
    },
    keywords=[
        "optimization",
        "metaheuristic",
        "evolutionary algorithms",
        "swarm intelligence",
        "artificial intelligence",
        "machine learning",
        "scientific computing",
    ],
    project_urls={
        "Documentation": "https://github.com/HoangggNam/LibraryOptimizer",
        "Source Code": "https://github.com/HoangggNam/LibraryOptimizer",
        "Bug Tracker": "https://github.com/HoangggNam/LibraryOptimizer/issues",
    },
)
