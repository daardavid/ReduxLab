# setup.py
"""
Configuración de instalación para PCA-SS.
Permite instalar el paquete como una librería Python.
"""

from setuptools import setup, find_packages
import os


# Leer README para la descripción larga
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()


# Leer requirements.txt
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [
            line.strip() for line in fh if line.strip() and not line.startswith("#")
        ]


setup(
    name="pca-ss",
    version="2.1.0",
    author="David Armando Abreu Rosique",
    author_email="davidabreu1110@gmail.com",
    description="Análisis de Componentes Principales para Datos Socioeconómicos",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/daardavid/PCA-SS",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Education",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
        "optional": [
            "adjustText>=0.7.3",
        ],
    },
    entry_points={
        "console_scripts": [
            "pca-ss=pca_gui_modern:main",
            "check-pca-deps=check_dependencies:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.json", "*.txt", "*.md", "THIRD_PARTY_LICENSES.txt"],
    },
    project_urls={
        "Bug Reports": "https://github.com/daardavid/PCA-SS/issues",
        "Source": "https://github.com/daardavid/PCA-SS",
        "Documentation": "https://github.com/daardavid/PCA-SS/wiki",
        "Ko-fi": "https://ko-fi.com/daardavid",
    },
)
