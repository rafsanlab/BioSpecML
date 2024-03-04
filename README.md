![GitHub Downloads (all assets, all releases)](https://img.shields.io/github/downloads/rafsanlab/biospecml/total)
![GitHub Release](https://img.shields.io/github/v/release/rafsanlab/biospecml)
![GitHub commit activity](https://img.shields.io/github/commit-activity/y/rafsanlab/biospecml)
<p align="center">
<img src="/img/biospecml-logo-v2.svg" alt="drawing" width="50%" />
</p>

# Introduction
A python code library package for spectra processing and analysis. Project is on-going development.

Features:
- read and apply processing on spectroscopy data
- robust plotting functions to plot spectral data
- prepare dataset and neural networks (UNet, LinearNet etc.)
- features packed training loop for model training
 
# Installation

### If you want to be able to edit the code:

1. clone the repository:
  ```python
  !git clone https://github.com/rafsanlab/biospecml.git
  ```
2. edits any .py files needed in src folder.
3. pip install from the cloned path (here is the example in Colab):
  ```python
  !pip install '/content/biospecml'
  ```
### If you want to use it directly:
install using pip:
  ```python
  !pip install git+https://github.com/rafsanlab/biospecml.git
  ```
