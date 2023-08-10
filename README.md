Project is on-going development.

# Installation
This is a guide of installation with code in Colab environment.
1. clone the repository:
```
!git clone https://github.com/rafsanlab/biospecml.git
```
1. pip install from the cloned path
```
!pip install '/content/biospecml'
```
1. test import
```
import biospecml.preprocessing as pp
pp.read_mat()
```
it should raise the following error:
```
TypeError: read_mat() missing 1 required positional argument: 'filename'
```
