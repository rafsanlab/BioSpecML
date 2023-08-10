Project is on-going development.

# Installation
This is a guide of installation with code in Colab environment.
#. clone the repository:
```
!git clone https://github.com/rafsanlab/biospecml.git
```
#. pip install from the cloned path
```
!pip install '/content/biospecml'
```
#. test import
```
import biospecml.preprocessing as pp
pp.read_mat()
```
it should raise the following error:
```
TypeError: read_mat() missing 1 required positional argument: 'filename'
```
