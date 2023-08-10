Project is on-going development.

# Installation
This is a guide of installation with code in Colab environment.
1. clone the repository:
```python
!git clone https://github.com/rafsanlab/biospecml.git
```
2. pip install from the cloned path
```python
!pip install '/content/biospecml'
```
3. test import
```python
import biospecml.preprocessing as pp
pp.read_mat()
```
it should raise the following error:
*TypeError: read_mat() missing 1 required positional argument: 'filename'*
