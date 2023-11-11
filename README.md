# Distrend: A disease trend analysis module based on liquid biopsy indicators
## Introduction
Distrend is an analytical module based on trend vectors of disease liquid biopsy indicators, authored by Yuchao Liang. The module evaluates features using trends in different metrics and can generate visual analysis charts. It needs to run in a python environment and we recommend that users install it via conda or pip. Dependencies for packages such as Numpy, pandas, and tqdm are required and will be installed automatically when you install Distrend.
```bash
# install git if you need to
$conda install git
# install Distrend by github
$conda install git+https://github.com/KingoftheNight/distrend.git
# for Chinese users, gitee will install more quickly
$conda install git+https://gitee.com/KingoftheNight/distrend.git
```
## Import Packages
```python
from distrend import distrend
```
## Module function overview
### 1.checkStdIndicators(X)
`checkStdIndicators(X)` Check whether the features of X are included in distrend's standard indicators. You can view the current standard indicator index in the following ways:
```python
indicators = sel.stdIndicators
print(indicators.keys())
```
If it does not contain your feature name (for example, tumor_diameter), you can add the entry in the following ways:
```python
indicators['tumor_diameter'] = {'Type': 'Tumor Morphology', 'From': 1, 'To': 5, 'Unit': 'cm'}
sel.stdIndicators.update(indicators)
```
You can use joblib or other tools to save custom stdIndicators variables for future use:
```python
from joblib import dump, load
# save
dump(sel.stdIndicators, 'your_stdIndicators.dic')
# load
sel.stdIndicators = load('your_stdIndicators.dic')
```
