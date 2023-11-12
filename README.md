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
Before starting the analysis, you need to prepare the data X and label y, which should be in DataFrame format. The following is the test data contained in distrend:
```python
sel = distrend()
X, y = sel.testData()
```
### 1. Check Feature
Check whether the indicators (features) of X are included in distrend's standard indicator range.
```python
sel.checkStdIndicators(X)
# Parameters:
# - X (DataFrame): input dataset.
```
You can view the current distrend standard indicators as follows:
```python
indicators = sel.stdIndicators
print(list(indicators.keys()))
```
If some indicators are not within the distrend range, you can update stdIndicators as follows:
```python
indicators['your_indicator'] = {'Type': 'Self indicators', 'From': 5.0, 'To': 38.0, 'Unit': 'g/L'}
sel.stdIndicators.update(indicators)
```
