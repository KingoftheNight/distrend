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
### 2. Analyze Trends
Analyze the trends of different classes in X according to y and calculate the weight of each indicator (given by the information entropy bias) :
```python
sel.fit(X, y)
# Parameters:
# - X (DataFrame): input dataset.
# - y (DataFrame): input label of dataset.
```
The sel is trained to contain the following attributes:
```python
trend = sel.trend  # indicator trends
information = sel.information  # indicator informations
scores = sel.scores  # indicator weight
matrix = sel.matrix  # trend matrix
```
A trend matrix converts X into a trend vector (each cell is a string instead of a number), which can be mapped to a numerical matrix in the following way:
```python
X_new = sel.transform(X)
```
If you just want to get X_new, you can also use the following methods:
```python
X_new = distrend().fit_transform(X, y)
```
### 3. Visual indicators weight
Distrend provides some simple functions for visualizing the analysis results, starting with visualizing the indicators weight scores obtained above:
```python
# before drawing, we want to sort the results
sorted_indices = scores.argsort()[::-1]
sorted_value, sort_label = [scores[i] for i in sorted_indices], [X.columns[i] for i in sorted_indices]
# plot
sel.plotFeatureImportance(sorted_value, sort_label)
# Parameters:
# - show   (int): the number of entries displayed.
# - out    (str): save image name.
# - title  (str): image title.
```
![mark](http://img.frankgene.top/blog/20231112/tFUeGCyouTmH.png)
### 4. Visual disease trends
Use the following method to show the trend of all indicators in different classes:
```python
sel.plotTrend()
# Parameters:
# - trend  (dict): indicator trends.
# - out    (str) : save image name.
# - title  (str) : image title.
```
![mark](http://img.frankgene.top/blog/20231112/wVER3PC96QjK.png)
### 5. Visual trend weblogo
Use the following method to show the information entropy of the indicator for all classes:
```python
sel.plotWeblogo()
# Parameters:
# - information  (dict): indicator informations.
# - showLabel    (bool): show indicator names.
# - out          (str) : save image name.
```
![mark](http://img.frankgene.top/blog/20231112/sq1hdA6h03La.png)
### 6. Visual trend density
Use the following method to show the trend and density of target indicator for all classes:
```python
sel.plotTrendDensity()
# Parameters:
# - indicator (dict): indicator name.
# - legend    (bool): show legends.
# - out       (str) : save image name.
```
![mark](http://img.frankgene.top/blog/20231112/UkYhlFijdvSU.png)
