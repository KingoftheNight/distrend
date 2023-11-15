# distrend: 基于趋势向量的疾病分析模块
# 作者: 梁雨朝
# 日期: 2023.11.12

# 导入包
import math
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import roc_curve
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.cm import ScalarMappable
pd.options.mode.chained_assignment = None
import matplotlib.pyplot as plt

# 模型
class distrend():
    def __init__(self):
        self.colors = ['#2279B5', '#FE7F0F']
        self.stdIndicators = {
            'NEUT#': {'Type': 'Blood Routine Test', 'From': 1.8, 'To': 6.3, 'Unit': '/L'},
            'LYMPH#': {'Type': 'Blood Routine Test', 'From': 1.1, 'To': 3.2, 'Unit': '/L'},
            'EO#': {'Type': 'Blood Routine Test', 'From': 0.02, 'To': 0.52, 'Unit': '/L'},
            'BASO#': {'Type': 'Blood Routine Test', 'From': 0.0, 'To': 0.06, 'Unit': '/L'},
            'MONO#': {'Type': 'Blood Routine Test', 'From': 0.1, 'To': 0.6, 'Unit': '/L'},
            'NEUT%': {'Type': 'Blood Routine Test', 'From': 40.0, 'To': 75.0, 'Unit': '%'},
            'LYMPH%': {'Type': 'Blood Routine Test', 'From': 20.0, 'To': 50.0, 'Unit': '%'},
            'EO%': {'Type': 'Blood Routine Test', 'From': 0.4, 'To': 8.0, 'Unit': '%'},
            'BASO%': {'Type': 'Blood Routine Test', 'From': 0.0, 'To': 1.0, 'Unit': '%'},
            'MONO%': {'Type': 'Blood Routine Test', 'From': 3.0, 'To': 10.0, 'Unit': '%'},
            'RDW-SD': {'Type': 'Blood Routine Test', 'From': 37.0, 'To': 50.0, 'Unit': 'fl'},
            'RDW-CV': {'Type': 'Blood Routine Test', 'From': 11.5, 'To': 14.5, 'Unit': '%'},
            'P-LCR': {'Type': 'Blood Routine Test', 'From': 13.0, 'To': 43.0, 'Unit': '%'},
            'HCT': {'Type': 'Blood Routine Test', 'From': 0.35, 'To': 0.45, 'Unit': 'L/L'},
            'MCV': {'Type': 'Blood Routine Test', 'From': 82.0, 'To': 100.0, 'Unit': 'fl'},
            'MCHC': {'Type': 'Blood Routine Test', 'From': 316.0, 'To': 354.0, 'Unit': 'g/L'},
            'MCH': {'Type': 'Blood Routine Test', 'From': 316.0, 'To': 354.0, 'Unit': 'pg'},
            'PLT': {'Type': 'Blood Routine Test', 'From': 125.0, 'To': 350.0, 'Unit': '/L'},
            'PDW': {'Type': 'Blood Routine Test', 'From': 15.0, 'To': 17.0, 'Unit': '%'},
            'HGB': {'Type': 'Blood Routine Test', 'From': 110.0, 'To': 150.0, 'Unit': 'g/L'},
            'MPV': {'Type': 'Blood Routine Test', 'From': 5.0, 'To': 15.0, 'Unit': 'fl'},
            'PCT': {'Type': 'Blood Routine Test', 'From': 0.06, 'To': 0.4, 'Unit': '%'},
            'RBC': {'Type': 'Blood Routine Test', 'From': 3.8, 'To': 5.1, 'Unit': '/L'},
            'WBC': {'Type': 'Blood Routine Test', 'From': 3.5, 'To': 9.5, 'Unit': '/L'},
            'PT': {'Type': 'Blood Routine Test', 'From': 8.2, 'To': 11.5, 'Unit': 's'},
            'PTA': {'Type': 'Blood Routine Test', 'From': 70.0, 'To': 130.0, 'Unit': '%'},
            'APTT': {'Type': 'Blood Routine Test', 'From': 25.0, 'To': 35.0, 'Unit': 's'},
            'FIB': {'Type': 'Blood Routine Test', 'From': 2.0, 'To': 4.0, 'Unit': 'g/L'},
            'TT': {'Type': 'Blood Routine Test', 'From': 16.0, 'To': 18.0, 'Unit': 's'},
            'INR': {'Type': 'Blood Routine Test', 'From': 0.8, 'To': 1.2, 'Unit': '-'},
            'TP': {'Type': 'Blood Biochemistry Test', 'From': 60.0, 'To': 80.0, 'Unit': 'g/L'},
            'ALB': {'Type': 'Blood Biochemistry Test', 'From': 35.0, 'To': 55.0, 'Unit': 'g/L'},
            'TBIL': {'Type': 'Blood Biochemistry Test', 'From': 3.42, 'To': 20.5, 'Unit': 'umol/L'},
            'DBIL': {'Type': 'Blood Biochemistry Test', 'From': 0.0, 'To': 7.0, 'Unit': 'umol/L'},
            'ADA': {'Type': 'Blood Biochemistry Test', 'From': 0.0, 'To': 25.0, 'Unit': 'U/L'},
            'ALP': {'Type': 'Blood Biochemistry Test', 'From': 45.0, 'To': 132.0, 'Unit': 'U/L'},
            'LDH': {'Type': 'Blood Biochemistry Test', 'From': 109.0, 'To': 245.0, 'Unit': 'U/L'},
            'CHE': {'Type': 'Blood Biochemistry Test', 'From': 3.7, 'To': 7.0, 'Unit': 'U/L'},
            'K': {'Type': 'Blood Biochemistry Test', 'From': 3.5, 'To': 5.3, 'Unit': 'mmol/L'},
            'NA': {'Type': 'Blood Biochemistry Test', 'From': 137.0, 'To': 147.0, 'Unit': 'mmol/L'},
            'CL': {'Type': 'Blood Biochemistry Test', 'From': 99.0, 'To': 110.0, 'Unit': 'mmol/L'},
            'Ca': {'Type': 'Blood Biochemistry Test', 'From': 2.08, 'To': 2.6, 'Unit': 'mmol/L'},
            'CO2CP': {'Type': 'Blood Biochemistry Test', 'From': 22.0, 'To': 29.0, 'Unit': 'mmol/L'},
            'UREA': {'Type': 'Blood Biochemistry Test', 'From': 2.14, 'To': 7.85, 'Unit': 'umol/L'},
            'Cr': {'Type': 'Blood Biochemistry Test', 'From': 41.0, 'To': 80.0, 'Unit': 'mmol/L'},
            'UA': {'Type': 'Blood Biochemistry Test', 'From': 155.0, 'To': 357.0, 'Unit': 'umol/L'},
            'GLU': {'Type': 'Blood Biochemistry Test', 'From': 3.9, 'To': 6.19, 'Unit': 'mmol/L'},
            'TG': {'Type': 'Blood Biochemistry Test', 'From': 0.4, 'To': 1.7, 'Unit': 'mmol/L'},
            'TC': {'Type': 'Blood Biochemistry Test', 'From': 3.1, 'To': 5.7, 'Unit': 'mmol/L'},
            'HDL': {'Type': 'Blood Biochemistry Test', 'From': 1.2, 'To': 1.65, 'Unit': 'mmol/L'},
            'LDL': {'Type': 'Blood Biochemistry Test', 'From': 2.59, 'To': 3.37, 'Unit': 'mmol/L'},
            'AMY': {'Type': 'Blood Biochemistry Test', 'From': 8.0, 'To': 53.0, 'Unit': 'U/L'},
            'LPS': {'Type': 'Blood Biochemistry Test', 'From': 1.0, 'To': 60.0, 'Unit': 'umol/L'},
            'CK-MB': {'Type': 'Blood Biochemistry Test', 'From': 0.0, 'To': 25.0, 'Unit': 'U/L'},
            'GLB': {'Type': 'Blood Biochemistry Test', 'From': 20.0, 'To': 40.0, 'Unit': 'g/L'},
            'A/G': {'Type': 'Blood Biochemistry Test', 'From': 1.2, 'To': 2.4, 'Unit': '-'},
            'IBIL': {'Type': 'Blood Biochemistry Test', 'From': 1.7, 'To': 10.2, 'Unit': '-'},
            'AG': {'Type': 'Blood Biochemistry Test', 'From': 8.0, 'To': 16.0, 'Unit': 'g/L'},
            'U-PRO': {'Type': 'Urine Routine Test', 'From': 0.0, 'To': 0.0, 'Unit': '-'},
            'U-URO': {'Type': 'Urine Routine Test', 'From': 0.0, 'To': 17.0, 'Unit': 'g/L'},
            'U-BIL': {'Type': 'Urine Routine Test', 'From': 0.0, 'To': 0.0, 'Unit': '-'},
            'U-pH': {'Type': 'Urine Routine Test', 'From': 4.6, 'To': 8.0, 'Unit': 'umol/L'},
            'U-KET': {'Type': 'Urine Routine Test', 'From': 0.0, 'To': 0.0, 'Unit': '-'},
            'U-SG': {'Type': 'Urine Routine Test', 'From': 1.003, 'To': 1.03, 'Unit': 'g/L'},
            'U-GLU': {'Type': 'Urine Routine Test', 'From': 0.0, 'To': 0.0, 'Unit': '-'},
            'U-COL': {'Type': 'Urine Routine Test', 'From': 0.0, 'To': 0.0, 'Unit': '-'},
            'U-SRC': {'Type': 'Urine Routine Test', 'From': 0.0, 'To': 3.4, 'Unit': 'g/L'},
            'U-cond': {'Type': 'Urine Routine Test', 'From': 5.0, 'To': 38.0, 'Unit': 'g/L'}
        }
        self.threshold = 0.11
        self.trend = {}
        self.words = ['N', 'D', 'C', 'H', 'P']
        self.words_color = {'N': '#ea8685', 'D': '#cf6a87', 'C': '#778beb', 'H': '#546de5', 'P': '#574b90'}
        self.words_path = {
            'N': 'M60,100H45.6L13.8,27.5h-0.6V100H0V0h14.4l31.7,72.5h0.6V0H60V100z',
            'D': 'M60,50.3c0,17.9-3.7,30.7-11,38.3C41.6,96.2,31.7,100,19.2,100H0V0h19.2c14,0,24.2,3.9,30.8,11.7		C56.7,19.5,60,32.4,60,50.3z M46.4,50.3c0-14.4-2.3-24.6-6.8-30.4c-4.5-5.8-11.3-8.8-20.4-8.8H13v77.8h6.2c9.1,0,15.8-2.8,20.4-8.5		C44.2,74.8,46.4,64.7,46.4,50.3z',
            'C': 'M60.1,60c-0.4,14.5-3.3,24.8-8.7,30.9c-5.5,6.1-12,9.1-19.7,9.1c-8.7,0-16.2-3.7-22.4-11.1		C3.1,81.4,0,69.5,0,53.1c0-17.9,3-31.2,9-40C15,4.4,22.7,0,32.2,0c8,0,14.6,3.1,19.9,9.4c5.3,6.3,7.7,15.1,7.4,26.6h-12		c0-8.4-1.3-14.7-3.8-18.9c-2.6-4.2-6.4-6.3-11.5-6.3c-5.8,0-10.5,3.1-13.9,9.4c-3.5,6.3-5.2,16.9-5.2,31.7		c0,13.7,1.7,23.3,5.2,28.9c3.5,5.5,7.9,8.3,13.4,8.3c4,0,7.6-2,10.9-6c3.3-4,4.9-11.7,4.9-23.1H60.1z',
            'H': 'M60,100H46.6V53.2H13.4V100H0V0h13.4v42.1h33.2V0H60V100z',
            'P': 'M60,30.4c0,9.4-2.7,16.8-8,22.2c-5.3,5.5-12.8,8.2-22.3,8.2H13.1V100H0V0h29.7C39.2,0,46.7,2.7,52,8.2		C57.3,13.7,60,21.1,60,30.4z M46.9,30.4c0-7.4-1.7-12.5-5.1-15.2c-3.4-2.7-8.6-4.1-15.4-4.1H13.1v38.6h13.1c6.9,0,12-1.4,15.4-4.1		C45.1,42.9,46.9,37.8,46.9,30.4z',
        }
        self.words_maps = {'N': 0, 'D': 0.25, 'C': 0.5, 'H': 0.75, 'P': 1}
    
    
    def __repr__(self):
        return "Processing finished!"

    
    def testData(self):
        df = pd.DataFrame()
        for key in list(self.stdIndicators.keys())[30:40]:
            df[key] = np.random.rand(200)*self.stdIndicators[key]['To']*1.5+self.stdIndicators[key]['From']*0.5
        df['label'] = np.random.choice([0, 1], size=200)
        return df.iloc[:,:-1], df.iloc[:,-1]


    def checkStdIndicators(self, X):
        for item in X.columns:
            if item not in self.stdIndicators:
                print(f"{item} is not in stdIndicators, please update it!")
    
    
    def area(self, content, each_fs, k):
        out = []
        t = 0
        for i in range(len(content)):
            t += each_fs[i]
            if t >= sum(each_fs) * k:
                out.append(round(content[i][1], 2))
                break
        t = 0
        for i in range(len(content)):
            t += each_fs[i]
            if t >= sum(each_fs) * (1 - k):
                out.append(round(content[i][1], 2))
                break
        return out
    
    
    def trendAnalyze(self, X):
        trend = {}
        final_trend = {}
        for fs in tqdm(X.columns, desc=f'Class {self.ori_y[X.index[0]]} Trends'):
            trend[fs] = []
            fs_stdIndicator = self.stdIndicators[fs]
            min_block, max_block = min([min(X[fs]), fs_stdIndicator['From']]), max([max(X[fs]), fs_stdIndicator['To']])
            trend[fs].append(min_block)
            block = pd.cut(X[fs], np.linspace(min_block, max_block, 1001)).value_counts().sort_index()
            low_index, low_block_left, low_block_right = 0, 0, 0
            for i in range(len(block)):
                if fs_stdIndicator['From'] in block.index[i]:
                    low_index = i
                    break
            for i in range(low_index, -1, -1):
                if low_block_left >= sum(block)*self.threshold*0.5:
                    trend[fs].append(block.index[i].left)
                    break
                elif i == 0:
                    trend[fs].append(block.index[i].left)
                else:
                    low_block_left += block[i]
            for i in range(low_index, len(block)):
                low_block_right += block[i]
                if low_block_right >= sum(block)*self.threshold*0.5:
                    trend[fs].append(block.index[i].right)
                    break
                elif i == len(block)-1:
                    trend[fs].append(block.index[i].right)
            high_index, high_block_left, high_block_right = 0, 0, 0
            for i in range(len(block)):
                if fs_stdIndicator['To'] in block.index[i]:
                    high_index = i
                    break
            for i in range(high_index, -1, -1):
                high_block_left += block[i]
                if high_block_left >= sum(block)*self.threshold*0.5:
                    trend[fs].append(block.index[i].left)
                    break
                elif i == 0:
                    trend[fs].append(block.index[i].left)
            for i in range(high_index, len(block)):
                high_block_right += block[i]
                if high_block_right >= sum(block)*self.threshold*0.5:
                    trend[fs].append(block.index[i].right)
                    break
                elif i == len(block)-1:
                    trend[fs].append(block.index[i].right)
            trend[fs].append(max_block)
            final_trend[fs] = {}
            for i in range(len(trend[fs])-1):
                if trend[fs][i] <= trend[fs][i+1]:
                    if i == 0:
                        final_trend[fs][self.words[i]] = pd.Interval(-float('inf'), trend[fs][i+1], closed='right')
                    else:
                        final_trend[fs][self.words[i]] = pd.Interval(trend[fs][i], trend[fs][i+1], closed='right')
                    if i == len(trend[fs])-2:
                        final_trend[fs][self.words[i]] = pd.Interval(trend[fs][i], float('inf'), closed='right')
                else:
                    if i == 0:
                        final_trend[fs][self.words[i]] = pd.Interval(-float('inf'), trend[fs][i], closed='right')
                    else:
                        final_trend[fs][self.words[i]] = pd.Interval(trend[fs][i+1], trend[fs][i], closed='right')
                    if i == len(trend[fs])-2:
                        final_trend[fs][self.words[i]] = pd.Interval(trend[fs][i+1], float('inf'), closed='right')
        return final_trend
    
    
    def trendTransform(self, X, trend):
        matrix = X.copy(deep=True)
        for fs in X.columns:
            for i in range(len(X[fs])):
                for word in self.words:
                    if X[fs].iloc[i] in trend[fs][word]:
                        matrix[fs].iloc[i] = word
        return matrix
    
    
    def fit(self, X, y):
        self.ori_X = X
        self.ori_y = y
        self.matrix = X.copy(deep=True)
        self.trend = {}
        for label_class in y.unique():
            # 计算趋势
            self.trend[label_class] = self.trendAnalyze(X[y.isin([label_class])])
            # 转换趋势向量
            self.matrix.update(self.trendTransform(X[y.isin([label_class])], self.trend[label_class]))
        # 计算信息熵
        self.information = self.informationAnalyze(self.matrix, y)
        # 计算信息熵bias
        self.bias = self.informationBias(information=self.information)
        self.scores = np.array([self.bias[item] for item in X.columns])
        # 计算疾病分数
        self.dMatrix = X.copy(deep=True)
        self.dMatrix = self.dMatrix.apply(self.apply_sigmoid)
        self.dMatrix['dValue'] = self.dMatrix.sum(axis=1)
        self.dValThred = self.dValueThred(self.dMatrix, y)
        return self
    
    
    def dValueThred(self, dMatrix, y):
        thred = {}
        for key in dMatrix.columns:
            fpr, tpr, thresholds = roc_curve(y, dMatrix[key])
            threshold_index = np.argmax(tpr - fpr)
            thred[key] = thresholds[threshold_index]
        return thred
    
    def apply_sigmoid(self, column):
        column_name = column.name
        return self.sigmoid(column) * self.bias[column_name]
    
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    
    def transform(self, X=None, maps=None):
        if not X:
            X = self.matrix
        if not maps:
            maps = self.words_maps
        new_matrix = X.copy(deep=True)
        for i in range(len(new_matrix.index)):
            for j in range(len(new_matrix.columns)):
                new_matrix.iloc[i, j] = maps[new_matrix.iloc[i, j]]
        return new_matrix
    
    
    def fit_transform(self, X, y, maps=None):
        self.matrix = X.copy(deep=True)
        self.trend = {}
        for label_class in y.unique():
            # 计算趋势
            self.trend[label_class] = self.trendAnalyze(X[y.isin([label_class])])
            # 转换趋势向量
            self.matrix.update(self.trendTransform(X[y.isin([label_class])], self.trend[label_class]))
        if not maps:
            maps = self.words_maps
        new_matrix = self.matrix.copy(deep=True)
        for i in range(len(new_matrix.index)):
            for j in range(len(new_matrix.columns)):
                new_matrix.iloc[i, j] = maps[new_matrix.iloc[i, j]]
        return new_matrix
    
    
    def informationBias(self, information=None):
        if not information:
            information = self.information
        result = {}
        for key, inf in information.items():
            for item, group in inf.items():
                if item not in result:
                    result[item] = []
                result[item].append([value for key, value in group.items()])
        for key, item in result.items():
            result[key] = np.linalg.norm(np.array(item[0]) - np.array(item[1]))
        return result
    
    
    def informationAnalyze(self, X, y):
        information = {}
        for label_class in y.unique():
            sequence = [''.join(X[y.isin([label_class])].iloc[i,:]) for i in range(len(X[y.isin([label_class])]))]
            # 提取信息熵矩阵
            count = []
            for i in range(len(sequence[0])):
                group = [0 for word in self.words]
                for line in sequence:
                    group[self.words.index(line[i])] += 1
                count.append(group)
            count = np.array(count)
            # 计算信息熵
            def weblogo_check(value):
                H = 0
                for i in value:
                    if i != 0:
                        H += i * math.log(i, 2)
                Rseq = math.log(len(value), 2) + H
                out = []
                for i in value:
                    if i != 0:
                        out.append(Rseq * i)
                    else:
                        out.append(0)
                return out
            # 频率
            gap, value = 0, {}
            for i in range(len(count)):
                group = list(count[i, j] / np.sum(count[i]) for j in range(len(count[i])))
                # 转信息熵
                group = weblogo_check(group)
                value[X.columns[i]] = {key: group[j] for j, key in enumerate(self.words)}
                gap = max(sum(group), gap)
            information[label_class] = value
        return information
        
    
    def plotWeblogo(self, information=None, showLabel=True, out='DisTrend_Weblogo.svg'):
        if not information:
            information = self.information
        # 绘图
        svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="{int(len(information[0]) * 24 + 70)}" height="{220 * len(information)}">'
        # 设置标尺像素比例
        def weblogo_gap(gap):
            if gap <= 1.5:
                gap, rd_l = 11, 1
            elif 1.5 < gap <= 2:
                gap, rd_l = 9, 2
            elif 2 < gap <= 2.5:
                gap, rd_l = 9, 2
            elif 2.5 < gap <= 3:
                gap, rd_l = 7, 3
            elif 3 < gap <= 3.5:
                gap, rd_l = 7, 3
            elif 3.5 < gap <= 4:
                gap, rd_l = 5, 4
            elif gap > 4:
                gap, rd_l = 5, 4
            return gap, rd_l
        # 删除0值并返回氨基酸索引
        def weblogo_del(line):
            out1 = []
            out2 = []
            for key, value in line.items():
                if value != 0:
                    out1.append(value)
                    out2.append(key)
            return [out1, out2]
        # 更新元素列表
        def weblogo_update(line, aa):
            out = [[], []]
            test = 'yes'
            for i in range(len(line[0])):
                if aa != line[0][i]:
                    out[0] = out[0] + [line[0][i]]
                    out[1] = out[1] + [line[1][i]]
                elif test != 'yes':
                    out[0] = out[0] + [line[0][i]]
                    out[1] = out[1] + [line[1][i]]
                else:
                    test = 'no'
            return out
        # 求取元素上下边界及缩放比例
        def weblogo_yc(line, y1, y2, r, gap):
            aa_value = min(line[0])
            aa_id = line[1][line[0].index(aa_value)]
            new_line = weblogo_update(line, aa_value)
            r += aa_value
            y1 -= r * gap * 5
            c = (y2 - y1) / 100
            return y1, y2, c, new_line, r, aa_id
        # 基础偏移量
        y_offset = -160
        for key, value in information.items():
            gap, rd_l = weblogo_gap(max([sum(num for num in value[item].values()) for item in value]))
            y_offset += 200
            # 总体偏移量
            y = y_offset
            # 定义xy轴和y轴标签
            ruler = f'<text fill="#333333" x="0" y="10" transform="translate(0,{y + 90})rotate(-90, 10, 5)">Class {key} (bit)</text><rect x="45" y="{y - 10}" width="2" height="120" fill="#333333"/><rect x="45" y="{y + 108}" width="{int(len(value) * 24 + 50)}" height="2" fill="#333333"/><rect x="57" y="{y + 109}" width="2" height="5" fill="#333333"/>'
            # 定义y轴刻度
            for j in range(rd_l + 1):
                # 定义y轴数字和y轴大刻度
                ruler += f'<text fill="#333333" x="25" y="{y + 114}">{j}</text><rect x="36" y="{y + 108}" width="10" height="2" fill="#333333"/>'
                if j < rd_l:
                    for i in range(4):
                        y -= gap
                        # 定义y轴小刻度
                        ruler += f'<rect x="41" y="{y + 108}" width="5" height="2" fill="#333333"/>'
                    y -= gap
            # 总体偏移量
            x = 33
            y = y_offset
            # 定义x轴刻度
            if showLabel:
                k, label = -1, list(value.keys()) + ['', '', '', '', '']
                for j in range(int(len(value) / 5) + 1):
                    for i in range(4):
                        k += 1
                        x += 24
                        # 定义x轴小刻度
                        ruler += f'<text fill="#333333" x="{x - 4}" y="{y + 134}" transform="translate(-2, -7) rotate(45, {x - 4}, {y + 134})">{label[k]}</text><rect x="{x}" y="{y + 109}" width="2" height="5" fill="#333333"/>'
                    k += 1
                    x += 24
                    # 定义x轴数字和x轴大刻度
                    ruler += f'<text fill="#333333" x="{x - 4}" y="{y + 134}" transform="translate(-2, -7) rotate(45, {x - 4}, {y + 134})">{label[k]}</text><rect x="{x}" y="{y + 109}" width="2" height="10" fill="#333333"/>'
            else:
                for j in range(int(len(value) / 5) + 1):
                    for i in range(4):
                        x += 24
                        # 定义x轴小刻度
                        ruler += f'<rect x="{x}" y="{y + 109}" width="2" height="5" fill="#333333"/>'
                    x += 24
                    # 定义x轴数字和x轴大刻度
                    ruler += f'<text fill="#333333" x="{x - 4}" y="{y + 134}">{(j + 1) * 5}</text><rect x="{x}" y="{y + 109}" width="2" height="10" fill="#333333"/>'
            # 总体偏移量
            x = 23
            logo = ''
            for item in value:
                # 定义偏移量
                x += 24
                y2 = y_offset + 108
                r = 0
                line = weblogo_del(value[item])
                li = len(line[0])
                for j in range(li):
                    y1 = y_offset + 108
                    if len(line[0]) != 0:
                        # 求取元素上下边界及缩放比例
                        y1, y2, c, line, r, aa_id = weblogo_yc(line, y1, y2, r, gap)
                        y2 = y1
                        # 写入元素
                        logo += f'<path fill="{self.words_color[aa_id]}" d="{self.words_path[aa_id]}" transform="translate({x},{y1})scale(0.4,{c})"/>'
            svg += ruler + logo
        # svg并输出
        with open(out, 'w', encoding='UTF-8') as f:
            f.write(svg+'</svg>')
    
    
    def plotTrend(self, trend=None, out=None, title='Comparison of feature trends in different classes'):
        if not trend:
            trend = self.trend
        values = {}
        for label_class, item in trend.items():
            for key, group in item.items():
                if key not in values:
                    values[key] = {}
                if label_class not in values[key]:
                    values[key][label_class] = []
                for word in group:
                    values[key][label_class].append(group[word].right)
                values[key][label_class][-1] = values[key][label_class][-2] * 1.2
        # 设置子图行数和列数
        cols = 5  # 计算列数
        rows = (len(values) + cols - 1) // cols  # 计算行数
        # 绘制组合图
        fig, axs = plt.subplots(rows, cols, figsize=(cols*3, rows*1.2))  # 设置图形的大小
        fig.suptitle(title, fontsize=16)  # 设置总标题
        i = -1
        for feature in values:
            i += 1
            row_idx = i // cols
            col_idx = i % cols
            ax = axs[row_idx, col_idx]
            left = [0 for j in values[feature]]
            for j, word in enumerate(self.words):
                group = []
                label = []
                for label_class in values[feature]:
                    group.append(values[feature][label_class][j])
                    label.append(f'Class {label_class}')
                if word == 'P':
                    group = [max(group) for n in group]
                ax.barh(label, group, label=word, left=left, color=self.words_color[word])
                left = group
            ax.set_title(feature)
            ax.set_xlim(0, max(left))
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.set_xlabel('')
            ax.set_ylabel('')
        # 删除未使用的子图
        if len(values) < rows * cols:
            for i in range(len(values), rows * cols):
                fig.delaxes(axs.flatten()[i])
            # axs.flatten()[len(values)-1].legend(ncol=len(self.words), loc='upper left', bbox_to_anchor=(1.15, 0.7))
        plt.tight_layout()  # 调整子图的布局，避免重叠
        if out:
            plt.savefig(out, dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()
    
    
    def plotTrendDensity(self, indicator=None, trend=None, legend=True, out=None):
        if not trend:
            trend = self.trend
        values = {}
        for label_class, item in trend.items():
            for key, group in item.items():
                if key not in values:
                    values[key] = {}
                if label_class not in values[key]:
                    values[key][label_class] = []
                for word in group:
                    values[key][label_class].append(group[word].right)
                values[key][label_class][-1] = values[key][label_class][-2] * 1.2
        if not indicator:
            indicator = list(values.keys())[0]
        values = {indicator: values[indicator]}
        # 绘制trend
        fig, axs = plt.subplots(2, 1, figsize=(4, 4), gridspec_kw={'height_ratios': [5, 1]})  # 设置图形的大小
        ax = axs[1]
        max_x = max(values[max(values)][max(values[max(values)])])
        left = [0 for j in values[indicator]]
        for j, word in enumerate(self.words):
            group = []
            label = []
            for label_class in values[indicator]:
                group.append(values[indicator][label_class][j])
                label.append(f'Class {label_class}')
            if legend:
                for k, x in enumerate(group):
                    if x-left[k] > 0.01:
                        if word == 'P':
                            ax.text((x+max(left))/2 - max_x*0.018, k, word, ha='left', va='center', color='#ffffff')
                        else:
                            ax.text((x+left[k])/2 - max_x*0.018, k, word, ha='left', va='center', color='#ffffff')
            if word == 'P':
                group = [max(group) for n in group]
            ax.barh(label, group, label=word, left=left, color=self.words_color[word])
            left = group
        ax.set_xlim(0, max(left))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.set_xlabel('')
        ax.set_ylabel('')
        # 绘制密度
        ax = axs[0]
        for label_class in self.ori_y.unique()[::-1]:
            data_class = self.ori_X[self.ori_y == label_class]
            data_class[indicator].plot(kind='density', ax=ax, color=self.colors[label_class], title=indicator, label=f'Class {label_class}')
        ax.set_xlim(0, max(left))
        ax.set_xlabel('')
        ax.set_ylabel('Density')
        if legend:
            ax.legend()
        plt.tight_layout()  # 调整子图的布局，避免重叠
        if out:
            plt.savefig(out, dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()
    
    
    def plotFeatureImportance(self, value, label, show=None, out=None, title='Feature Importance Score'):
        if show:
            plt.figure(figsize=(4, show*0.4))
            plt.ylim(-0.5, show+0.6)
            plt.barh(['Other Features'] + label[::-1][-show:], [sum(value[::-1][:-show])] + value[::-1][-show:], color='#5B9BD5')
        else:
            plt.figure(figsize=(4, len(label)*0.4))
            plt.ylim(-0.5, len(label)-0.2)
            plt.barh(label[::-1], value[::-1], color='#5B9BD5')
        plt.title(title)
        plt.ylabel('Features')
        plt.xlabel('Importance Score')
        if out:
            plt.savefig(out, dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()


    def plotForce(self, X=None, out=None):
        if not X:
            X = self.ori_X.iloc[0]
        # 获取force
        force = []
        for key in X.keys():
            force.append((self.bias[key] / (1 + np.exp(-X[key]))) - self.dValThred[key])
        # 归一化
        force = 2 * (np.array(force) - np.min(force)) / (np.max(force) - np.min(force)) - 1
        # 绘制force
        plt.figure(figsize=(len(force)*0.45, 0.45))
        cmap = LinearSegmentedColormap.from_list('custom_cmap', ['#568AAD', '#FFFFFF', '#F2AA6C'], N=256)
        colors = cmap(force)
        plt.scatter([i for i in range(len(force))], [0 for i in range(len(force))], s=300, c=colors, cmap=cmap)
        plt.xticks([i for i in range(len(force))], list(X.keys()), rotation=90)
        plt.xlim(-0.55, len(force)-0.45)
        plt.ylim(-0.5, 0.55)
        plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelleft=False)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['bottom'].set_visible(False)
        plt.gca().spines['left'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        if out:
            plt.savefig(out, dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        

