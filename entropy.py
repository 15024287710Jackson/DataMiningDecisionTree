from math import log
import operator
# 计算数据的熵（entropy）
def entropy(dataSet):
    # 数据条数，计算数据集中实例的总数
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        # 每行数据的最后一个类别（也就是标签）
        currentLable = featVec[-1]
        if currentLable not in labelCounts.keys():
            labelCounts[currentLable] = 0
        # 统计有多少个类以及每个类的数量
        labelCounts[currentLable] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        # 计算单个类的熵值
        prob = float(labelCounts[key]) / numEntries
        # 累加每个类的熵值
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt