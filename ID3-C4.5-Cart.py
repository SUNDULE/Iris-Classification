from math import log
import operator
import treePlotter
import random

def load_data(iris_path='./Iris.csv', rate=0.8):
    labels = ['花萼长', '花萼宽', '花瓣长', '花瓣宽']
    label2vec = {'Iris-setosa': '0', 'Iris-versicolor': '1', 'Iris-virginica': '2'}
    with open(iris_path) as f:
        data = f.readlines()
    dataset = list()
    for sample in data:
        sample = sample.replace('\n', '')
        row = sample.split(',')
        label = label2vec[row[-1]]
        row = row[:-1]
        row.append(label)
        dataset.append(row)
    random.shuffle(dataset)
    dataset = dataset[:int(len(dataset)*rate)]
    testset = dataset[int(len(dataset)*rate):]
    return dataset, testset,labels


# 计算信息熵
def jisuanEnt(dataset):
    numEntries = len(dataset)
    labelCounts = {}
    # 给所有可能分类创建字典
    for featVec in dataset:
        currentlabel = featVec[-1]
        if currentlabel not in labelCounts.keys():
            labelCounts[currentlabel] = 0
        labelCounts[currentlabel] += 1
    Ent = 0.0
    for key in labelCounts:
        p = float(labelCounts[key]) / numEntries
        Ent = Ent - p * log(p, 2)  # 以2为底求对数
    return Ent


# 划分数据集
def splitdataset(dataset, axis, value):
    retdataset = []  # 创建返回的数据集列表
    for featVec in dataset:  # 抽取符合划分特征的值
        if featVec[axis] == value:
            reducedfeatVec = featVec[:axis]  # 去掉axis特征
            reducedfeatVec.extend(featVec[axis + 1:])  # 将符合条件的特征添加到返回的数据集列表
            retdataset.append(reducedfeatVec)
    return retdataset



# ID3算法
def ID3_chooseBestFeatureToSplit(dataset):
    numFeatures = len(dataset[0]) - 1
    baseEnt = jisuanEnt(dataset)
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):  # 遍历所有特征
        # for example in dataset:
        # featList=example[i]
        featList = [example[i] for example in dataset]
        uniqueVals = set(featList)  # 将特征列表创建成为set集合，元素不可重复。创建唯一的分类标签列表
        newEnt = 0.0
        for value in uniqueVals:  # 计算每种划分方式的信息熵
            subdataset = splitdataset(dataset, i, value)
            p = len(subdataset) / float(len(dataset))
            newEnt += p * jisuanEnt(subdataset)
        infoGain = baseEnt - newEnt
        print(u"ID3中第%d个特征的信息增益为：%.3f" % (i, infoGain))
        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain  # 计算最好的信息增益
            bestFeature = i
    return bestFeature


# C4.5算法
def C45_chooseBestFeatureToSplit(dataset):
    numFeatures = len(dataset[0]) - 1
    baseEnt = jisuanEnt(dataset)
    bestInfoGain_ratio = 0.0
    bestFeature = -1
    for i in range(numFeatures):  # 遍历所有特征
        featList = [example[i] for example in dataset]
        uniqueVals = set(featList)  # 将特征列表创建成为set集合，元素不可重复。创建唯一的分类标签列表
        newEnt = 0.0
        IV = 0.0
        for value in uniqueVals:  # 计算每种划分方式的信息熵
            subdataset = splitdataset(dataset, i, value)
            p = len(subdataset) / float(len(dataset))
            newEnt += p * jisuanEnt(subdataset)
            IV = IV - p * log(p, 2)
        infoGain = baseEnt - newEnt
        if (IV == 0):  # fix the overflow bug
            continue
        infoGain_ratio = infoGain / IV  # 这个feature的infoGain_ratio
        print(u"C4.5中第%d个特征的信息增益率为：%.3f" % (i, infoGain_ratio))
        if (infoGain_ratio > bestInfoGain_ratio):  # 选择最大的gain ratio
            bestInfoGain_ratio = infoGain_ratio
            bestFeature = i  # 选择最大的gain ratio对应的feature
    return bestFeature


# CART算法
def CART_chooseBestFeatureToSplit(dataset):
    numFeatures = len(dataset[0]) - 1
    bestGini = 999999.0
    bestFeature = -1
    for i in range(numFeatures):
        featList = [example[i] for example in dataset]
        uniqueVals = set(featList)
        gini = 0.0
        for value in uniqueVals:
            subdataset = splitdataset(dataset, i, value)
            p = len(subdataset) / float(len(dataset))
            subp = len(splitdataset(subdataset, -1, '0')) / float(len(subdataset))
        gini += p * (1.0 - pow(subp, 2) - pow(1 - subp, 2))
        print(u"CART中第%d个特征的基尼值为：%.3f" % (i, gini))
        if (gini < bestGini):
            bestGini = gini
            bestFeature = i
    return bestFeature


def majorityCnt(classList):
    '''
    数据集已经处理了所有属性，但是类标签依然不是唯一的，
    此时我们需要决定如何定义该叶子节点，在这种情况下，我们通常会采用多数表决的方法决定该叶子节点的分类
    '''
    classCont = {}
    for vote in classList:
        if vote not in classCont.keys():
            classCont[vote] = 0
        classCont[vote] += 1
    sortedClassCont = sorted(classCont.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCont[0][0]


# 利用ID3算法创建决策树
def ID3_createTree(dataset, labels):
    classList = [example[-1] for example in dataset]
    if classList.count(classList[0]) == len(classList):
        # 类别完全相同，停止划分
        return classList[0]
    if len(dataset[0]) == 1:
        # 遍历完所有特征时返回出现次数最多的
        return majorityCnt(classList)
    bestFeat = ID3_chooseBestFeatureToSplit(dataset)
    bestFeatLabel = labels[bestFeat]
    print(u"此时最优索引为：" + (bestFeatLabel))
    ID3Tree = {bestFeatLabel: {}}
    del (labels[bestFeat])
    # 得到列表包括节点所有的属性值
    featValues = [example[bestFeat] for example in dataset]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        ID3Tree[bestFeatLabel][value] = ID3_createTree(splitdataset(dataset, bestFeat, value), subLabels)
    return ID3Tree


def C45_createTree(dataset, labels):
    classList = [example[-1] for example in dataset]
    if classList.count(classList[0]) == len(classList):
        # 类别完全相同，停止划分
        return classList[0]
    if len(dataset[0]) == 1:
        # 遍历完所有特征时返回出现次数最多的
        return majorityCnt(classList)
    bestFeat = C45_chooseBestFeatureToSplit(dataset)
    bestFeatLabel = labels[bestFeat]
    print(u"此时最优索引为：" + (bestFeatLabel))
    C45Tree = {bestFeatLabel: {}}
    del (labels[bestFeat])
    # 得到列表包括节点所有的属性值
    featValues = [example[bestFeat] for example in dataset]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        C45Tree[bestFeatLabel][value] = C45_createTree(splitdataset(dataset, bestFeat, value), subLabels)
    return C45Tree


def CART_createTree(dataset, labels):
    classList = [example[-1] for example in dataset]
    if classList.count(classList[0]) == len(classList):
        # 类别完全相同，停止划分
        return classList[0]
    if len(dataset[0]) == 1:
        # 遍历完所有特征时返回出现次数最多的
        return majorityCnt(classList)
    bestFeat = CART_chooseBestFeatureToSplit(dataset)
    # print(u"此时最优索引为："+str(bestFeat))
    bestFeatLabel = labels[bestFeat]
    print(u"此时最优索引为：" + (bestFeatLabel))
    CARTTree = {bestFeatLabel: {}}
    del (labels[bestFeat])
    # 得到列表包括节点所有的属性值
    featValues = [example[bestFeat] for example in dataset]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        CARTTree[bestFeatLabel][value] = CART_createTree(splitdataset(dataset, bestFeat, value), subLabels)
    return CARTTree


def classify(inputTree, featLabels, testVec):
    """
    输入：决策树，分类标签，测试数据
    输出：决策结果
    描述：跑决策树
    """
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    classLabel = '0'
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key], featLabels, testVec)
            else:
                classLabel = secondDict[key]
    return classLabel


def classifytest(inputTree, featLabels, testDataSet):
    """
    输入：决策树，分类标签，测试数据集
    输出：决策结果
    描述：跑决策树
    """
    classLabelAll = []
    for testVec in testDataSet:
        classLabelAll.append(classify(inputTree, featLabels, testVec))
    return classLabelAll

def measureAccuracy(classifyResult, testSet):
    correct = 0
    for i in range(len(testSet)):
        if classifyResult[i] == testSet[i][-1]:
            correct += 1
    accuracy = float(correct) / len(testSet)
    return accuracy


if __name__ == '__main__':
    dataset,testset,labels = load_data()
    print('dataset', dataset)
    print("---------------------------------------------")
    print(u"数据集长度", len(dataset))
    print("Ent(D):", jisuanEnt(dataset))
    print("---------------------------------------------")

    print(u"以下为首次寻找最优索引:\n")
    print(u"ID3算法的最优特征索引为:" + str(ID3_chooseBestFeatureToSplit(dataset)))
    print("--------------------------------------------------")
    print(u"C4.5算法的最优特征索引为:" + str(C45_chooseBestFeatureToSplit(dataset)))
    print("--------------------------------------------------")
    print(u"CART算法的最优特征索引为:" + str(CART_chooseBestFeatureToSplit(dataset)))
    print(u"首次寻找最优索引结束！")
    print("---------------------------------------------")

    print(u"下面开始创建相应的决策树-------")

    while (True):
        dec_tree = str(input("请选择决策树:->(1:ID3; 2:C4.5; 3:CART)|('enter q to quit!')|："))
        # ID3决策树
        if dec_tree == '1':
            labels_tmp = labels[:]  # 拷贝，createTree会改变labels
            ID3desicionTree = ID3_createTree(dataset, labels_tmp)
            print('ID3desicionTree:\n', ID3desicionTree)
            # treePlotter.createPlot(ID3desicionTree)
            treePlotter.ID3_Tree(ID3desicionTree)
            classifyResult = classifytest(ID3desicionTree, labels, testset)
            print("下面为测试数据集结果：")
            print('ID3_TestSet_classifyResult:\n', classifytest(ID3desicionTree, labels, testset))
            accuracy = measureAccuracy(classifyResult, testset)
            print('Accuracy:', accuracy)
            print("---------------------------------------------")

        # C4.5决策树
        if dec_tree == '2':
            labels_tmp = labels[:]  # 拷贝，createTree会改变labels
            C45desicionTree = C45_createTree(dataset, labels_tmp)
            print('C45desicionTree:\n', C45desicionTree)
            treePlotter.C45_Tree(C45desicionTree)
            classifyResult = classifytest(C45desicionTree, labels, testset)
            print("下面为测试数据集结果：")
            print('C4.5_TestSet_classifyResult:\n', classifytest(C45desicionTree, labels, testset))
            accuracy = measureAccuracy(classifyResult, testset)
            print('Accuracy:', accuracy)
            print("---------------------------------------------")

        # CART决策树
        if dec_tree == '3':
            labels_tmp = labels[:]  # 拷贝，createTree会改变labels
            CARTdesicionTree = CART_createTree(dataset, labels_tmp)
            print('CARTdesicionTree:\n', CARTdesicionTree)
            treePlotter.CART_Tree(CARTdesicionTree)
            classifyResult = classifytest(CARTdesicionTree, labels, testset)
            print("下面为测试数据集结果：")
            print('CART_TestSet_classifyResult:\n', classifytest(CARTdesicionTree, labels, testset))
            accuracy = measureAccuracy(classifyResult, testset)
            print('Accuracy:', accuracy)
        if dec_tree == 'q':
            break
