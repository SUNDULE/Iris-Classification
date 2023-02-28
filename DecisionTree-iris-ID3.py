import numpy as np
import random

def load_data(iris_path='./Iris.csv', rate=0.8):
    label2vec = {'Iris-setosa': 0.0, 'Iris-versicolor': 1.0, 'Iris-virginica': 2.0}
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
    train_data = np.array(dataset[:int(len(dataset)*rate)], dtype=float)
    test_data = np.array(dataset[int(len(dataset)*rate):], dtype=float)
    return np.rint(train_data), np.rint(test_data)

def get_rela_entropy(dataset, feature:int):
    def get_entropy(dataset):
        label_tags = list(set(dataset[:, -1]))#就0，1，2三个标签
        label_length = len(dataset[:, -1])#总共的标签数目，这里就120个
        tmp_entropy = 0
        for label_tag in label_tags:
            tmp = sum([1 for d in dataset if d[-1]==label_tag])
            tmp_entropy += (tmp/label_length)*np.math.log(tmp/label_length, 2)
        entropy = -tmp_entropy
        return entropy

    feature_tags = list(set(dataset[:, feature]))
    sub_entropy = 0
    for feature_tag in feature_tags:
        sub_dataset = [d for d in dataset if d[feature]==feature_tag]
        sub_dataset = np.array(sub_dataset)
        tmp_entropy = get_entropy(sub_dataset)
        sub_entropy += (len(sub_dataset)/len(dataset)) * tmp_entropy
    rela_entropy = get_entropy(dataset) - sub_entropy#熵 - 条件熵
    return rela_entropy#信息增益

def select_feature(dataset, features):
    # 这段代码实现了特征选择功能，它从给定的数据集中选择一个最优特征。
    # 具体来说，函数的输入是一个数据集dataset和一个特征列表features，其中每个特征都是一个整数。
    # 函数计算dataset中每个特征的相关熵，并将这些值保存在rela_entropys列表中。然后，它返回features中具有最大相关熵的特征。
    # 这个for循环对features中的每个特征进行遍历，
    # 对于每个特征，使用get_rela_entropy函数计算它的相关熵，并将结果添加到rela_entropys列表中。
    # 注意，这里使用feature:int语法将feature强制转换为整数类型。
    # 这行代码使用max函数找到rela_entropys中的最大值，并使用index方法返回该最大值的索引。最终，函数返回features中相应的特征。
    rela_entropys = list()
    for feature in features:
        feature:int
        rela_entropy = get_rela_entropy(dataset, feature)
        rela_entropys.append(rela_entropy)
    return features[rela_entropys.index(max(rela_entropys))]



def major_label(labels):
# 这个函数实现了一个简单的投票策略，它用于选择一个数据集中出现最频繁的标签作为该数据集的标签。
# 具体来说，函数的输入是一个标签列表labels，它包含数据集中的所有标签。
# 函数计算每个标签在labels中出现的次数，并将这些值保存在tag_num列表中。
# 然后，它找到tag_num中出现最频繁的标签，并返回该标签。
    tags = list(set(labels))
    tag_num = [sum([1 for i in labels if i==label]) for label in tags]
    #这两行代码计算每个标签在labels中出现的次数。
    # 首先，list(set(labels))生成一个包含所有不同标签的列表。
    # 对于每个标签，使用列表解析式[1 for i in labels if i==label]计算标签在labels中出现的次数，
    # 然后使用sum函数将这些次数相加，得到该标签在labels中出现的总次数。这些值存储在tag_num列表中，该列表的长度等于标签的数量。
    k = tag_num.index(max(tag_num))
    return tags[k]
    #这两行代码找到tag_num中的最大值，并返回该值在tag_num中的索引。
    # 这个索引用于获取tags中相应的标签，并将其作为函数的输出返回。
    # 由于tag_num和tags是按照相同的顺序组织的，所以索引k对应的标签就是出现最频繁的标签。


def build_tree(dataset, features) -> dict:
#这个函数实现了一个基于决策树的分类器，它采用递归方式构建树。
#具体来说，函数的输入是数据集dataset和特征列表features，
# 其中dataset是一个二维NumPy数组，每行代表一个样本，每列代表一个特征。
# 数组的最后一列是标签列，它包含每个样本的标签。features是一个整数列表，它包含所有可以用于分类的特征的索引。
    labels = dataset[:, -1]
    if len(set(labels)) == 1:
        return {'label': labels[0]}
    #这段代码首先提取数据集中的所有标签，并检查它们是否都相同。
    # 如果是，则返回一个字典，该字典包含一个名为label的键，其值为标签的唯一值。
    if not len(features):
        return {'label': major_label(labels)}
    #这段代码检查特征列表是否为空。如果是，则返回一个字典，该字典包含一个名为label的键，其值为标签的多数派。
    # 这种情况可能会发生，当样本的所有特征都已经用于分类，但是它们不能使数据集的标签不同。
    best_feature = select_feature(dataset, features)
    tree = {'feature': best_feature, 'children': {}}
    #这段代码使用select_feature函数来选择最佳的特征，然后创建一个新的字典tree，
    # 该字典包含一个名为feature的键，其值为最佳特征的索引，以及一个名为children的键，
    # 其值为一个空字典。children字典将用于保存决策树中的子节点。
    feature_tags = list(set(dataset[:, best_feature]))
    for feature_tag in feature_tags:
        sub_dataset = [d for d in dataset if d[best_feature]==feature_tag]
        sub_dataset = np.array(sub_dataset)
        if len(sub_dataset) == 0:
            tree['children'][feature_tag] = {'label': major_label(labels)}
        else:
            sub_features = [i for i in features if i != best_feature]
            tree['children'][feature_tag] = build_tree(sub_dataset, sub_features)
            #这段代码对每个最佳特征的可能取值进行循环，对于每个取值，将数据集分为一个子集，并递归调用build_tree函数以构建子树。
            # 该子树将作为一个字典插入到children字典中，
            # 它的键是最佳特征的值，它的值是一个字典，它表示决策树中的一个子节点。
    return tree



def classifier(tree:dict, features_data, default):
    '''
函数首先定义了一个内部函数 classify()，用于对单个样本进行分类，它接受两个参数：

tree: dict，决策树，是一个字典，存储了决策树的信息。
sample，一个样本，是一个列表或数组，包含了所有的特征。
classify() 的实现方式是递归调用，如果当前节点不是叶子节点，就继续递归调用下一个子节点；如果当前节点是叶子节点，就返回该节点的分类标签。
然后 classifier() 对每个输入的特征数据进行分类，调用 classify() 函数，最后将所有的分类结果存储在一个列表 predict_vec 中返回。
如果某个样本无法分类，则将其预测为 default。
    '''
    def classify(tree:dict, sample):
        for k, v in tree.items():
            if k != 'feature':
                return tree['label']
            else:
                return classify(tree['children'][sample[tree['feature']]],sample)
    predict_vec = list()
    for features_sample in features_data:
        try:
            predict = classify(tree, features_sample)
        except KeyError:
            predict = default
        predict_vec.append(predict)
    return predict_vec

if __name__=="__main__":
    train_data, test_data = load_data()
    tree = build_tree(train_data, list(range(train_data.shape[1]-1)))
    #调用 build_tree 函数构建决策树，使用训练数据作为输入，
    #从特征中选择最优特征，将数据集划分成更小的数据集，并递归地构建子树，最终得到一个完整的决策树。

    #print(tree)
    test_data_labels = test_data[:, -1]
    test_data_features = test_data[:, :-1]
    default = major_label(test_data_labels)
    #调用 major_label 函数获取测试数据的多数类别作为默认值。
    predict_vec = classifier(tree, test_data_features, default)
    #调用 classifier 函数对测试数据进行分类，将得到的分类结果存储在 predict_vec 中。

    #print(predict_vec)
    accuracy = np.mean(np.array(predict_vec==test_data_labels))
    print("准确率为：",accuracy)