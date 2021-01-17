import pandas as pd
import math
#导入数据
training_data = pd.read_csv("hw2-decision-tree-input.txt", sep=",")
training_data.head()
# print(training_data.columns.values[0:5]);  #查看列名
#print(training_data.shape[1]); #列数-----特征+target
#type(training_data.shape[0])
# print(training_data.shape[0]); #行数-----数据数量
train = training_data[training_data.columns.values[0:training_data.shape[1]-1]]
# print(train)
train_label = training_data[training_data.columns.values[training_data.shape[1]-1:training_data.shape[1]]]
# print(train_label.values)
rootEntrogy = 0.000;

# print(training_data.columns.values[0:training_data.shape[1]-1]);

firstright = [];
firstleft = [];
firstOne = 0.0;
firstZero = 0.0;
# for countOfData in range(training_data.shape[0]):
#     print(countOfData)
#计算root熵（root）
featureroot=training_data.columns.values[5];
train_label_root=train_label[featureroot];
for countOfData in range(training_data.shape[0]):
    if(train_label_root[countOfData]==1):
        firstOne=firstOne+1;
        firstright.append(countOfData);
    else:
        firstZero=firstZero+1;
        firstleft.append(countOfData);
rootEntrogy=0.0-(float(firstOne)/float(training_data.shape[0])*math.log(float(firstOne)/float(training_data.shape[0]))/math.log(2)+float(firstZero)/float(training_data.shape[0])*math.log(float(firstZero)/float(training_data.shape[0]))/math.log(2));
print("rootEntrogy:",rootEntrogy)


#通过计算每个feature的熵来确定Gain最大
maxGainofEntrogy=0.0;
for feature in training_data.columns.values[0:training_data.shape[1]-1]:
        train_second_feature = train[feature];
        # leaf node
        feature_one = 0.0;
        feature_zero = 0.0;
        second_left = [];
        second_right = [];
        # 利用特征分裂得到的数量和位置
        for dis_num in range(training_data.shape[0]):
            if (train_second_feature[dis_num] == 1):
                feature_one = feature_one + 1;
                second_left.append(dis_num);
            else:
                feature_zero = feature_zero + 1;
                second_right.append(dis_num);
        # 特征分了之前确定label
        feature_left_one=0;
        feature_left_zore = 0;
        for left_num in  second_left:
            if(train_label.values[left_num][0]==1):
                feature_left_one=feature_left_one+1;
            else:
                feature_left_zore=feature_left_zore+1;
        # entrogy_feature_left=0.0;
        if(feature_left_one==0):
            entrogy_feature_left = float(len(second_left)) / float(training_data.shape[0]) * ( float(feature_left_zore) / float(
                len(second_left)) * math.log(float(feature_left_zore) / float(len(second_left))) / math.log(2))
        if (feature_left_zore == 0):
            entrogy_feature_left = float(len(second_left)) / float(training_data.shape[0]) * (
                    float(feature_left_one) / float(len(second_left)) * math.log(
                float(feature_left_one) / float(len(second_left))) / math.log(2))
        if (feature_left_one > 0 and feature_left_zore > 0):
           entrogy_feature_left = float(len(second_left)) / float(training_data.shape[0]) * (
                    float(feature_left_one) / float(len(second_left)) * math.log(
                float(feature_left_one) / float(len(second_left))) / math.log(2) + float(feature_left_zore) / float(
                len(second_left)) * math.log(float(feature_left_zore) / float(len(second_left))) / math.log(2))
        # print(entrogy_feature_left)
        # right node
        feature_right_one=0;
        feature_right_zore = 0;
        for right_num in second_right:
            if(train_label.values[right_num][0]==1):
                feature_right_one=feature_right_one+1;
            else:
                feature_right_zore=feature_right_zore+1;
        # entrogy_feature_right = 0.0;
        if (feature_right_one == 0):
            entrogy_feature_right = float(len(second_right)) / float(training_data.shape[0]) * (
                        float(feature_right_zore) / float(
                    len(second_right)) * math.log(float(feature_right_zore) / float(len(second_right))) / math.log(2))
        if (feature_right_zore == 0):
            entrogy_feature_right = float(len(second_right)) / float(training_data.shape[0]) * (
                    float(feature_right_one) / float(len(second_right)) * math.log(
                float(feature_right_one) / float(len(second_right))) / math.log(2))
        if (feature_right_one != 0 and feature_right_zore != 0):
            entrogy_feature_right = float(len(second_right)) / float(training_data.shape[0]) * (
                    float(feature_right_one) / float(len(second_right)) * math.log(
                float(feature_right_one) / float(len(second_right))) / math.log(2) + float(feature_right_zore) / float(
                len(second_right)) * math.log(float(feature_right_zore) / float(len(second_right))) / math.log(2))
        # print(entrogy_feature_right)
        Gain=rootEntrogy-(0.0-entrogy_feature_left-entrogy_feature_right);
        if(Gain>maxGainofEntrogy):
            maxGainofEntrogy=Gain;
            rootFeature=feature;
        print(feature)
        print("Gain:",Gain)
print("maxGainofEntrogy:", maxGainofEntrogy)
print("rootFeature:",rootFeature)

#计算左叶的熵
train_left_node=train[rootFeature];
left_node_One_one=0;
left_node_One_zero=0;
left_leaf_num=0;
left_node_maxGainofEntrogy=0.0;
for countOfData in range(training_data.shape[0]):
    if(train_left_node[countOfData]==1 and train_label_root[countOfData]==1 ):
        left_node_One_one=left_node_One_one+1;
        left_leaf_num=left_leaf_num+1;
    elif(train_left_node[countOfData]==1 and train_label_root[countOfData]==0):
        left_node_One_zero=left_node_One_zero+1;
        left_leaf_num=left_leaf_num+1;
if(left_node_One_one==0):
    left_leaf_Entrogy = 0.0 - (float(left_node_One_zero) / float(
        left_leaf_num) * math.log(float(left_node_One_zero) / float(left_leaf_num)) / math.log(2));
elif(left_node_One_zero==0):
    left_leaf_Entrogy = 0.0 - (float(left_node_One_one) / float(left_leaf_num) * math.log(
        float(left_node_One_one) / float(left_leaf_num)) / math.log(2) );
else:
    left_leaf_Entrogy=0.0-(float(left_node_One_one)/float(left_leaf_num)*math.log(float(left_node_One_one)/float(left_leaf_num))/math.log(2)+float(left_node_One_zero)/float(left_leaf_num)*math.log(float(left_node_One_zero)/float(left_leaf_num))/math.log(2));
print("left_leaf_num:",left_leaf_num)
print("left_leaf_Entrogy:",left_leaf_Entrogy)

#计算左叶的二次分裂
if(left_node_One_one!=0 and left_node_One_zero!=0):
    train_left_node = train[rootFeature];
    for feature in training_data.columns.values[0:training_data.shape[1]-1]:
     if(feature!=rootFeature):
         train_second_feature = train[feature];
         feature_one = 0.0;
         feature_zero = 0.0;
         second_left = [];
         second_right = [];
         # 利用特征分裂得到的数量和位置
         for dis_num in range(training_data.shape[0]):
             if ( train_label.values[dis_num][0] == 1 and train_left_node[dis_num]==1):
                 feature_one = feature_one + 1;
                 second_left.append(dis_num);
             elif(train_label.values[dis_num][0] == 0 and train_left_node[dis_num]==1):
                 feature_zero = feature_zero + 1;
                 second_right.append(dis_num);
         # 特征分了之前确定label
         feature_left_one = 0;
         feature_left_zore = 0;
         for left_num in second_left:
             if (train_second_feature[left_num] == 1):
                 feature_left_one = feature_left_one + 1;
             else:
                 feature_left_zore = feature_left_zore + 1;
         # entrogy_feature_left=0.0;
         if (feature_left_one == 0):
             entrogy_feature_left = float(len(second_left)) / float(len(second_left)+len(second_right)) * (
                         float(feature_left_zore) / float(
                     len(second_left)) * math.log(float(feature_left_zore) / float(len(second_left))) / math.log(2))
         if (feature_left_zore == 0):
             entrogy_feature_left = float(len(second_left)) / float(len(second_left)+len(second_right)) * (
                     float(feature_left_one) / float(len(second_left)) * math.log(
                 float(feature_left_one) / float(len(second_left))) / math.log(2))
         if (feature_left_one > 0 and feature_left_zore > 0):
             entrogy_feature_left = float(len(second_left)) / float(len(second_left)+len(second_right)) * (
                     float(feature_left_one) / float(len(second_left)) * math.log(
                 float(feature_left_one) / float(len(second_left))) / math.log(2) + float(feature_left_zore) / float(
                 len(second_left)) * math.log(float(feature_left_zore) / float(len(second_left))) / math.log(2))
         # print(entrogy_feature_left)
         # right node
         feature_right_one = 0;
         feature_right_zore = 0;
         for right_num in second_right:
             if (train_label.values[right_num][0] == 1):
                 feature_right_one = feature_right_one + 1;
             else:
                 feature_right_zore = feature_right_zore + 1;
         # entrogy_feature_right = 0.0;
         if (feature_right_one == 0):
             entrogy_feature_right = float(len(second_right)) / float(len(second_left)+len(second_right)) * (
                     float(feature_right_zore) / float(
                 len(second_right)) * math.log(float(feature_right_zore) / float(len(second_right))) / math.log(2))
         if (feature_right_zore == 0):
             entrogy_feature_right = float(len(second_right)) / float(len(second_left)+len(second_right)) * (
                     float(feature_right_one) / float(len(second_right)) * math.log(
                 float(feature_right_one) / float(len(second_right))) / math.log(2))
         if (feature_right_one != 0 and feature_right_zore != 0):
             entrogy_feature_right = float(len(second_right)) / float(len(second_left)+len(second_right)) * (
                     float(feature_right_one) / float(len(second_right)) * math.log(
                 float(feature_right_one) / float(len(second_right))) / math.log(2) + float(feature_right_zore) / float(
                 len(second_right)) * math.log(float(feature_right_zore) / float(len(second_right))) / math.log(2))
         # print(entrogy_feature_right)
         left_node_Gain = left_leaf_Entrogy - (0.0 - len(second_left)/(len(second_left)+len(second_right))*entrogy_feature_left - len(second_right)/(len(second_left)+len(second_right))*entrogy_feature_right);
         if (left_node_Gain > left_node_maxGainofEntrogy):
             left_node_maxGainofEntrogy = left_node_Gain;
             left_node_Feature = feature;
         print(feature)
         print("feature_left_one:",feature_left_one)
         print("feature_left_zore:", feature_left_zore)
         print("feature_right_one:",feature_right_one)
         print("feature_right_zore:", feature_right_zore)
         print("len(second_left)+len(second_right):",len(second_left) + len(second_right))
    print("left_node_maxGainofEntrogy:",left_node_maxGainofEntrogy)
    print("left_node_Feature:",left_node_Feature)



#计算右叶的熵
train_right_node=train[rootFeature];
right_node_One_one=0;
right_node_One_zero=0;
right_leaf_num=0;
for countOfData in range(training_data.shape[0]):
    if(train_right_node[countOfData]==0 and train_label_root[countOfData]==1 ):
        right_node_One_one=right_node_One_one+1;
        right_leaf_num=right_leaf_num+1;
    elif(train_right_node[countOfData]==0 and train_label_root[countOfData]==0):
        right_node_One_zero=right_node_One_zero+1;
        right_leaf_num=right_leaf_num+1;

if(right_node_One_one==0):
    right_leaf_Entrogy = 0.0 - ( float(right_node_One_zero) / float(
        right_leaf_num) * math.log(float(right_node_One_zero) / float(right_leaf_num)) / math.log(2));
elif(right_node_One_zero==0):
    right_leaf_Entrogy = 0.0 - (float(right_node_One_one) / float(right_leaf_num) * math.log(
        float(right_node_One_one) / float(right_leaf_num)) / math.log(2));
else:
    right_leaf_Entrogy=0.0-(float(right_node_One_one)/float(right_leaf_num)*math.log(float(right_node_One_one)/float(right_leaf_num))/math.log(2)+float(right_node_One_zero)/float(right_leaf_num)*math.log(float(right_node_One_zero)/float(right_leaf_num))/math.log(2));
print("right_leaf_num",right_leaf_num)
print("right_leaf_Entrogy",right_leaf_Entrogy)

right_node_maxGainofEntrogy=0.0
if(right_node_One_one!=0 and right_node_One_zero!=0):
    train_right_node = train[rootFeature];
    for feature in training_data.columns.values[0:training_data.shape[1]-1]:
     if(feature!=rootFeature):
         train_second_feature = train[feature];
         feature_one = 0.0;
         feature_zero = 0.0;
         second_left = [];
         second_right = [];
         # 利用特征分裂得到的数量和位置
         for dis_num in range(training_data.shape[0]):
             if ( train_label.values[dis_num][0] == 1 and train_right_node[dis_num]==0):
                 feature_one = feature_one + 1;
                 second_left.append(dis_num);
             elif(train_label.values[dis_num][0] == 0 and train_right_node[dis_num]==0):
                 feature_zero = feature_zero + 1;
                 second_right.append(dis_num);
         # 特征分了之前确定label
         feature_left_one = 0;
         feature_left_zore = 0;
         for left_num in second_left:
             if (train_second_feature[dis_num] == 1):
                 feature_left_one = feature_left_one + 1;
             else:
                 feature_left_zore = feature_left_zore + 1;
         # entrogy_feature_left=0.0;
         if (feature_left_one == 0):
             entrogy_feature_left = float(len(second_left)) / float(len(second_left)+len(second_right)) * (
                         float(feature_left_zore) / float(
                     len(second_left)) * math.log(float(feature_left_zore) / float(len(second_left))) / math.log(2))
         if (feature_left_zore == 0):
             entrogy_feature_left = float(len(second_left)) / float(len(second_left)+len(second_right)) * (
                     float(feature_left_one) / float(len(second_left)) * math.log(
                 float(feature_left_one) / float(len(second_left))) / math.log(2))
         if (feature_left_one > 0 and feature_left_zore > 0):
             entrogy_feature_left = float(len(second_left)) / float(len(second_left)+len(second_right)) * (
                     float(feature_left_one) / float(len(second_left)) * math.log(
                 float(feature_left_one) / float(len(second_left))) / math.log(2) + float(feature_left_zore) / float(
                 len(second_left)) * math.log(float(feature_left_zore) / float(len(second_left))) / math.log(2))
         # print(entrogy_feature_left)
         # right node
         feature_right_one = 0;
         feature_right_zore = 0;
         for right_num in second_right:
             if (train_label.values[right_num][0] == 1):
                 feature_right_one = feature_right_one + 1;
             else:
                 feature_right_zore = feature_right_zore + 1;
         # entrogy_feature_right = 0.0;
         if (feature_right_one == 0):
             entrogy_feature_right = float(len(second_right)) / float(len(second_left)+len(second_right)) * (
                     float(feature_right_zore) / float(
                 len(second_right)) * math.log(float(feature_right_zore) / float(len(second_right))) / math.log(2))
         if (feature_right_zore == 0):
             entrogy_feature_right = float(len(second_right)) / float(len(second_left)+len(second_right)) * (
                     float(feature_right_one) / float(len(second_right)) * math.log(
                 float(feature_right_one) / float(len(second_right))) / math.log(2))
         if (feature_right_one != 0 and feature_right_zore != 0):
             entrogy_feature_right = float(len(second_right)) / float(len(second_left)+len(second_right)) * (
                     float(feature_right_one) / float(len(second_right)) * math.log(
                 float(feature_right_one) / float(len(second_right))) / math.log(2) + float(feature_right_zore) / float(
                 len(second_right)) * math.log(float(feature_right_zore) / float(len(second_right))) / math.log(2))
         # print(entrogy_feature_right)
         right_node_Gain = right_leaf_Entrogy - (0.0 - len(second_left)/(len(second_left)+len(second_right))*entrogy_feature_left - len(second_right)/(len(second_left)+len(second_right))*entrogy_feature_right);
         if (right_node_Gain > right_node_maxGainofEntrogy):
             right_node_maxGainofEntrogy = right_node_Gain;
             right_node_Feature = feature;
         # print(feature)
     print("len(second_left)+len(second_right):", len(second_left) + len(second_right))
     print("right_node_maxGainofEntrogy:"+right_node_maxGainofEntrogy)
     print("right_node_Feature:"+right_node_Feature)

#left  to corresponding 1
#right  to corresponding 0
if(right_node_One_one!=0 and right_node_One_zero!=0 and left_node_One_one!=0 and left_node_One_zero!=0):
    msg="The root node is attribute ", rootFeature, ". Its left edge has label 0", ". Its right edge has label 1", ". Its left child node’s attribute is ", right_node_Feature,". Its right child node’s attribute is ", left_node_Feature, ". ";
    print("The root node is attribute ", rootFeature, ". Its left edge has label 0"
          ". Its right edge has label 1", ". Its left child node’s attribute is ", right_node_Feature,
          ". Its right child node’s attribute is ", left_node_Feature, ". ")

if(right_node_One_one==0 or right_node_One_zero==0):
    msg='The root node is attribute ', rootFeature, ". Its left edge has label 0",". Its right edge has label 1", ". Its left child node’s attribute is no heart attack",". Its right child node’s attribute is ", left_node_Feature, ". ";
    print("The root node is attribute ", rootFeature, ". Its left edge has label 0",
          ". Its right edge has label 1", ". Its left child node’s attribute is no heart attack",
          ". Its right child node’s attribute is ", left_node_Feature, ". ")

if(left_node_One_one==0 or left_node_One_zero==0):
    msg="The root node is attribute ", rootFeature, ". Its left edge has label 0", ". Its right edge has label 1",  ". Its left child node’s attribute is ", right_node_Feature,". Its right child node’s attribute is heart attack. ";
    print("The root node is attribute ", rootFeature, ". Its left edge has label 0",
          ". Its right edge has label 1", ". Its left child node’s attribute is ", right_node_Feature,
          ". Its right child node’s attribute is heart attack. ")

msgOutput=""
for little_tuple in msg:
    msgOutput = msgOutput + str(little_tuple);
print(msgOutput)
file = open("tree_model.txt", 'w')
file.write(msgOutput)