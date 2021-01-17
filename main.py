import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
import graphviz

training_data = pd.read_csv("hw2-decision-tree-input.txt", sep=",")
traindata = training_data.values
print(traindata)
training_data.head()
type(training_data)
traindata[0]
train = traindata[:, 1:len(traindata[0])]
print(train)
test = traindata[:, 5]
print(test)
clf = tree.DecisionTreeClassifier(criterion='entropy')
print(clf)
clf.fit(train, test)
play_feature_E = 'chest pain', 'male', 'smoke', 'drink', 'exercise'
play_class = 'yes', 'no'
dot_data = tree.export_graphviz(clf, out_file=None)
graph = graphviz.Source(dot_data)
graph.render("test1")
dot_data2 = tree.export_graphviz(clf, out_file=None, feature_names=play_feature_E, class_names=play_class,
                                filled=True, rounded=True, special_characters=True)
graph = graphviz.Source(dot_data2)
print(dot_data2)
graph.render("test2")

