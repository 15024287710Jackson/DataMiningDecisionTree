import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
import graphviz

training_data = pd.read_csv("hw2-decision-tree-input.txt", sep=",")
training_data.head()
train = training_data[['chest pain', 'male', 'smoke', 'drink', 'exercise']]
print(train)
trainlabel = training_data[['heart attack']]
print(trainlabel)
clf = tree.DecisionTreeClassifier(criterion='entropy')
print(clf)
model=clf.fit(train, trainlabel)
print(model)
dot_data = tree.export_graphviz(model, out_file=None)
graph = graphviz.Source(dot_data)
graph.render("hw2-decision-tree")

