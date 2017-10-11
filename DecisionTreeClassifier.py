import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn import cross_validation
from sklearn.tree import export_graphviz

def load_data():
    iris=datasets.load_iris()
    xtrain=iris.data
    ytrain=iris.target
    return cross_validation.train_test_split(xtrain,ytrain,test_size=0.25,random_state=0,stratify=ytrain)

def test_DecisionTreeClassifier(*data):
    xtrain,xtest,ytrain,ytest=data
    clf=DecisionTreeClassifier()
    clf.fit(xtrain,ytrain)

    print("training score:%f"%(clf.score(xtrain,ytrain)))
    print("testing score:%f"%(clf.score(xtest,ytest)))

def test_DecisionTreeClassifier_criterion(*data):
    xtrain,xtest,ytrain,ytest=data
    criterions=['gini','entropy']
    for criterion in criterions:
        clf=DecisionTreeClassifier(criterion=criterion)
        clf.fit(xtrain,ytrain)
        print("criterion:%s"%criterion)
        print("training score:%f"%(clf.score(xtrain,ytrain)))
        print("testing score:%f"%(clf.score(xtest,ytest)))

xtrain,xtest,ytrain,ytest=load_data()
clf=DecisionTreeClassifier()
clf.fit(xtrain,ytrain)
export_graphviz(clf,"D:/o")