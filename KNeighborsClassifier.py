import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors,datasets,cross_validation

def load_classification_data():
    digits=datasets.load_digits()
    xtrain=digits.data
    ytrain=digits.target
    return cross_validation.train_test_split(xtrain,ytrain,test_size=0.25,random_state=0,stratify=ytrain)

def create_regression_data(n):
    x=5*np.random.rand(n,1)
    y=np.sin(x).ravel()
    y[::5]+=1*(0.5-np.random.rand(int(n/5)))
    return cross_validation.train_test_split(x,y,test_size=0.25,random_state=0)

def test_KNeighborsClassifier(*data):
    xtrain,xtest,ytrain,ytest=data
    clf=neighbors.KNeighborsClassifier()
    clf.fit(xtrain,ytrain)
    print("training score:%f"%(clf.score(xtrain,ytrain)))
    print("testing score:%f"%(clf.score(xtest,ytest)))


def test_KNeighborsClassifier_k_w(*data):
    xtrain,xtest,ytrain,yrest=data
    ks=np.linspace(1,ytrian.size,num=100,endpoint=False,dtype='int')
    weights=['uniform','distance']

    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)
    for weight in weights:
        trainingscores=[]
        testingscores=[]
        for k in ks:
            clf=neighbors.KNeighborsClassifier(weights=weight,n_neighbors=k)
            clf.fit(xtrain,ytrian)
            trainingscores.append(clf.score(xtrain,ytrain))
            testingscores.append(clf.score(xtest,ytest))
        ax.plot(ks,testingscores,label='testing scores:weight=%s'%(weight))
        ax.plot(ks,trainingscores,label="training scores:weight=%s"%(weight))
    ax.legend(loc='best')
    ax.set_xlabel('k')
    ax.set_ylabel('score')
    ax.set_ylim(0,1.05)
    plt.show()

xtrain,xtest,ytrian,ytest=load_classification_data()
test_KNeighborsClassifier_k_w(xtrain,xtest,ytrian,ytest)