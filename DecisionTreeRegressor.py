import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn import cross_validation
import matplotlib.pyplot as plt

def creat_data(n):
    np.random.seed(0)
    x=5*np.random.rand(n,1)
    y=np.sin(x).ravel()
    noise_num=(int)(n/5)
    y[::5]+=3*(0.5-np.random.rand(noise_num))
    return cross_validation.train_test_split(x,y,test_size=0.25,random_state=1)

def test_DecisionTreeRegressor(*data):
    xtrain,xtest,ytrain,ytest=data
    regr=DecisionTreeRegressor(max_depth=1)
    regr.fit(xtrain,ytrain)
    print("train score:%f"%(regr.score(xtrain,ytrain)))
    print("test score:%f"%(regr.score(xtest,ytest)))
    #绘图
    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)
    x=np.arange(0.0,5.0,0.01)[:,np.newaxis]
    y=regr.predict(x)
    ax.scatter(xtrain,ytrain,label='train sample',c='g')
    ax.scatter(xtest,ytest,label="test sample",c='r')
    ax.plot(x,y,label='predict_value',linewidth=2,alpha=0.5)
    ax.set_xlabel('data')
    ax.set_ylabel('target')
    ax.set_title('decision tree regression')
    ax.legend(framealpha=0.5)
    plt.show()

def test_DecisionTreeRegressor_depth(*data,maxdepth):
    xtrain,xtest,ytrain,ytest=data
    depths=np.arange(1,maxdepth)
    traingscore=[]
    testscore=[]
    for depth in depths:
        regr=DecisionTreeRegressor(max_depth=depth)
        regr.fit(xtrain,ytrain)
        traingscore.append(regr.score(xtrain,ytrain))
        testscore.append(regr.score(xtest,ytest))
        #绘图
    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)
    ax.plot(depths,traingscore,label="training score")
    ax.plot(depths,testscore,label='testing score')
    ax.set_xlabel('maxdepth')
    ax.set_ylabel('score')
    ax.legend(framealpha=0.5)
    plt.show()



xtrain,xtest,ytrain,ytest=creat_data(100)
test_DecisionTreeRegressor(xtrain,xtest,ytrain,ytest)