import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets,linear_model,discriminant_analysis,cross_validation

def load_data():
    diabetes=datasets.load_diabetes()
    return cross_validation.train_test_split(diabetes.data,diabetes.target,test_size=.25,random_state=0)

def test_LinearRegression(*data):
    x_train,x_test,y_train,y_test=data
    regr=linear_model.LinearRegression()
    regr.fit(x_train,y_train)
    print('Coefficients:%s,intercept %.2f'%(regr.coef_,regr.intercept_))
    print('Resdidual sum of squares:%.2f'%np.mean((regr.predict(x_test)-y_test)**2))
    print('Score:%.2f'%regr.score(x_test,y_test))

x_train,x_test,y_train,y_test=load_data()
test_LinearRegression(x_train,x_test,y_train,y_test)
