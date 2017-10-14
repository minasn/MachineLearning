import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets,decomposition,manifold

def load_data():
    iris=datasets.load_iris()
    return iris.data,iris.target

def test_PCA(*data):
    x,y=data
    pca=decomposition.PCA(n_components=None)
    pca.fit(x)
    print('explained variance ratio:%s'%str(pca.explained_variance_ratio_))

def plot_PCA(*data):
    x,y=data
    pca=decomposition.PCA(n_components=2)
    pca.fit(x)
    x_r=pca.transform(x)

    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)
    colors=((1,0,0),(0,1,0),(0,0,1),(0.5,0.5,0),(0,0.5,0.5),(0.5,0,0.5),(0.4,0.6,0),(0.6,0.4,0),(0,0.6,0.4),(0.5,0.3,0.2))
    for label,color in zip(np.unique(y),colors):
        position=y==label
        ax.scatter(x_r[position,0],x_r[position,1],label="target=%d"%label,color=color)

    ax.set_xlabel('x[0]')
    ax.set_ylabel('y[0]')
    ax.legend(loc="best")
    ax.set_title('PCA')
    plt.show()

x,y=load_data()
plot_PCA(x,y)