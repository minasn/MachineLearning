import matplotlib.pyplot as plt
import math
import numpy as np

x=np.linspace(0,1,50)
y=[]
for i in x:
    if(i==0 or i==1):
        y.append(0)
    else:
        y.append(-i*math.log(i,2)-(1-i)*math.log(1-i,2))
plt.plot(x,y)
plt.show()