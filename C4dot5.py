import re
from scipy.sparse import lil_matrix,linalg
from scipy import linalg as la
import numpy as np
import math
from sklearn.decomposition import PCA

trainlist=[]
trainlabels=[]
testlist=[]
testlabels=[]
index=[]
#提取训练集doc名称
with open('e:\\courses\\knowledgeanalysis\\regression\\train.res','r',encoding='UTF-8')as f:
	s=[]
	queryid=201
	line=f.readline()
	while line:		
		temp=re.split('\s+',line.strip())
		if temp[0]==str(queryid):
			s.append(temp[2])#article_id
			line=f.readline()
		else:
			trainlist.append(s[:200]+s[-200:])
			s=[]
			queryid+=1
	trainlist.append(s)
	
	label=[]
	for i in range(200):
		label.append(1)
	for i in range(200):
		label.append(0)
	for i in range(50):
		trainlabels.append(label)

#提取测试集doc名称
with open('e:\\courses\\knowledgeanalysis\\regression\\test.res','r',encoding='UTF-8')as f:
	s=[]
	score={}
	tempindex=[]
	queryid=201
	line=f.readline()
	while line:
		temp=re.split('\s+',line.strip())
		if temp[0]==str(queryid):
			s.append(temp[2])
			tempindex.append(temp[3])
			score[temp[2]]=float(temp[4])#分数
			line=f.readline()
		else:
			testlist.append(s[:50]+s[-50:])
			s=[]
			index.append(tempindex[:50]+tempindex[-50:])
			tempindex=[]
			queryid+=1
	testlist.append(s)
	index.append(tempindex)

	label=[]
	for i in range(50):
		label.append(1)
	for i in range(50):
		label.append(0)
	for i in range(50):
		testlabels.append(label)


def splitdataset(dataset,col,value,dataclass):
	subdataset=[]
	subclass=[]
	for i in range(len(dataset)):
		if dataset[i][col]==value:
			subdataset.append(dataset[i])
			subclass.append(dataclass[i])
	return subdataset,subclass


def createtree(dataset,dataclass,labels,threshold=0):
	classlist=[example for example in dataclass]
	if classlist.count(classlist[0])==len(classlist):
		#类别相同，停止划分
		return classlist[0]
	class0P=classlist.count(classlist[0])/len(classlist)
	class1P=1-class0P
	HD=-(class0P*math.log(class0P,2)+class1P*math.log(class1P,2))
	maxg=0
	feature=-1
	for label in labels:
		f=np.array(([[0,0],[0,0]]))
		for i in range(len(dataset)):
			if dataset[i][-1]>0:
				f[1,int(dataset[i][label])]+=1
			else:
				f[0,int(dataset[i][label])]+=1
		f0cn=f[0,0]
		f0cp=f[0,1]
		f1cn=f[1,0]
		f1cp=f[1,1]
		f0=f0cn+f0cp
		f1=f1cn+f1cp
		if f0==0:
			temp11p=0
			temp12p=0
		else:
			temp11p=f0cn/f0
			temp12p=f0cp/f0
		if f1==0:
			temp21p=0
			temp22p=0
		else:				
			temp21p=f1cn/f1
			temp22p=f1cp/f1
		if temp11p!=0:
			temp11p=temp11p*math.log(temp11p,2)
		if temp12p!=0:
			temp12p=temp12p*math.log(temp12p,2)
		if temp21p!=0:
			temp21p=temp21p*math.log(temp21p,2)
		if temp22p!=0:
			temp22p=temp22p*math.log(temp22p,2)
		HDX=f0/i*(-temp11p-temp12p)+f1/i*(-temp21p-temp22p)
		g=(HD-HDX)/HD
		if g>maxg:
			maxg=g
			feature=label
	if maxg>=threshold:
		labels.remove(feature)
		mytree={feature:{}}
		featurevalue=[example[feature] for example in dataset]
		uniquevalues=set(featurevalue)
		for value in uniquevalues:
			sublabels=labels[:]
			subdataset,subclass=splitdataset(dataset,feature,value,dataclass)
			mytree[feature][value]=createtree(subdataset,subclass,sublabels,threshold)
	else:
		classdict={}
		for example in classlist:
			if example in classdict:
				classdict[example]+=1
			else:
				classdict[example]=1
		maxclassnum=0
		maxclass=classlist[0]
		for key in list(classdict.keys()):
			if classdict[key]>maxclassnum:
				maxclassnum=classdict[key]
				maxclass=key
		return maxclass
	return mytree



def classify(inputtree,labels,data):
	first=list(inputtree.keys())[0]
	seconddict=inputtree[first]
	index=labels.index(first)
	classlabel=0
	for key in seconddict.keys():
		if data[index]==key:
			if type(seconddict[key]).__name__=='dict':
				classlabel=classify(seconddict[key],labels,data)
			else:
				classlabel=seconddict[key]
	return classlabel



def classifyAll(inputtree,labels,testdata):
	print('classifyall')
	classlabels=[]
	for data in testdata:
		classlabels.append(classify(inputtree,labels,data))
	return classlabels



for querynum in range(50):
	#词表大小
	word2int={}
	count=0
	trainvectors=[]
	for docnum in range(400):
		vector=[]
		with open('E:\\courses\\knowledgeanalysis\\cleanerdata\\titlevector\\%s\\%s.txt'%(201+querynum,trainlist[querynum][docnum]),'r',encoding='UTF-8')as f:
			line=f.readline()
			while line:
				if int(line) not in word2int:
					word2int[int(line)]=count
					vector.append(count)
					count+=1
				else:
					vector.append(word2int[int(line)])
				line=f.readline()
			trainvectors.append(vector)


	testvectors=[]
	for docnum in range(100):
		vector=[]
		with open('E:\\courses\\knowledgeanalysis\\cleanerdata\\titlevector\\%s\\%s.txt'%(201+querynum,testlist[querynum][docnum]),'r',encoding='UTF-8')as f:
			line=f.readline()
			while line:
				if int(line) in word2int:
					vector.append(word2int[int(line)])
				line=f.readline()
			testvectors.append(vector)

	samplemat=np.zeros((400,len(word2int)))
	samplelabels=list(range(len(word2int)))
	testmat=np.zeros((100,len(word2int)))
	
	for i in range(400):
		for col in trainvectors[i]:
			samplemat[i,col]=1
	
	for i in range(100):
		for col in testvectors[i]:
			testmat[i,col]=1


	print(samplemat.shape)
	labels=samplelabels[:]
	decisiontree=createtree(samplemat.tolist(),trainlabels[querynum],labels,0.97)
	print(decisiontree)
	result=classifyAll(decisiontree,samplelabels,testmat.tolist())
	nmae=0
	s=''
	for i in range(100):
		s=s+'%s Q0 %s %s %s Hiemstra_LM0.15_Bo1bfree_d_3_t_10\n'%(querynum+201,testlist[querynum][i],index[querynum][i],result[i])
		nmae+=abs(result[i]-testlabels[querynum][i])
	with open('E:\\courses\\knowledgeanalysis\\algorithm\\decisiontree\\C4dot5.res','a',encoding='UTF-8')as f:
		f.write(s)
	print('mae=%s'%(nmae/100))
	with open('E:\\courses\\knowledgeanalysis\\algorithm\\decisiontree\\C4dot5mae.txt','a',encoding='UTF-8')as f:
		f.write('querynum:%s   mae=%s\n'%(querynum,nmae/100))